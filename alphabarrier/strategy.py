"""Causal Bitcoin direction classifier with an execution example."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alphabarrier.evaluation import (
    Window,
    block_bootstrap_ci,
    make_windows,
    normalize_price_features,
)
from alphabarrier.execution import (
    COMMISSION_PER_SIDE,
    CONFIDENCE_FLOOR,
    TREND_EMA_SPAN,
    build_strategy_positions,
    continuous_backtest,
)
from alphabarrier.features import (
    DOWN,
    LABEL_HORIZON_BARS,
    SIDEWAYS,
    UP,
    get_momentum_feature_columns,
)
from alphabarrier.order_flow import ORDER_FLOW_COLUMNS, build_order_flow_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_PATH = ROOT / "results" / "walk_forward.json"
BAR_HOURS = 4
MODEL_C = 0.5
CALIBRATION_MONTHS = 12
MATURITY_LAG = 13
STATE_SPANS = (3, 9, 24, 72, 240, 720)
DECISION_THRESHOLDS = tuple(np.round(np.arange(0.47, 0.5301, 0.005), 3))
LABEL_EMBARGO = pd.Timedelta(hours=BAR_HOURS * LABEL_HORIZON_BARS)
DEVELOPMENT_START = pd.Timestamp("2021-07-01")
LATER_EVALUATION_START = pd.Timestamp("2025-07-01")
LATER_EVALUATION_END = pd.Timestamp("2026-07-01")
DEVELOPMENT_SIGNAL_END = LATER_EVALUATION_START - LABEL_EMBARGO
SCREENED_CONFIGURATIONS_LOWER_BOUND = 270


@dataclass(frozen=True)
class FoldScore:
    test_start: str
    threshold: float
    calibration_directional_rows: int
    calibration_accuracy: float
    directional_rows: int
    correct: int
    accuracy: float
    causal_baseline_accuracy: float


def build_mature_outcome_features(
    target: pd.Series,
    *,
    maturity_lag: int = MATURITY_LAG,
    spans: tuple[int, ...] = STATE_SPANS,
) -> pd.DataFrame:
    """Build exponentially weighted state from fully matured outcomes."""
    if maturity_lag <= LABEL_HORIZON_BARS:
        raise ValueError("maturity_lag must exceed the 12-bar label horizon")
    if not spans or any(span < 1 for span in spans):
        raise ValueError("spans must contain positive integers")
    signed = target.map({DOWN: -1.0, SIDEWAYS: 0.0, UP: 1.0}).shift(maturity_lag)
    output = pd.DataFrame(index=target.index)
    for span in spans:
        output[f"mature_outcome_ewm_{span}"] = signed.ewm(
            span=span,
            min_periods=max(3, span // 4),
            adjust=False,
        ).mean()
    return output


def choose_decision_threshold(
    probability: np.ndarray,
    target: pd.Series,
    *,
    thresholds: tuple[float, ...] = DECISION_THRESHOLDS,
) -> tuple[float, float, int]:
    """Choose calibration accuracy, breaking ties toward a neutral threshold."""
    if len(probability) != len(target):
        raise ValueError("probability and target must have equal length")
    if not thresholds or any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be strictly between 0 and 1")
    directional = target.notna().to_numpy() & target.ne(SIDEWAYS).to_numpy()
    if not directional.any():
        return 0.5, float("nan"), 0
    truth_up = target.eq(UP).to_numpy()[directional]
    probabilities = probability[directional]
    accuracies = np.asarray(
        [np.mean((probabilities >= threshold) == truth_up) for threshold in thresholds]
    )
    best = np.flatnonzero(np.isclose(accuracies, accuracies.max()))
    chosen = min(best, key=lambda index: abs(thresholds[index] - 0.5))
    return float(thresholds[chosen]), float(accuracies[chosen]), int(directional.sum())


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index < end)]


def _training_rows(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    usable = frame.dropna(subset=features + ["target"])
    return usable.loc[usable["target"].ne(SIDEWAYS)]


def _fit(frame: pd.DataFrame, features: list[str]):
    train = _training_rows(frame, features)
    values = train[features].to_numpy()
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(
        C=MODEL_C,
        class_weight="balanced",
        max_iter=3_000,
        random_state=42,
    )
    model.fit(scaler.transform(values), train["target"].eq(UP).astype(int))
    return np.nanmedian(values, axis=0), scaler, model


def _predict(fitted, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    medians, scaler, model = fitted
    values = frame[features].to_numpy().copy()
    missing = np.where(np.isnan(values))
    values[missing] = np.take(medians, missing[1])
    return model.predict_proba(scaler.transform(values))[:, 1]


def evaluate(
    frame: pd.DataFrame, features: list[str], windows: list[Window]
) -> tuple[list[FoldScore], np.ndarray, pd.DataFrame]:
    """Fit, causally calibrate, and call every row in each test window."""
    scores: list[FoldScore] = []
    correct_parts: list[np.ndarray] = []
    signal_parts: list[pd.DataFrame] = []
    for window in windows:
        calibration_start = window.test_start - pd.DateOffset(
            months=CALIBRATION_MONTHS
        )
        calibration_end = window.test_start - LABEL_EMBARGO
        inner_train_end = calibration_start - LABEL_EMBARGO
        inner_train = _slice(frame, window.train_start, inner_train_end)
        calibration = _slice(frame, calibration_start, calibration_end)
        outer_train = _slice(frame, window.train_start, window.train_end)
        test = _slice(frame, window.test_start, window.test_end)
        if len(_training_rows(inner_train, features)) < 500 or len(test) < 50:
            continue

        calibration_probability = _predict(_fit(inner_train, features), calibration, features)
        threshold, calibration_accuracy, calibration_rows = choose_decision_threshold(
            calibration_probability, calibration["target"]
        )
        probability = _predict(_fit(outer_train, features), test, features)
        directional = (
            test["target"].notna().to_numpy()
            & test["target"].ne(SIDEWAYS).to_numpy()
        )
        truth_up = test["target"].eq(UP).to_numpy()
        predicted_up = probability >= threshold
        correct = predicted_up[directional] == truth_up[directional]

        train_directional = _training_rows(outer_train, features)
        causal_up = train_directional["target"].eq(UP).mean() >= 0.5
        baseline = truth_up[directional] == causal_up
        scores.append(
            FoldScore(
                test_start=str(window.test_start),
                threshold=threshold,
                calibration_directional_rows=calibration_rows,
                calibration_accuracy=calibration_accuracy,
                directional_rows=int(directional.sum()),
                correct=int(correct.sum()),
                accuracy=float(correct.mean()),
                causal_baseline_accuracy=float(baseline.mean()),
            )
        )
        correct_parts.append(correct)
        signal_parts.append(
            pd.DataFrame(
                {
                    "side": np.where(predicted_up, 1, -1),
                    "confidence": np.abs(probability - threshold),
                },
                index=test.index,
            )
        )
    if not scores:
        raise RuntimeError("no walk-forward windows were evaluated")
    return scores, np.concatenate(correct_parts), pd.concat(signal_parts).sort_index()


def summarize(
    scores: list[FoldScore],
    correct: np.ndarray,
    signals: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, object]:
    low, high = block_bootstrap_ci(correct)
    directional_rows = sum(score.directional_rows for score in scores)
    baseline_correct = sum(
        score.causal_baseline_accuracy * score.directional_rows for score in scores
    )
    positions = build_strategy_positions(signals, frame)
    backtesting_example = continuous_backtest(positions, frame)
    return {
        "windows": len(scores),
        "call_coverage": 1.0,
        "directional_samples": directional_rows,
        "correct": int(correct.sum()),
        "directional_accuracy": float(correct.mean()),
        "directional_accuracy_block_bootstrap_ci95": [low, high],
        "causal_majority_baseline_accuracy": baseline_correct / directional_rows,
        "window_rate_above_50_percent": float(
            np.mean([score.accuracy > 0.5 for score in scores])
        ),
        "backtesting_example": backtesting_example,
        "folds": [asdict(score) for score in scores],
    }


def run(data_dir: str | Path = DATA_DIR) -> dict[str, object]:
    data_dir = Path(data_dir)
    price = pd.read_pickle(data_dir / "cache" / "bitstamp_h12_pt1.5_sl1.5.pkl")
    momentum = get_momentum_feature_columns(price)
    normalized = normalize_price_features(price, momentum)
    order_flow = build_order_flow_features(price.index, data_dir / "market")
    outcome_state = build_mature_outcome_features(price["target"])
    frame = normalized.join(order_flow).join(outcome_state)
    features = momentum + list(ORDER_FLOW_COLUMNS) + list(outcome_state.columns)
    windows = make_windows(
        first_test=str(DEVELOPMENT_START),
        last_test=DEVELOPMENT_SIGNAL_END,
        purge_bars=LABEL_HORIZON_BARS,
    )
    scores, correct, signals = evaluate(frame, features, windows)
    result = summarize(scores, correct, signals, frame)
    return {
        "status": "development_result",
        "claim_warning": (
            "Not a holdout result and not evidence of a deployable trading strategy."
        ),
        "protocol": (
            "six-month purged walk-forward; forced direction on every bar; prior "
            "purged 12-month threshold calibration; final label-horizon embargo"
        ),
        "development_start": str(DEVELOPMENT_START),
        "development_boundary": str(LATER_EVALUATION_START),
        "development_signal_end_exclusive": str(DEVELOPMENT_SIGNAL_END),
        "later_evaluation_end_exclusive": str(LATER_EVALUATION_END),
        "screened_configurations_lower_bound": SCREENED_CONFIGURATIONS_LOWER_BOUND,
        "model": {
            "type": "balanced_logistic_regression",
            "C": MODEL_C,
            "always_call": True,
            "price_features": len(momentum),
            "order_flow_features": len(ORDER_FLOW_COLUMNS),
            "mature_outcome_features": len(outcome_state.columns),
            "outcome_lag_bars": MATURITY_LAG,
            "decision_threshold_grid": list(DECISION_THRESHOLDS),
        },
        "backtesting_example": {
            "side_mode": "long_only",
            "weak_calls": "hold_previous_model_state",
            "confidence_floor": CONFIDENCE_FLOOR,
            "trend_filter_ema_span": TREND_EMA_SPAN,
            "commission_per_side": COMMISSION_PER_SIDE,
        },
        "result": result,
    }


def main() -> None:
    output = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    RESULT_PATH.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
