"""Causal Bitcoin direction classifier with baselined execution examples."""

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
    deflated_sharpe_ratio,
    make_windows,
    normalize_price_features,
)
from alphabarrier.execution import (
    COMMISSION_PER_SIDE,
    CONFIDENCE_FLOOR,
    FEE_SENSITIVITY_GRID,
    REBALANCE_BAND,
    SIGNAL_EWMA_SPAN,
    SIZING_FLOOR,
    TREND_EMA_LADDER,
    TREND_EMA_SPAN,
    build_scaled_positions,
    build_strategy_positions,
    continuous_backtest,
    fee_sensitivity,
    position_log_returns,
    reference_baselines,
    trend_gate,
)
from alphabarrier.features import (
    DOWN,
    LABEL_HORIZON_BARS,
    SIDEWAYS,
    UP,
    get_momentum_feature_columns,
    realized_barrier_span,
)
from alphabarrier.order_flow import ORDER_FLOW_COLUMNS, build_order_flow_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_PATH = ROOT / "results" / "walk_forward.json"
BAR_HOURS = 4
BARS_PER_YEAR = 365.25 * 24 / BAR_HOURS
MODEL_C = 0.5
CALIBRATION_MONTHS = 12
MATURITY_LAG = 13
STATE_SPANS = (3, 9, 24, 72, 240, 720)
DECISION_THRESHOLDS = tuple(np.round(np.arange(0.47, 0.5301, 0.005), 3))
LABEL_EMBARGO = pd.Timedelta(hours=BAR_HOURS * LABEL_HORIZON_BARS)
STEP_MONTHS = 3
FOLD_GRID_OFFSETS = tuple(range(STEP_MONTHS))
EWMA_SPAN_GRID = (1, 6, 12, 24, 48)
REBALANCE_BAND_GRID = (0.0, 0.05, 0.10, 0.20, 0.30)
SIZING_FLOOR_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
TREND_SPAN_GRID = (90, 180, 270, 360, 540, 720, 900, 1080, 1200, 1440, 1800, 2160)
DEVELOPMENT_START = pd.Timestamp("2021-07-01")
LATER_EVALUATION_START = pd.Timestamp("2025-07-01")
LATER_EVALUATION_END = pd.Timestamp("2026-07-01")
DEVELOPMENT_SIGNAL_END = LATER_EVALUATION_START - LABEL_EMBARGO
SCREENED_CONFIGURATIONS_LOWER_BOUND = 320


@dataclass(frozen=True)
class FoldScore:
    test_start: str
    threshold: float
    logit_scale: float
    signal_span: float
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


def build_barrier_spans(frame: pd.DataFrame) -> pd.Series:
    """Bars each row's label took to resolve, measured on the cached frame.

    Forward-looking by construction, exactly like ``target``: it may only be
    read from a purged calibration window, never used as a feature.
    """
    prices = frame.rename(
        columns={
            "raw_open": "open",
            "raw_high": "high",
            "raw_low": "low",
            "raw_close": "close",
        }
    )
    return realized_barrier_span(
        prices[["open", "high", "low", "close"]], frame["raw_atr"]
    )


def measure_signal_span(
    calibration: pd.DataFrame, *, default: float = float(SIGNAL_EWMA_SPAN)
) -> float:
    """Mean bars this fold's calibration labels took to touch a barrier.

    The signal is smoothed over this rather than over ``LABEL_HORIZON_BARS``,
    which is the ceiling on that number and roughly two and a half times it.
    """
    if "barrier_span" not in calibration.columns:
        return default
    spans = calibration["barrier_span"].dropna()
    return float(spans.mean()) if len(spans) else default


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


def measure_logit_scale(probability: np.ndarray, threshold: float) -> float:
    """Spread of the calibration log-odds around the fold's own threshold.

    This is the unit the sizing map divides by, so a call one spread above the
    threshold means the same thing in every fold.
    """
    if len(probability) < 3:
        return 1.0
    offsets = _logit_array(probability) - _logit_array(np.asarray([threshold]))[0]
    scale = float(np.std(offsets, ddof=1))
    return scale if np.isfinite(scale) and scale > 1e-6 else 1.0


def _logit_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index < end)]


def _training_rows(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    usable = frame.dropna(subset=features + ["target"])
    return usable.loc[usable["target"].ne(SIDEWAYS)]


def impute_and_call(model, medians: np.ndarray, scaler: StandardScaler | None):
    """Wrap a fitted classifier as a caller that fills gaps with train medians."""

    def call(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        values = frame[features].to_numpy().copy()
        missing = np.where(np.isnan(values))
        values[missing] = np.take(medians, missing[1])
        if scaler is not None:
            values = scaler.transform(values)
        return model.predict_proba(values)[:, 1]

    return call


def fit_logistic(frame: pd.DataFrame, features: list[str]):
    """Fit the published balanced logistic regression on the directional rows."""
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
    return impute_and_call(model, np.nanmedian(values, axis=0), scaler)


def evaluate(
    frame: pd.DataFrame,
    features: list[str],
    windows: list[Window],
    *,
    fitter=fit_logistic,
) -> tuple[list[FoldScore], np.ndarray, pd.DataFrame]:
    """Fit, causally calibrate, and call every row in each test window.

    ``fitter`` takes a training slice and returns a caller, so an alternative
    model class can be screened through the identical purge, calibration and
    embargo rather than through a second copy of them.
    """
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

        calibration_probability = fitter(inner_train, features)(calibration, features)
        threshold, calibration_accuracy, calibration_rows = choose_decision_threshold(
            calibration_probability, calibration["target"]
        )
        logit_scale = measure_logit_scale(calibration_probability, threshold)
        signal_span = measure_signal_span(calibration)
        probability = fitter(outer_train, features)(test, features)
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
                logit_scale=logit_scale,
                signal_span=signal_span,
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
                    "probability": probability,
                    "threshold": threshold,
                    "logit_scale": logit_scale,
                    "signal_span": signal_span,
                },
                index=test.index,
            )
        )
    if not scores:
        raise RuntimeError("no walk-forward windows were evaluated")
    return scores, np.concatenate(correct_parts), pd.concat(signal_parts).sort_index()


def offset_windows(
    month_offset: int,
    *,
    first_test: pd.Timestamp = DEVELOPMENT_START,
    last_test: pd.Timestamp = DEVELOPMENT_SIGNAL_END,
    step_months: int = STEP_MONTHS,
) -> list[Window]:
    """The walk-forward grid shifted by an arbitrary whole number of months."""
    return make_windows(
        first_test=str(first_test + pd.DateOffset(months=month_offset)),
        last_test=last_test,
        step_months=step_months,
        purge_bars=LABEL_HORIZON_BARS,
        bar_hours=BAR_HOURS,
    )


def _sensitivity_row(
    knob: str, value: float, positions: pd.Series, frame: pd.DataFrame
) -> dict[str, float | str | None]:
    result = continuous_backtest(positions, frame)
    return {
        "knob": knob,
        "value": float(value),
        "total_return": result["total_return"],
        "sharpe": result["sharpe"],
        "mean_weight": result["mean_weight"],
        "fee_sides": result["fee_sides"],
    }


def execution_sensitivity(
    signals: pd.DataFrame, frame: pd.DataFrame
) -> list[dict[str, float | str | None]]:
    """Sweep every execution knob one at a time and report all outcomes."""
    rows = []
    for span in EWMA_SPAN_GRID:
        rows.append(
            _sensitivity_row(
                "signal_ewma_span",
                span,
                build_scaled_positions(signals, frame, ewma_span=span),
                frame,
            )
        )
    for band in REBALANCE_BAND_GRID:
        rows.append(
            _sensitivity_row(
                "rebalance_band",
                band,
                build_scaled_positions(signals, frame, rebalance_band=band),
                frame,
            )
        )
    for floor in SIZING_FLOOR_GRID:
        rows.append(
            _sensitivity_row(
                "sizing_floor",
                floor,
                build_scaled_positions(signals, frame, floor=floor),
                frame,
            )
        )
    return rows


def trend_span_sensitivity(
    signals: pd.DataFrame, frame: pd.DataFrame
) -> dict[str, object]:
    """Sweep the trend filter's lookback, against its own baseline at each span.

    The gate does most of the work in this strategy and its lookback was never
    swept, so the level of every published execution number rested on it. There
    is no plateau: single spans move the strategy across a wide range while the
    model's uplift over the gate it sits on barely moves. That gap is the part
    of the result that is about the classifier; the level is not.
    """
    rows = []
    for span in TREND_SPAN_GRID:
        strategy = continuous_backtest(
            build_scaled_positions(signals, frame, trend_spans=span), frame
        )
        baseline = continuous_backtest(
            reference_index_trend(signals.index, frame, trend_spans=span), frame
        )
        rows.append(
            {
                "trend_ema_span": span,
                "sharpe": strategy["sharpe"],
                "trend_only_sharpe": baseline["sharpe"],
                "uplift": strategy["sharpe"] - baseline["sharpe"],
                "total_return": strategy["total_return"],
                "max_drawdown": strategy["max_drawdown"],
                "fee_sides": strategy["fee_sides"],
            }
        )
    ladder = continuous_backtest(build_scaled_positions(signals, frame), frame)
    ladder_baseline = continuous_backtest(
        reference_index_trend(signals.index, frame), frame
    )
    single = np.array([row["sharpe"] for row in rows], dtype=float)
    uplift = np.array([row["uplift"] for row in rows], dtype=float)
    return {
        "note": (
            "Single-span gates, then the ladder the strategy actually uses. The "
            "spread of the single-span column is the cost of choosing one "
            "lookback; the spread of the uplift column is what the model "
            "contributes regardless of that choice."
        ),
        "single_span": rows,
        "single_span_sharpe_mean": float(single.mean()),
        "single_span_sharpe_sd": float(single.std(ddof=1)),
        "single_span_uplift_mean": float(uplift.mean()),
        "single_span_uplift_sd": float(uplift.std(ddof=1)),
        "ladder": {
            "spans": list(TREND_EMA_LADDER),
            "sharpe": ladder["sharpe"],
            "trend_only_sharpe": ladder_baseline["sharpe"],
            "uplift": ladder["sharpe"] - ladder_baseline["sharpe"],
        },
    }


def fold_grid_dispersion(
    frame: pd.DataFrame, features: list[str]
) -> dict[str, object]:
    """Re-run the whole protocol on every calendar shift of the fold grid.

    The grid start date is arbitrary, so the spread across shifts is the noise
    floor: a change smaller than it is not evidence of anything.
    """
    rows = []
    for offset in FOLD_GRID_OFFSETS:
        windows = offset_windows(offset)
        scores, correct, signals = evaluate(frame, features, windows)
        scaled = continuous_backtest(build_scaled_positions(signals, frame), frame)
        gate = continuous_backtest(build_strategy_positions(signals, frame), frame)
        trend = continuous_backtest(
            reference_index_trend(signals.index, frame), frame
        )
        rows.append(
            {
                "month_offset": offset,
                "windows": len(scores),
                "directional_accuracy": float(correct.mean()),
                "scaled_sharpe": scaled["sharpe"],
                "binary_gate_sharpe": gate["sharpe"],
                "trend_only_sharpe": trend["sharpe"],
            }
        )
    scaled = np.array([row["scaled_sharpe"] for row in rows], dtype=float)
    gate = np.array([row["binary_gate_sharpe"] for row in rows], dtype=float)
    trend = np.array([row["trend_only_sharpe"] for row in rows], dtype=float)
    return {
        "note": (
            "Spread across arbitrary calendar shifts of the fold grid. Read every "
            "reported difference against it."
        ),
        "offsets": rows,
        "scaled_sharpe_mean": float(scaled.mean()),
        "scaled_sharpe_sd": float(scaled.std(ddof=1)),
        "binary_gate_sharpe_mean": float(gate.mean()),
        "binary_gate_sharpe_sd": float(gate.std(ddof=1)),
        "trend_only_sharpe_mean": float(trend.mean()),
        "trend_only_sharpe_sd": float(trend.std(ddof=1)),
        "scaled_minus_trend_mean": float((scaled - trend).mean()),
        "binary_gate_minus_trend_mean": float((gate - trend).mean()),
    }


def reference_index_trend(
    index: pd.DatetimeIndex,
    frame: pd.DataFrame,
    *,
    trend_spans: int | tuple[int, ...] = TREND_EMA_LADDER,
) -> pd.Series:
    """The trend-filter-only baseline path restricted to the signal index."""
    return trend_gate(frame, trend_spans).reindex(index).fillna(0.0)


def yearly_breakdown(
    positions: dict[str, pd.Series], frame: pd.DataFrame
) -> list[dict[str, object]]:
    """Calendar-year return and mean weight for each position path."""
    years: dict[int, dict[str, object]] = {}
    for name, path in positions.items():
        returns, _, _ = position_log_returns(path, frame)
        weights = path.reindex(returns.index).fillna(0.0)
        for year, values in returns.groupby(returns.index.year):
            row = years.setdefault(int(year), {"year": int(year)})
            row[f"{name}_return"] = float(np.expm1(values.sum()))
            row[f"{name}_mean_weight"] = float(weights.loc[values.index].mean())
    return [years[year] for year in sorted(years)]


def summarize(
    scores: list[FoldScore],
    correct: np.ndarray,
    signals: pd.DataFrame,
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, object]:
    low, high = block_bootstrap_ci(correct)
    directional_rows = sum(score.directional_rows for score in scores)
    baseline_correct = sum(
        score.causal_baseline_accuracy * score.directional_rows for score in scores
    )
    scaled = build_scaled_positions(signals, frame)
    gate = build_strategy_positions(signals, frame)
    baselines = reference_baselines(signals.index, frame, signals=signals)
    sensitivity = execution_sensitivity(signals, frame)
    trend_spans = trend_span_sensitivity(signals, frame)
    dispersion = fold_grid_dispersion(frame, features)

    trial_sharpes = np.array(
        [row["sharpe"] for row in sensitivity if row["sharpe"] is not None], dtype=float
    )
    trial_variance = float(np.var(trial_sharpes, ddof=1) / BARS_PER_YEAR)
    returns, _, _ = position_log_returns(scaled, frame)
    deflated = deflated_sharpe_ratio(
        returns.to_numpy(),
        trials=SCREENED_CONFIGURATIONS_LOWER_BOUND,
        trial_sharpe_variance=trial_variance,
    )
    deflated["variance_source"] = (
        "per-bar variance of the execution sensitivity sweep; screened "
        "configurations vary more than one knob at a time, so this understates "
        "the true trial variance and the deflated value is optimistic"
    )
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
        "measured_signal_span": {
            "note": (
                "Mean bars each fold's purged calibration labels took to touch a "
                "barrier. The signal is smoothed over this, not over the "
                f"{LABEL_HORIZON_BARS}-bar horizon, which is only its ceiling."
            ),
            "mean": float(np.mean([score.signal_span for score in scores])),
            "min": float(np.min([score.signal_span for score in scores])),
            "max": float(np.max([score.signal_span for score in scores])),
            "label_horizon_bars": LABEL_HORIZON_BARS,
        },
        "execution_baselines": baselines,
        "backtesting_example": continuous_backtest(scaled, frame),
        "binary_gate_variant": continuous_backtest(gate, frame),
        "fee_sensitivity": {
            "backtesting_example": fee_sensitivity(scaled, frame),
            "binary_gate_variant": fee_sensitivity(gate, frame),
            "trend_only": fee_sensitivity(
                reference_index_trend(signals.index, frame), frame
            ),
        },
        "execution_sensitivity": sensitivity,
        "trend_span_sensitivity": trend_spans,
        "fold_grid_offset_dispersion": dispersion,
        "deflated_sharpe": deflated,
        "yearly_breakdown": yearly_breakdown(
            {
                "buy_and_hold": pd.Series(1.0, index=signals.index),
                "trend_only": reference_index_trend(signals.index, frame),
                "binary_gate": gate,
                "backtesting_example": scaled,
            },
            frame,
        ),
        "folds": [asdict(score) for score in scores],
    }


def build_frame(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the frozen 35-feature frame from the checked-in cache."""
    data_dir = Path(data_dir)
    price = pd.read_pickle(data_dir / "cache" / "bitstamp_h12_pt1.5_sl1.5.pkl")
    momentum = get_momentum_feature_columns(price)
    normalized = normalize_price_features(price, momentum)
    order_flow = build_order_flow_features(price.index, data_dir / "market")
    outcome_state = build_mature_outcome_features(price["target"])
    frame = normalized.join(order_flow).join(outcome_state)
    frame["barrier_span"] = build_barrier_spans(price)
    features = momentum + list(ORDER_FLOW_COLUMNS) + list(outcome_state.columns)
    return frame, features


def run(data_dir: str | Path = DATA_DIR) -> dict[str, object]:
    frame, features = build_frame(data_dir)
    momentum = [name for name in features if name.startswith("XBTUSD_")]
    state = [name for name in features if name.startswith("mature_outcome_")]
    windows = make_windows(
        first_test=str(DEVELOPMENT_START),
        last_test=DEVELOPMENT_SIGNAL_END,
        step_months=STEP_MONTHS,
        purge_bars=LABEL_HORIZON_BARS,
    )
    scores, correct, signals = evaluate(frame, features, windows)
    result = summarize(scores, correct, signals, frame, features)
    return {
        "status": "development_result",
        "claim_warning": (
            "Not a holdout result and not evidence of a deployable trading strategy."
        ),
        "protocol": (
            f"{STEP_MONTHS}-month purged walk-forward; forced direction on every "
            "bar; prior purged 12-month threshold calibration; final label-horizon "
            "embargo"
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
            "mature_outcome_features": len(state),
            "outcome_lag_bars": MATURITY_LAG,
            "decision_threshold_grid": list(DECISION_THRESHOLDS),
        },
        "backtesting_example": {
            "side_mode": "long_only",
            "sizing": "continuous, from the log-odds offset to the fold threshold",
            "signal_ewma_span": "per fold, the calibration window's measured barrier span",
            "sizing_floor": SIZING_FLOOR,
            "rebalance_band": REBALANCE_BAND,
            "trend_filter_ema_ladder": list(TREND_EMA_LADDER),
            "commission_per_side": COMMISSION_PER_SIDE,
            "fee_sensitivity_grid": list(FEE_SENSITIVITY_GRID),
        },
        "binary_gate_variant": {
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
