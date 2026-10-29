"""Screen alternative model classes and label horizons against the logistic.

Everything here was tried because it should plausibly have helped and is
recorded because it did not. Each candidate runs through the identical purged
walk-forward, calibration and embargo as the published model, over the long
history rather than the development window, so a candidate that only wins on
the four years the current design was chosen on is visible as such.

Run as ``python -m scripts.screen_alternatives``. It is slower than the main
protocol -- the tree ensemble and the bagged linear model dominate -- so it
defaults to a single fold-grid offset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alphabarrier.execution import build_scaled_positions, continuous_backtest
from alphabarrier.features import (
    LABEL_HORIZON_BARS,
    SIDEWAYS,
    UP,
    get_triple_barrier_labels,
    realized_barrier_span,
)
from alphabarrier.replication import (
    PERIODS,
    replication_features,
    replication_windows,
)
from alphabarrier.strategy import (
    DATA_DIR,
    build_frame,
    evaluate,
    fit_logistic,
    impute_and_call,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "alternatives.json"
BAGGED_C_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)
BAGGED_DRAWS = 3
HORIZON_LADDER = (6, 12, 24)


def _training_rows(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    usable = frame.dropna(subset=features + ["target"])
    return usable.loc[usable["target"].ne(SIDEWAYS)]


class _Bagged:
    """Average the probabilities of logistic fits over bootstraps and penalties.

    Variance reduction without added capacity is the kind of change that tends
    to survive out of sample, which is why it is worth measuring.
    """

    def __init__(self, models, scaler):
        self._models = models
        self._scaler = scaler

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        scaled = self._scaler.transform(values)
        mean = np.mean(
            [model.predict_proba(scaled)[:, 1] for model in self._models], axis=0
        )
        return np.column_stack([1.0 - mean, mean])


def fit_bagged_logistic(frame: pd.DataFrame, features: list[str]):
    train = _training_rows(frame, features)
    values = train[features].to_numpy()
    labels = train["target"].eq(UP).astype(int).to_numpy()
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    generator = np.random.default_rng(42)
    models = [
        LogisticRegression(
            C=penalty, class_weight="balanced", max_iter=3_000, random_state=42
        ).fit(scaled[draw], labels[draw])
        for penalty in BAGGED_C_GRID
        for draw in (
            generator.choice(len(scaled), len(scaled), replace=True)
            for _ in range(BAGGED_DRAWS)
        )
    ]
    return impute_and_call(
        _Bagged(models, scaler), np.nanmedian(values, axis=0), scaler=None
    )


def fit_gradient_boosting(frame: pd.DataFrame, features: list[str]):
    """A shallow tree ensemble: the obvious way to buy interactions."""
    train = _training_rows(frame, features)
    values = train[features].to_numpy()
    model = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        min_samples_leaf=100,
        class_weight="balanced",
        random_state=42,
    ).fit(values, train["target"].eq(UP).astype(int))
    return impute_and_call(model, np.nanmedian(values, axis=0), scaler=None)


def horizon_frames(
    frame: pd.DataFrame, features: list[str], horizons: tuple[int, ...]
) -> dict[int, tuple[pd.DataFrame, list[str]]]:
    """Relabel the cache at each horizon, with horizon-matched state features.

    The twelve-bar horizon is exactly as arbitrary as the trend filter's
    lookback was, so the same fix is worth trying: average over a ladder of
    them instead of choosing one.
    """
    momentum = [name for name in features if name.startswith("XBTUSD_")]
    prices = frame.rename(
        columns={
            "raw_open": "open",
            "raw_high": "high",
            "raw_low": "low",
            "raw_close": "close",
        }
    )[["open", "high", "low", "close"]]
    built = {}
    for horizon in horizons:
        target = get_triple_barrier_labels(
            prices, frame["raw_atr"], horizon_bars=horizon
        ).map({-1.0: 0.0, 0.0: 1.0, 1.0: 2.0})
        state = _horizon_state(target, horizon)
        relabelled = frame[momentum + ["raw_close", "raw_atr"]].join(state)
        relabelled["target"] = target
        relabelled["barrier_span"] = realized_barrier_span(
            prices, frame["raw_atr"], horizon_bars=horizon
        )
        built[horizon] = (relabelled, momentum + list(state.columns))
    return built


def _horizon_state(target: pd.Series, horizon: int) -> pd.DataFrame:
    """Matured-outcome state lagged past this horizon rather than past twelve."""
    signed = target.map({0.0: -1.0, 1.0: 0.0, 2.0: 1.0}).shift(horizon + 1)
    return pd.DataFrame(
        {
            f"mature_outcome_ewm_{span}": signed.ewm(
                span=span, min_periods=max(3, span // 4), adjust=False
            ).mean()
            for span in (3, 9, 24, 72, 240, 720)
        }
    )


def _period_scores(
    signal_sets: list[pd.DataFrame], frame: pd.DataFrame
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for name, start, end in PERIODS:
        measured = []
        for signals in signal_sets:
            inside = signals.index >= start
            if end is not None:
                inside &= signals.index < end
            window = signals.loc[inside]
            if len(window) < 200:
                continue
            result = continuous_backtest(build_scaled_positions(window, frame), frame)
            if result["sharpe"] is not None:
                measured.append(result["sharpe"])
        scores[name] = float(np.mean(measured)) if measured else None
    return scores


def screen_model(
    frame: pd.DataFrame, features: list[str], offsets: tuple[int, ...], fitter
) -> dict[str, object]:
    signal_sets, accuracies = [], []
    for offset in offsets:
        _, correct, signals = evaluate(
            frame, features, replication_windows(frame, offset), fitter=fitter
        )
        signal_sets.append(signals)
        accuracies.append(float(correct.mean()))
    return {
        "directional_accuracy": float(np.mean(accuracies)),
        "sharpe": _period_scores(signal_sets, frame),
    }


def screen_horizon_ladder(
    frame: pd.DataFrame, features: list[str], offsets: tuple[int, ...]
) -> dict[str, object]:
    built = horizon_frames(frame, features, HORIZON_LADDER)
    per_horizon, ladder_sets = {}, []
    for offset in offsets:
        sizes = []
        for horizon, (relabelled, columns) in built.items():
            _, correct, signals = evaluate(
                relabelled, columns, replication_windows(relabelled, offset)
            )
            per_horizon.setdefault(horizon, {"accuracy": [], "signals": []})
            per_horizon[horizon]["accuracy"].append(float(correct.mean()))
            per_horizon[horizon]["signals"].append(signals)
            sizes.append(signals)
        shared = sizes[0].index
        for signals in sizes[1:]:
            shared = shared.intersection(signals.index)
        blended = sizes[0].loc[shared].copy()
        # Average the calls in probability space; the fold threshold and spread
        # are already what makes them comparable across horizons.
        blended["probability"] = np.mean(
            [signals.loc[shared, "probability"].to_numpy() for signals in sizes], axis=0
        )
        blended["threshold"] = np.mean(
            [signals.loc[shared, "threshold"].to_numpy() for signals in sizes], axis=0
        )
        blended["logit_scale"] = np.mean(
            [signals.loc[shared, "logit_scale"].to_numpy() for signals in sizes], axis=0
        )
        blended["signal_span"] = np.mean(
            [signals.loc[shared, "signal_span"].to_numpy() for signals in sizes], axis=0
        )
        ladder_sets.append(blended)
    return {
        "horizons": {
            str(horizon): {
                "directional_accuracy": float(np.mean(record["accuracy"])),
                "sharpe": _period_scores(record["signals"], frame),
            }
            for horizon, record in per_horizon.items()
        },
        "ladder": {"sharpe": _period_scores(ladder_sets, frame)},
    }


def run(
    data_dir: str | Path = DATA_DIR, *, offsets: tuple[int, ...] = (0,)
) -> dict[str, object]:
    frame, features = build_frame(data_dir)
    reduced = replication_features(features)
    candidates = {
        "published_logistic": fit_logistic,
        "bagged_logistic": fit_bagged_logistic,
        "gradient_boosting": fit_gradient_boosting,
    }
    models, elapsed = {}, {}
    for name, fitter in candidates.items():
        started = time.monotonic()
        models[name] = screen_model(frame, reduced, offsets, fitter)
        elapsed[name] = round(time.monotonic() - started, 1)
    started = time.monotonic()
    ladder = screen_horizon_ladder(frame, features, offsets)
    elapsed["horizon_ladder"] = round(time.monotonic() - started, 1)
    return {
        "status": "negative_results",
        "question": (
            "Does a different model class, or averaging over the arbitrary label "
            "horizon, beat the published logistic regression?"
        ),
        "protocol": {
            "features": f"{len(reduced)} momentum and matured-outcome features",
            "fold_grid_offsets": list(offsets),
            "periods": [name for name, _, _ in PERIODS],
            "execution": "the current continuous rule, identical for every candidate",
            "label_horizon_bars": LABEL_HORIZON_BARS,
            "horizon_ladder": list(HORIZON_LADDER),
            "seconds": elapsed,
        },
        "model_classes": models,
        "label_horizon": ladder,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--offsets", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    result = run(args.data_dir, offsets=tuple(args.offsets))
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
