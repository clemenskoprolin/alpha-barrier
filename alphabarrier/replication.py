"""Replicate the execution rule on the history before the development window.

The four development years are where every execution choice in this repository
was made, so they cannot also be the evidence that the choices were right. The
cache reaches back to 2012, and the order-flow block is the only thing that
does not: momentum and matured-outcome state cover the whole span. Since the
soft-versus-hard question is about what to do with a call rather than about how
the call is produced, that reduced feature set is enough to ask it, and it buys
seven and a half years and three bear markets outside the development window.

Run as ``python -m alphabarrier.replication``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphabarrier.evaluation import make_windows
from alphabarrier.execution import (
    COMMISSION_PER_SIDE,
    REBALANCE_BAND,
    SIGNAL_EWMA_SPAN,
    SIZING_FLOOR,
    TREND_EMA_LADDER,
    TREND_EMA_SPAN,
    apply_rebalance_band,
    build_scaled_positions,
    build_strategy_positions,
    continuous_backtest,
    position_log_returns,
    signal_size,
    trend_gate,
    trend_state,
)
from alphabarrier.features import LABEL_HORIZON_BARS
from alphabarrier.strategy import (
    BARS_PER_YEAR,
    DATA_DIR,
    DEVELOPMENT_START,
    FOLD_GRID_OFFSETS,
    LABEL_EMBARGO,
    LATER_EVALUATION_START,
    STEP_MONTHS,
    build_frame,
    evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "long_history.json"
REPLICATION_START = pd.Timestamp("2014-01-01")
SHORT_SLEEVE_GRID = (0.25, 0.5, 1.0)
OFF_TREND_PARTICIPATION_GRID = (0.15, 0.5)
PERIODS = (
    ("pre_development", REPLICATION_START, DEVELOPMENT_START),
    ("development", DEVELOPMENT_START, LATER_EVALUATION_START),
    ("later", LATER_EVALUATION_START, None),
    ("all", REPLICATION_START, None),
)


def replication_features(features: list[str]) -> list[str]:
    """The subset of the feature set that exists back to 2014.

    The twelve Binance order-flow features start in 2017 for spot and 2020 for
    the perpetual, so they are dropped rather than back-filled.
    """
    return [
        name
        for name in features
        if name.startswith("XBTUSD_") or name.startswith("mature_outcome_")
    ]


def replication_windows(
    frame: pd.DataFrame, month_offset: int
) -> list:
    """The walk-forward grid over the full cache, shifted by whole months."""
    return make_windows(
        first_test=str(REPLICATION_START + pd.DateOffset(months=month_offset)),
        last_test=frame.index[-1] - LABEL_EMBARGO,
        step_months=STEP_MONTHS,
        purge_bars=LABEL_HORIZON_BARS,
        bar_hours=4,
    )


def _rules(frame: pd.DataFrame) -> dict[str, object]:
    """Every position rule the replication compares, over one signal frame."""
    return {
        "current": lambda signals: build_scaled_positions(signals, frame),
        "previous_continuous": lambda signals: build_scaled_positions(
            signals,
            frame,
            ewma_span=SIGNAL_EWMA_SPAN,
            floor=0.0,
            trend_spans=TREND_EMA_SPAN,
        ),
        "binary_gate": lambda signals: build_strategy_positions(signals, frame),
        "trend_only": lambda signals: trend_gate(frame)
        .reindex(signals.index)
        .fillna(0.0),
        "trend_only_single_span": lambda signals: trend_state(frame)
        .reindex(signals.index)
        .fillna(0.0),
        "buy_and_hold": lambda signals: pd.Series(1.0, index=signals.index),
    }


def short_sleeve(
    signals: pd.DataFrame, frame: pd.DataFrame, scale: float
) -> pd.Series:
    """Short the trend-off bars in proportion to the model's down conviction."""
    gate = trend_gate(frame).reindex(signals.index).fillna(0.0)
    mirrored = signals.assign(
        probability=1.0 - signals["probability"],
        threshold=1.0 - signals["threshold"],
    )
    target = signal_size(signals) * gate - scale * signal_size(mirrored) * (1.0 - gate)
    return apply_rebalance_band(target, REBALANCE_BAND)


def off_trend_participation(
    signals: pd.DataFrame, frame: pd.DataFrame, share: float
) -> pd.Series:
    """Keep a fraction of the long exposure on the bars the gate would flatten."""
    gate = trend_gate(frame).reindex(signals.index).fillna(0.0)
    target = (signal_size(signals) * (gate + share * (1.0 - gate))).clip(0.0, 1.0)
    return apply_rebalance_band(target, REBALANCE_BAND)


def off_trend_edge(
    signals: pd.DataFrame, frame: pd.DataFrame, *, quantiles: int = 5
) -> dict[str, object]:
    """What is actually in the bars the gate flattens, before trying to trade them.

    The graded ladder already participates on most of what the old binary gate
    discarded, so the question is only about what is left, and the answer has to
    be read per conviction bucket rather than in aggregate.
    """
    single = trend_state(frame).reindex(signals.index).fillna(0.0)
    ladder = trend_gate(frame).reindex(signals.index).fillna(0.0)
    ahead = (frame["raw_close"].shift(-1) / frame["raw_close"] - 1.0).reindex(
        signals.index
    )
    size = signal_size(signals, floor=0.0)
    report: dict[str, object] = {
        "note": "Mean next-bar return in basis points, by gate and conviction.",
        "single_span_gate_flat_share": float((single == 0.0).mean()),
        "ladder_gate_flat_share": float((ladder == 0.0).mean()),
    }
    for name, gate in (("single_span_gate", single), ("ladder_gate", ladder)):
        flat = gate == 0.0
        up = size > 0.5
        buckets = pd.qcut(size[flat], quantiles, labels=False, duplicates="drop")
        report[name] = {
            "flat_bars": int(flat.sum()),
            "up_call_mean_bp": float(ahead[flat & up].mean() * 1e4),
            "up_call_bars": int((flat & up).sum()),
            "down_call_mean_bp": float(ahead[flat & ~up].mean() * 1e4),
            "down_call_bars": int((flat & ~up).sum()),
            "size_quantile_mean_bp": [
                float(ahead[flat][buckets == bucket].mean() * 1e4)
                for bucket in sorted(pd.unique(buckets.dropna()))
            ],
        }
    return report


def sleeve_table(
    signal_sets: list[pd.DataFrame], frame: pd.DataFrame
) -> dict[str, object]:
    """Price the two ways of trading the bars the trend gate flattens.

    Shorting them in proportion to down conviction, or keeping part of the long
    exposure through them. Neither survives; :func:`off_trend_edge` says why.
    """
    variants: dict[str, object] = {
        "current": lambda signals: build_scaled_positions(signals, frame)
    }
    for scale in SHORT_SLEEVE_GRID:
        variants[f"short_sleeve_{scale:g}x"] = (
            lambda signals, scale=scale: short_sleeve(signals, frame, scale)
        )
    for share in OFF_TREND_PARTICIPATION_GRID:
        variants[f"off_trend_participation_{share:g}x"] = (
            lambda signals, share=share: off_trend_participation(signals, frame, share)
        )

    rows = {}
    for name, build in variants.items():
        scores: dict[str, float | None] = {}
        for period, start, end in PERIODS:
            measured = []
            for signals in signal_sets:
                inside = signals.index >= start
                if end is not None:
                    inside &= signals.index < end
                window = signals.loc[inside]
                if len(window) < 200:
                    continue
                sharpe = continuous_backtest(build(window), frame)["sharpe"]
                if sharpe is not None:
                    measured.append(sharpe)
            scores[period] = float(np.mean(measured)) if measured else None
        rows[name] = scores
    return {
        "note": (
            "Sharpe by period, averaged over the fold-grid shifts. Read against "
            "the fold-grid noise floor, not against zero."
        ),
        "variants": rows,
        "edge": off_trend_edge(signal_sets[0], frame),
    }


def _annualized_sharpe(returns: pd.Series) -> float | None:
    deviation = returns.std(ddof=1)
    if not len(returns) > 1 or not deviation > 0:
        return None
    return float(returns.mean() / deviation * np.sqrt(BARS_PER_YEAR))


def period_table(
    signal_sets: list[pd.DataFrame], frame: pd.DataFrame
) -> list[dict[str, object]]:
    """Backtest every rule over every period, averaged across fold-grid shifts."""
    rows = []
    for name, start, end in PERIODS:
        row: dict[str, object] = {
            "period": name,
            "start": str(start),
            "end_exclusive": str(end) if end is not None else str(frame.index[-1]),
        }
        for rule, build in _rules(frame).items():
            measured: dict[str, list[float]] = {}
            for signals in signal_sets:
                inside = signals.index >= start
                if end is not None:
                    inside &= signals.index < end
                window = signals.loc[inside]
                if len(window) < 200:
                    continue
                for key, value in continuous_backtest(build(window), frame).items():
                    if isinstance(value, (int, float)):
                        measured.setdefault(key, []).append(float(value))
            if not measured:
                continue
            row[rule] = {
                key: float(np.mean(values))
                for key, values in measured.items()
                if key in {"total_return", "sharpe", "max_drawdown", "mean_weight"}
            }
        rows.append(row)
    return rows


def yearly_sharpe_table(
    signal_sets: list[pd.DataFrame], frame: pd.DataFrame
) -> dict[str, object]:
    """Per-calendar-year Sharpe for each rule, and the paired soft-minus-hard gap.

    Aggregate Sharpe over a decade can be carried by two good years. Counting
    the years each rule wins cannot be.
    """
    builders = _rules(frame)
    yearly: dict[str, dict[int, list[float]]] = {rule: {} for rule in builders}
    for rule, build in builders.items():
        for signals in signal_sets:
            returns, _, _ = position_log_returns(build(signals), frame)
            for year, values in returns.groupby(returns.index.year):
                if len(values) < 200:
                    continue
                sharpe = _annualized_sharpe(values)
                if sharpe is not None:
                    yearly[rule].setdefault(int(year), []).append(sharpe)

    years = sorted(yearly["current"])
    rows = []
    for year in years:
        row: dict[str, object] = {"year": year}
        for rule in builders:
            if year in yearly[rule]:
                row[rule] = float(np.mean(yearly[rule][year]))
        rows.append(row)

    def paired(soft: str, hard: str) -> dict[str, object]:
        gaps = np.array(
            [row[soft] - row[hard] for row in rows if soft in row and hard in row]
        )
        before = [
            row[soft] - row[hard]
            for row in rows
            if row["year"] < DEVELOPMENT_START.year and soft in row and hard in row
        ]
        error = gaps.std(ddof=1) / np.sqrt(len(gaps)) if len(gaps) > 1 else np.nan
        return {
            "comparison": f"{soft} minus {hard}",
            "years": int(len(gaps)),
            "years_won": int((gaps > 0).sum()),
            "mean_gap": float(gaps.mean()),
            "sd_gap": float(gaps.std(ddof=1)) if len(gaps) > 1 else None,
            "t_statistic": float(gaps.mean() / error) if error > 0 else None,
            "pre_development_years": int(len(before)),
            "pre_development_years_won": int(sum(gap > 0 for gap in before)),
            "pre_development_mean_gap": float(np.mean(before)) if before else None,
        }

    return {
        "note": (
            "Sharpe within each calendar year, averaged over the fold-grid "
            "shifts. The paired rows count years rather than pooling them."
        ),
        "years": rows,
        "paired": [
            paired("previous_continuous", "binary_gate"),
            paired("current", "binary_gate"),
            paired("current", "previous_continuous"),
        ],
    }


def run(data_dir: str | Path = DATA_DIR) -> dict[str, object]:
    frame, features = build_frame(data_dir)
    reduced = replication_features(features)
    signal_sets, accuracies, windows = [], [], []
    for offset in FOLD_GRID_OFFSETS:
        scores, correct, signals = evaluate(
            frame, reduced, replication_windows(frame, offset)
        )
        signal_sets.append(signals)
        accuracies.append(float(correct.mean()))
        windows.append(len(scores))
    return {
        "status": "replication_outside_the_development_window",
        "question": (
            "Does the execution rule chosen on 2021-2025 hold up on the history "
            "before it?"
        ),
        "warning": (
            "The pre-development years were inspected while the current rule was "
            "being designed, so they are evidence against the rule they falsified "
            "and only a hypothesis for the one that replaced it."
        ),
        "protocol": {
            "start": str(REPLICATION_START),
            "end_exclusive": str(frame.index[-1]),
            "features": (
                "momentum and matured-outcome state only; the order-flow block "
                "starts in 2017 and 2020 and is dropped"
            ),
            "feature_count": len(reduced),
            "step_months": STEP_MONTHS,
            "fold_grid_offsets": list(FOLD_GRID_OFFSETS),
            "windows_per_offset": windows,
            "commission_per_side": COMMISSION_PER_SIDE,
        },
        "rules": {
            "current": (
                f"continuous sizing, floor {SIZING_FLOOR:.4f}, measured signal "
                f"span, EMA ladder {list(TREND_EMA_LADDER)}"
            ),
            "previous_continuous": (
                f"continuous sizing, no floor, span {SIGNAL_EWMA_SPAN}, single "
                f"{TREND_EMA_SPAN}-bar gate"
            ),
            "binary_gate": f"hysteretic binary state, single {TREND_EMA_SPAN}-bar gate",
        },
        "directional_accuracy": float(np.mean(accuracies)),
        "periods": period_table(signal_sets, frame),
        "yearly_sharpe": yearly_sharpe_table(signal_sets, frame),
        "trend_off_sleeves": sleeve_table(signal_sets, frame),
    }


def main() -> None:
    output = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    RESULT_PATH.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
