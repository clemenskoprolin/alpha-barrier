"""Evaluate the locked model/execution recipe on a later data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from alphabarrier.evaluation import Window, block_bootstrap_ci
from alphabarrier.execution import (
    COMMISSION_PER_SIDE,
    CONFIDENCE_FLOOR,
    REBALANCE_BAND,
    SIZING_FLOOR,
    TREND_EMA_LADDER,
    TREND_EMA_SPAN,
    build_scaled_positions,
    build_strategy_positions,
    continuous_backtest,
    fee_sensitivity,
    reference_baselines,
)
from alphabarrier.features import LABEL_HORIZON_BARS
from alphabarrier.strategy import (
    LABEL_EMBARGO,
    LATER_EVALUATION_END,
    LATER_EVALUATION_START,
    build_frame,
    evaluate,
    yearly_breakdown,
)


def run(
    data_dir: str | Path,
    *,
    test_start: str | pd.Timestamp = LATER_EVALUATION_START,
    test_end: str | pd.Timestamp = LATER_EVALUATION_END,
) -> dict[str, object]:
    """Run the fixed model over one retrospective later-evaluation period."""
    data_dir = Path(data_dir)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    frame, features = build_frame(data_dir)
    window = Window(
        train_start=pd.Timestamp("2012-01-01"),
        train_end=test_start - LABEL_EMBARGO,
        test_start=test_start,
        test_end=test_end,
    )
    scores, correct, signals = evaluate(frame, features, [window])
    scaled = build_scaled_positions(signals, frame)
    gate = build_strategy_positions(signals, frame)
    low, high = block_bootstrap_ci(correct)
    return {
        "status": "retrospective_later_evaluation",
        "warning": (
            "The model configuration was selected after extensive screening that "
            "included parts of this period. Treat this as a retrospective split, "
            "not a pristine holdout claim."
        ),
        "test_start": str(test_start),
        "test_end_exclusive": str(test_end),
        "protocol": {
            "train_end_exclusive": str(window.train_end),
            "label_horizon_bars": LABEL_HORIZON_BARS,
            "decision_threshold": scores[0].threshold,
            "execution": {
                "side_mode": "long_only",
                "sizing": "continuous, from the log-odds offset to the fold threshold",
                "signal_ewma_span": scores[0].signal_span,
                "sizing_floor": SIZING_FLOOR,
                "rebalance_band": REBALANCE_BAND,
                "trend_filter_ema_ladder": list(TREND_EMA_LADDER),
                "commission_per_side": COMMISSION_PER_SIDE,
            },
            "binary_gate_variant": {
                "weak_calls": "hold_previous_model_state",
                "confidence_floor": CONFIDENCE_FLOOR,
                "trend_filter_ema_span": TREND_EMA_SPAN,
            },
        },
        "classification": {
            "directional_samples": int(len(correct)),
            "correct": int(correct.sum()),
            "directional_accuracy": float(correct.mean()),
            "directional_accuracy_block_bootstrap_ci95": [low, high],
            "causal_majority_baseline_accuracy": scores[0].causal_baseline_accuracy,
        },
        "execution_baselines": reference_baselines(
            signals.index, frame, signals=signals
        ),
        "execution": continuous_backtest(scaled, frame),
        "binary_gate_variant": continuous_backtest(gate, frame),
        "fee_sensitivity": {
            "execution": fee_sensitivity(scaled, frame),
            "binary_gate_variant": fee_sensitivity(gate, frame),
        },
        "yearly_breakdown": yearly_breakdown(
            {"backtesting_example": scaled, "binary_gate": gate}, frame
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-start", default=str(LATER_EVALUATION_START.date()))
    parser.add_argument("--test-end", default=str(LATER_EVALUATION_END.date()))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.data_dir, test_start=args.test_start, test_end=args.test_end)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
