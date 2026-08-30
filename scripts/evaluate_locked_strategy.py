"""Evaluate the locked model/execution recipe on a later data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from alphabarrier.evaluation import (
    Window,
    block_bootstrap_ci,
    normalize_price_features,
)
from alphabarrier.execution import (
    COMMISSION_PER_SIDE,
    CONFIDENCE_FLOOR,
    TREND_EMA_SPAN,
    build_strategy_positions,
    continuous_backtest,
)
from alphabarrier.features import LABEL_HORIZON_BARS, get_momentum_feature_columns
from alphabarrier.order_flow import ORDER_FLOW_COLUMNS, build_order_flow_features
from alphabarrier.strategy import (
    LABEL_EMBARGO,
    LATER_EVALUATION_END,
    LATER_EVALUATION_START,
    build_mature_outcome_features,
    evaluate,
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
    price = pd.read_pickle(data_dir / "cache" / "bitstamp_h12_pt1.5_sl1.5.pkl")
    momentum = get_momentum_feature_columns(price)
    normalized = normalize_price_features(price, momentum)
    order_flow = build_order_flow_features(price.index, data_dir / "market")
    outcome_state = build_mature_outcome_features(price["target"])
    frame = normalized.join(order_flow).join(outcome_state)
    features = momentum + list(ORDER_FLOW_COLUMNS) + list(outcome_state.columns)
    window = Window(
        train_start=pd.Timestamp("2012-01-01"),
        train_end=test_start - LABEL_EMBARGO,
        test_start=test_start,
        test_end=test_end,
    )
    scores, correct, signals = evaluate(frame, features, [window])

    positions = build_strategy_positions(signals, frame)

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
                "weak_calls": "hold_previous_model_state",
                "confidence_floor": CONFIDENCE_FLOOR,
                "trend_filter_ema_span": TREND_EMA_SPAN,
                "commission_per_side": COMMISSION_PER_SIDE,
            },
        },
        "classification": {
            "directional_samples": int(len(correct)),
            "correct": int(correct.sum()),
            "directional_accuracy": float(correct.mean()),
            "directional_accuracy_block_bootstrap_ci95": [low, high],
            "causal_majority_baseline_accuracy": scores[0].causal_baseline_accuracy,
        },
        "execution": continuous_backtest(positions, frame),
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
