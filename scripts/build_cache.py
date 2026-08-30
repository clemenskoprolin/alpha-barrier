"""Build the checked-in causal feature and label cache from Bitstamp OHLCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alphabarrier.features import (
    DOWN,
    SIDEWAYS,
    UP,
    add_momentum_features,
    atr,
    get_triple_barrier_labels,
    read_ohlcv,
)


ROOT = Path(__file__).resolve().parents[1]


def build(
    source: str | Path,
    destination: str | Path,
    *,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    raw = read_ohlcv(source)
    if end is not None:
        raw = raw.loc[raw.index < pd.Timestamp(end)]
    frame = add_momentum_features(raw).add_prefix("XBTUSD_")
    labels = get_triple_barrier_labels(raw[["high", "low", "close"]], atr(raw))
    frame["target"] = labels.map({-1: DOWN, 0: SIDEWAYS, 1: UP})
    for column in ("open", "high", "low", "close", "volume"):
        frame[f"raw_{column}"] = raw[column]
    frame["raw_atr"] = atr(raw)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(destination)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "training_data" / "bitstamp" / "XBTUSD_240.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "cache" / "bitstamp_h12_pt1.5_sl1.5.pkl",
    )
    parser.add_argument("--end", default="2025-12-31 23:59:59")
    args = parser.parse_args()
    frame = build(args.source, args.output, end=args.end)
    print(f"wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main()
