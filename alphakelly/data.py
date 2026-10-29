"""Volatility, long-horizon trend state, and the forward returns being forecast.

``alphabarrier`` predicts which of two barriers is touched first, which is a
class. This package predicts a return, so it needs three things the classifier
never had: a causal volatility estimate to express returns in, a forward return
target expressed in those units, and the long-horizon trend state that
``alphabarrier`` applied as an external multiplicative gate rather than showing
to the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabarrier.evaluation import normalize_price_features
from alphabarrier.execution import TREND_EMA_LADDER
from alphabarrier.strategy import DATA_DIR, build_frame as build_barrier_frame


# Half a day to a week. Geometric, like every other ladder here, and chosen to
# bracket the 12-bar barrier horizon rather than to be centred on it.
HORIZONS = (3, 6, 12, 24, 48)
MAX_HORIZON = max(HORIZONS)
# One day to two months of four-hour bars.
VOL_SPANS = (6, 24, 96, 384)
# The targets are volatility-normalized, so a five-sigma move is already far
# outside anything the linear model can be asked to fit. The clip only stops a
# near-zero volatility estimate from turning one bar into the whole regression.
TARGET_CLIP = 5.0
TREND_FEATURE_SPANS = TREND_EMA_LADDER
VOL_COLUMN = "kelly_vol"
FORWARD_PREFIX = "fwd_y_"
SPOT_TARGET = f"{FORWARD_PREFIX}1"


def realized_volatility(
    frame: pd.DataFrame, *, spans: tuple[int, ...] = VOL_SPANS
) -> pd.Series:
    """Causal per-bar log-return standard deviation, averaged over a span ladder.

    Second moments are averaged, not standard deviations, because variance is
    what the Kelly weight divides by. The ladder is the same refusal to pick a
    lookback that fixed ``alphabarrier``'s trend filter: a single half-life is a
    draw from a wide spread, and averaging the range removes the choice.
    """
    if not spans or any(span < 1 for span in spans):
        raise ValueError("spans must contain positive integers")
    squared = np.log(frame["raw_close"].astype(float)).diff() ** 2
    variance = sum(
        squared.ewm(span=span, adjust=False, min_periods=max(4, span // 4)).mean()
        for span in spans
    ) / float(len(spans))
    return np.sqrt(variance).rename(VOL_COLUMN)


def long_trend_features(
    frame: pd.DataFrame,
    *,
    spans: tuple[int, ...] = TREND_FEATURE_SPANS,
    window: int = 1_000,
) -> pd.DataFrame:
    """The trend ladder as features rather than as a multiplicative veto.

    ``alphabarrier`` multiplies its position by the share of this ladder the
    close sits above. That is a hard overlay bolted onto an accuracy-trained
    classifier because the classifier had no way to express it. A model trained
    on returns can weigh long-horizon trend against everything else itself, so
    the ladder is handed over as six normalized gaps plus the share, and the
    overlay is kept only as a comparison.
    """
    if not spans or any(span < 1 for span in spans):
        raise ValueError("spans must contain positive integers")
    close = frame["raw_close"].astype(float)
    gaps = pd.DataFrame(index=frame.index)
    states = []
    for span in spans:
        average = close.ewm(span=span, adjust=False, min_periods=span).mean()
        gaps[f"trend_gap_{span}"] = close / average - 1.0
        states.append(close.gt(average).astype(float).where(average.notna()))
    output = normalize_price_features(gaps, list(gaps.columns), window)
    output["trend_ladder_share"] = sum(states) / float(len(states)) - 0.5
    return output


def forward_targets(
    frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    clip: float = TARGET_CLIP,
) -> pd.DataFrame:
    """Forward log returns divided by the move they were expected to make.

    ``y_h = r(t, t+h) / (sigma_t * sqrt(h))`` has unit noise variance at every
    horizon, and its expectation is ``sqrt(h) * mu_t / sigma_t``. Dividing a
    forecast of it by ``sqrt(h)`` therefore recovers the same per-bar drift from
    every horizon, which is why averaging across the ladder is coherent here
    while averaging across ``alphabarrier``'s label horizons was not: those were
    three different questions, and these are one question asked three ways.

    ``fwd_y_1`` is the one-bar case, and is what the per-fold forecast scale is
    calibrated against, because a position held for one bar earns exactly it.
    """
    if clip <= 0:
        raise ValueError("clip must be positive")
    log_close = np.log(frame["raw_close"].astype(float))
    scale = volatility.replace(0.0, np.nan)
    output = pd.DataFrame(index=frame.index)
    for horizon in (1, *horizons):
        forward = log_close.shift(-horizon) - log_close
        output[f"{FORWARD_PREFIX}{horizon}"] = (
            forward / (scale * np.sqrt(horizon))
        ).clip(-clip, clip)
    return output.replace([np.inf, -np.inf], np.nan)


def next_bar_return(frame: pd.DataFrame) -> pd.Series:
    """Simple close-to-close return a position established at this bar earns."""
    closes = frame["raw_close"].astype(float)
    return (closes.shift(-1) / closes - 1.0).rename("next_return")


def build_frame(
    data_dir: str | Path = DATA_DIR, *, horizons: tuple[int, ...] = HORIZONS
) -> tuple[pd.DataFrame, list[str]]:
    """The 35 ``alphabarrier`` features plus a 7-feature long-trend block.

    Everything the classifier used is kept, because the ablation in that
    project's README says all three of its blocks contribute. What is added is
    the trend state it could only apply from outside, and the volatility and
    forward-return columns the growth objective needs.
    """
    frame, features = build_barrier_frame(data_dir)
    volatility = realized_volatility(frame)
    trend = long_trend_features(frame)
    frame = frame.join(trend)
    frame[VOL_COLUMN] = volatility
    frame = frame.join(forward_targets(frame, volatility, horizons=horizons))
    return frame, features + list(trend.columns)


def replication_features(features: list[str]) -> list[str]:
    """The subset that exists back to 2013, for the pre-development history.

    The twelve Binance order-flow features start in 2017 and 2020. Dropping them
    is what buys the seven years outside the window where every choice was made,
    exactly as in ``alphabarrier.replication``.
    """
    return [
        name
        for name in features
        if name.startswith("XBTUSD_")
        or name.startswith("mature_outcome_")
        or name.startswith("trend_")
    ]
