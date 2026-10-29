"""Causal price features and fixed-horizon triple-barrier labels."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


DOWN, SIDEWAYS, UP = 0, 1, 2
LABEL_HORIZON_BARS = 12
PROFIT_TAKE_ATR_MULT = 1.5
STOP_LOSS_ATR_MULT = 1.5
DIRECTION_FEATURE_PATTERNS = (
    "log_return",
    "close_to_ema",
    "rsi",
    "macd",
    "bollinger",
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    loss = (-delta.clip(upper=0.0)).ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    relative_strength = _safe_divide(gain, loss)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    """Exponentially weighted average true range."""
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()


def add_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the frozen 17 causal momentum features used by the model."""
    result = pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)
    log_return_1 = np.log(close).diff()

    for lag in (1, 2, 3, 6, 12, 24, 48):
        result[f"feature_log_return_{lag}"] = np.log(
            _safe_divide(close, close.shift(lag))
        )

    for window in (6, 12, 24, 48, 96):
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        result[f"feature_close_to_ema_{window}"] = _safe_divide(close, ema) - 1.0

    result["feature_rsi_14"] = _rsi(close, 14) / 100.0
    fast_ema = close.ewm(span=12, adjust=False, min_periods=26).mean()
    slow_ema = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["feature_macd_pct"] = _safe_divide(macd, close)
    result["feature_macd_signal_pct"] = _safe_divide(macd_signal, close)
    result["feature_macd_hist_pct"] = _safe_divide(macd - macd_signal, close)

    rolling_mid = close.rolling(20, min_periods=20).mean()
    rolling_std = close.rolling(20, min_periods=20).std(ddof=0)
    result["feature_bollinger_z"] = _safe_divide(close - rolling_mid, rolling_std)
    return result


def _first_touch(
    prices: pd.DataFrame,
    atr_series: pd.Series,
    horizon_bars: int,
    pt_mult: float,
    sl_mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk the barriers once, returning each entry's outcome and its bar count.

    The outcome is ``1``/``-1`` for the barrier touched first, ``0`` when the
    horizon expires untouched, and NaN when both barriers fall inside one OHLC
    bar, because the intrabar ordering is unknowable. The bar count is the
    number of bars until that touch, and is NaN whenever the outcome is not a
    barrier touch.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1")
    if pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("barrier multipliers must be positive")

    outcomes = np.full(len(prices), np.nan)
    spans = np.full(len(prices), np.nan)
    close = prices["close"].to_numpy(dtype=float)
    high = prices["high"].to_numpy(dtype=float)
    low = prices["low"].to_numpy(dtype=float)
    atr_values = atr_series.to_numpy(dtype=float)
    for entry in range(0, len(prices) - horizon_bars):
        if not np.isfinite(close[entry]) or not np.isfinite(atr_values[entry]):
            continue
        upper = close[entry] + pt_mult * atr_values[entry]
        lower = close[entry] - sl_mult * atr_values[entry]
        outcome = 0
        for position in range(entry + 1, entry + horizon_bars + 1):
            hit_upper = high[position] >= upper
            hit_lower = low[position] <= lower
            if hit_upper and hit_lower:
                outcome = np.nan
                break
            if hit_upper or hit_lower:
                outcome = 1 if hit_upper else -1
                spans[entry] = position - entry
                break
        outcomes[entry] = outcome
    return outcomes, spans


def get_triple_barrier_labels(
    prices: pd.DataFrame,
    atr_series: pd.Series,
    *,
    horizon_bars: int = LABEL_HORIZON_BARS,
    pt_mult: float = PROFIT_TAKE_ATR_MULT,
    sl_mult: float = STOP_LOSS_ATR_MULT,
) -> pd.Series:
    """Label the first barrier touched strictly after each signal bar.

    Samples where both barriers touch in one OHLC bar are left missing because
    the intrabar ordering is unknowable. Final rows without a full horizon are
    also left missing.
    """
    outcomes, _ = _first_touch(prices, atr_series, horizon_bars, pt_mult, sl_mult)
    return pd.Series(outcomes, index=prices.index, dtype=float)


def realized_barrier_span(
    prices: pd.DataFrame,
    atr_series: pd.Series,
    *,
    horizon_bars: int = LABEL_HORIZON_BARS,
    pt_mult: float = PROFIT_TAKE_ATR_MULT,
    sl_mult: float = STOP_LOSS_ATR_MULT,
) -> pd.Series:
    """Bars until each directional label actually resolves.

    The label horizon is a ceiling, not the lifetime of a call: a barrier is
    touched well before it. This is the measured lifetime, and it is what the
    signal smoothing should be scaled to rather than the horizon constant.
    Rows whose label is not a barrier touch are left missing.
    """
    _, spans = _first_touch(prices, atr_series, horizon_bars, pt_mult, sl_mult)
    return pd.Series(spans, index=prices.index, dtype=float)


def read_ohlcv(path: str | os.PathLike[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] not in (6, 7):
        raise ValueError(f"{path} has {frame.shape[1]} columns; expected 6 or 7")
    frame.columns = ["time", "open", "high", "low", "close", "volume", "trades"][:
        frame.shape[1]
    ]
    numeric_time = pd.to_numeric(frame.pop("time"), errors="coerce")
    time_unit = "ms" if numeric_time.dropna().median() > 10_000_000_000 else "s"
    frame.index = pd.to_datetime(
        numeric_time, unit=time_unit, utc=True
    ).dt.tz_localize(None)
    frame.index.name = "time"
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if "feature_" in column)


def get_momentum_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in get_feature_columns(frame)
        if any(pattern in column for pattern in DIRECTION_FEATURE_PATTERNS)
    ]
