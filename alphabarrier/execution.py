"""Locked long-only Alpha Barrier execution strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


CONFIDENCE_FLOOR = 0.02
TREND_EMA_SPAN = 720
COMMISSION_PER_SIDE = 0.001


def build_strategy_positions(
    signals: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    confidence_floor: float = CONFIDENCE_FLOOR,
    trend_ema_span: int = TREND_EMA_SPAN,
) -> pd.Series:
    """Build the locked long-only, hysteretic, trend-gated position path.

    A confident up call enters long and a confident down call exits. Weak calls
    leave the model state unchanged. The final position is long only while the
    close is above its causal 720-bar exponential moving average.
    """
    if confidence_floor < 0.0:
        raise ValueError("confidence_floor must be non-negative")
    if trend_ema_span < 1:
        raise ValueError("trend_ema_span must be positive")
    missing = {"side", "confidence"}.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")

    model_state = np.zeros(len(signals), dtype=float)
    previous = 0.0
    for offset, (_, signal) in enumerate(signals.iterrows()):
        if not np.isfinite(signal["side"]):
            previous = 0.0
        elif float(signal["confidence"]) >= confidence_floor:
            previous = 1.0 if float(signal["side"]) > 0.0 else 0.0
        model_state[offset] = previous

    trend_average = frame["raw_close"].ewm(
        span=trend_ema_span,
        adjust=False,
        min_periods=trend_ema_span,
    ).mean()
    trend_state = frame["raw_close"].gt(trend_average).astype(float)
    positions = model_state * trend_state.reindex(signals.index).fillna(0.0).to_numpy()
    return pd.Series(positions, index=signals.index, name="position")


def continuous_backtest(
    positions: pd.Series,
    frame: pd.DataFrame,
    *,
    commission_per_side: float = COMMISSION_PER_SIDE,
) -> dict[str, float | None]:
    """Backtest the supplied position path, charging only actual turnover.

    A position established at a bar close earns the following close-to-close
    return. The final position is explicitly closed. Slippage, spread, borrow,
    and funding are not modeled.
    """
    if commission_per_side < 0.0:
        raise ValueError("commission_per_side must be non-negative")
    if positions.empty:
        return {
            "total_return": None,
            "annualized_log_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "bar_exposure": 0.0,
            "fee_sides": 0.0,
            "commission_per_side": commission_per_side,
        }

    aligned = positions.reindex(frame.index)
    active = aligned.notna().to_numpy()
    offsets = np.flatnonzero(active)
    if not len(offsets):
        raise ValueError("no position timestamps overlap frame")
    first, last = int(offsets[0]), int(offsets[-1])
    closes = frame["raw_close"].astype(float)
    next_return = closes.shift(-1) / closes - 1.0
    if last + 1 >= len(frame) or not np.isfinite(next_return.iloc[last]):
        raise ValueError("frame needs one valid close after its final position")

    position = aligned.fillna(0.0).to_numpy(dtype=float)
    position[:first] = 0.0
    position[last + 1 :] = 0.0
    previous_position = np.r_[0.0, position[:-1]]
    fee_sides = np.abs(position - previous_position)
    simple = position * next_return.to_numpy() - commission_per_side * fee_sides
    closing_fee_sides = abs(position[last])
    simple[last] -= commission_per_side * closing_fee_sides
    usable = (np.arange(len(frame)) >= first) & (np.arange(len(frame)) <= last)
    usable &= np.isfinite(simple)
    values = np.log1p(simple[usable])
    timestamps = frame.index[usable]
    years = max(
        (timestamps[-1] - timestamps[0]).total_seconds() / (365.25 * 86400),
        1 / 365.25,
    )
    bars_per_year = len(values) / years
    volatility = values.std(ddof=1) if len(values) > 1 else float("nan")
    equity = np.exp(np.cumsum(values))
    peaks = np.maximum.accumulate(np.r_[1.0, equity])
    drawdown = np.r_[1.0, equity] / peaks - 1.0
    return {
        "total_return": float(np.exp(values.sum()) - 1.0),
        "annualized_log_return": float(values.sum() / years),
        "sharpe": (
            float(values.mean() / volatility * np.sqrt(bars_per_year))
            if volatility > 0
            else None
        ),
        "max_drawdown": float(drawdown.min()),
        "bar_exposure": float(np.mean(position[usable] != 0.0)),
        "fee_sides": float(fee_sides[usable].sum() + closing_fee_sides),
        "commission_per_side": commission_per_side,
    }
