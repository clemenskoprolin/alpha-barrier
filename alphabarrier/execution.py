"""Alpha Barrier execution rules, reference baselines, and fee sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphabarrier.features import LABEL_HORIZON_BARS


CONFIDENCE_FLOOR = 0.02
TREND_EMA_SPAN = 720
TREND_EMA_LADDER = (90, 180, 360, 720, 1440, 2160)
COMMISSION_PER_SIDE = 0.001
SIGNAL_EWMA_SPAN = LABEL_HORIZON_BARS
REBALANCE_BAND = 0.10
# Flat when the call sits a full calibration log-odds spread below the fold's
# own threshold. sigmoid(-1) is that point in the units the sizing map emits,
# so the floor introduces no constant of its own.
SIZING_FLOOR = 1.0 / (1.0 + np.e)
FEE_SENSITIVITY_GRID = (0.0, 0.0005, 0.001, 0.002, 0.004)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def trend_state(frame: pd.DataFrame, *, trend_ema_span: int = TREND_EMA_SPAN) -> pd.Series:
    """Causal long-trend gate: close above its own exponential moving average."""
    if trend_ema_span < 1:
        raise ValueError("trend_ema_span must be positive")
    average = frame["raw_close"].ewm(
        span=trend_ema_span, adjust=False, min_periods=trend_ema_span
    ).mean()
    return frame["raw_close"].gt(average).astype(float)


def trend_gate(
    frame: pd.DataFrame, spans: int | tuple[int, ...] = TREND_EMA_LADDER
) -> pd.Series:
    """Graded long-trend gate: the share of an EMA ladder the close sits above.

    No lookback is defensible on its own. Sweeping the single 720-bar span the
    strategy used to gate on moves Sharpe between 0.70 and 1.22 with no plateau,
    which makes any one choice a draw from that spread rather than a finding.
    Averaging a geometric ladder spanning fifteen days to a year replaces the
    choice with the whole range, and cuts the sensitivity to it roughly
    fourfold. Passing a single span reproduces the old binary gate.
    """
    ladder = (spans,) if isinstance(spans, int) else tuple(spans)
    if not ladder:
        raise ValueError("spans must not be empty")
    states = [trend_state(frame, trend_ema_span=span) for span in ladder]
    return sum(states) / float(len(ladder))


def build_strategy_positions(
    signals: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    confidence_floor: float = CONFIDENCE_FLOOR,
    trend_ema_span: int = TREND_EMA_SPAN,
) -> pd.Series:
    """Build the original binary long-only, hysteretic, trend-gated position path.

    A confident up call enters long and a confident down call exits. Weak calls
    leave the model state unchanged. The final position is long only while the
    close is above its causal 720-bar exponential moving average. Retained as the
    published comparison point for :func:`build_scaled_positions`.
    """
    if confidence_floor < 0.0:
        raise ValueError("confidence_floor must be non-negative")
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

    trend = trend_state(frame, trend_ema_span=trend_ema_span)
    positions = model_state * trend.reindex(signals.index).fillna(0.0).to_numpy()
    return pd.Series(positions, index=signals.index, name="position")


def _smooth_by_fold(size: pd.Series, spans: pd.Series) -> pd.Series:
    """EWMA each fold's calls over that fold's own measured label lifetime."""
    parts = [
        part.ewm(span=max(1.0, float(spans.loc[part.index[0]])), adjust=False).mean()
        for _, part in size.groupby(spans.ne(spans.shift()).cumsum(), sort=False)
    ]
    return pd.concat(parts).sort_index()


def signal_size(
    signals: pd.DataFrame,
    *,
    ewma_span: int | None = None,
    floor: float = SIZING_FLOOR,
) -> pd.Series:
    """Map each call onto a continuous size in ``[0, 1]``.

    The probability is expressed as a log-odds offset from its own fold
    threshold, divided by the log-odds spread measured on that fold's purged
    calibration window, and squashed. The result is scale-free and centred on
    0.5 at the threshold.

    It is then smoothed over the lifetime of the thing being predicted. That is
    each fold's ``signal_span``, the mean bars its calibration window took to
    actually touch a barrier -- about five, not the twelve-bar horizon, which is
    only a ceiling. ``ewma_span`` overrides that with a fixed span for sweeps.

    Finally ``floor`` is subtracted and the remainder rescaled, so a call far
    enough below the fold threshold sizes to zero instead of to a small
    residual. Being able to reach flat is the one thing the binary gate this
    rule replaced had over it, and on history outside the development window it
    was worth more than the grading.
    """
    if ewma_span is not None and ewma_span < 1:
        raise ValueError("ewma_span must be positive")
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor must be in [0, 1)")
    required = {"probability", "threshold", "logit_scale"}
    if ewma_span is None:
        required.add("signal_span")
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")

    offset = _logit(signals["probability"].to_numpy(dtype=float)) - _logit(
        signals["threshold"].to_numpy(dtype=float)
    )
    scale = np.maximum(signals["logit_scale"].to_numpy(dtype=float), 1e-6)
    size = pd.Series(
        1.0 / (1.0 + np.exp(-np.clip(offset / scale, -30.0, 30.0))),
        index=signals.index,
    )
    if ewma_span is None:
        smoothed = _smooth_by_fold(size, signals["signal_span"])
    else:
        smoothed = size.ewm(span=ewma_span, adjust=False).mean()
    smoothed = smoothed.fillna(0.5)
    if floor:
        smoothed = ((smoothed - floor) / (1.0 - floor)).clip(0.0, 1.0)
    return smoothed.rename("size")


def apply_rebalance_band(target: pd.Series, band: float = REBALANCE_BAND) -> pd.Series:
    """Hold the current position until the target drifts by at least ``band``."""
    if band < 0.0:
        raise ValueError("band must be non-negative")
    if not band:
        return target.rename("position")
    values = target.to_numpy(dtype=float)
    held = np.zeros(len(values), dtype=float)
    current = 0.0
    for offset, value in enumerate(values):
        if abs(value - current) >= band:
            current = value
        held[offset] = current
    return pd.Series(held, index=target.index, name="position")


def build_scaled_positions(
    signals: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    ewma_span: int | None = None,
    rebalance_band: float = REBALANCE_BAND,
    floor: float = SIZING_FLOOR,
    trend_spans: int | tuple[int, ...] = TREND_EMA_LADDER,
) -> pd.Series:
    """Build the continuous long-only, trend-gated position path.

    The call sizes the position instead of switching it, but can still size it
    to zero. The trend gate is graded across an EMA ladder rather than switched
    at one arbitrary lookback, and the rebalance band converts the remaining
    small moves into no trade at all.
    """
    size = signal_size(signals, ewma_span=ewma_span, floor=floor)
    trend = trend_gate(frame, trend_spans).reindex(signals.index)
    target = (size * trend.fillna(0.0)).clip(0.0, 1.0)
    return apply_rebalance_band(target, rebalance_band)


def position_log_returns(
    positions: pd.Series,
    frame: pd.DataFrame,
    *,
    commission_per_side: float = COMMISSION_PER_SIDE,
) -> tuple[pd.Series, np.ndarray, float]:
    """Per-bar net log returns of a position path, with its weights and fee sides.

    A position established at a bar close earns the following close-to-close
    return. The final position is explicitly closed. Slippage, spread, borrow,
    and funding are not modeled.
    """
    if commission_per_side < 0.0:
        raise ValueError("commission_per_side must be non-negative")
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
    returns = pd.Series(np.log1p(simple[usable]), index=frame.index[usable])
    return returns, position[usable], float(fee_sides[usable].sum() + closing_fee_sides)


def continuous_backtest(
    positions: pd.Series,
    frame: pd.DataFrame,
    *,
    commission_per_side: float = COMMISSION_PER_SIDE,
) -> dict[str, float | None]:
    """Backtest the supplied position path, charging only actual turnover."""
    if positions.empty:
        return {
            "total_return": None,
            "annualized_log_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "bar_exposure": 0.0,
            "mean_weight": 0.0,
            "fee_sides": 0.0,
            "commission_per_side": commission_per_side,
        }
    returns, weights, fees = position_log_returns(
        positions, frame, commission_per_side=commission_per_side
    )
    values = returns.to_numpy()
    timestamps = returns.index
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
        "bar_exposure": float(np.mean(weights != 0.0)),
        "mean_weight": float(np.mean(weights)),
        "fee_sides": fees,
        "commission_per_side": commission_per_side,
    }


def reference_baselines(
    index: pd.DatetimeIndex,
    frame: pd.DataFrame,
    *,
    signals: pd.DataFrame | None = None,
    trend_spans: int | tuple[int, ...] = TREND_EMA_LADDER,
    trend_ema_span: int = TREND_EMA_SPAN,
    commission_per_side: float = COMMISSION_PER_SIDE,
) -> dict[str, dict[str, float | None]]:
    """Backtest the do-nothing comparisons over the identical index and fees.

    Every reported execution result must be read against these. ``trend_only``
    is the gate the current rule sits on and is therefore the number that rule
    has to beat; ``trend_only_single_span`` is the old 720-bar gate the binary
    variant still uses. ``model_only`` drops the gate entirely so the
    classifier's own contribution is visible.
    """
    baselines = {
        "buy_and_hold": pd.Series(1.0, index=index),
        "trend_only": trend_gate(frame, trend_spans).reindex(index).fillna(0.0),
        "trend_only_single_span": trend_state(
            frame, trend_ema_span=trend_ema_span
        ).reindex(index).fillna(0.0),
    }
    if signals is not None:
        baselines["model_only_binary"] = _model_state(signals)
        baselines["model_only_scaled"] = signal_size(signals)
    return {
        name: continuous_backtest(
            position, frame, commission_per_side=commission_per_side
        )
        for name, position in baselines.items()
    }


def _model_state(
    signals: pd.DataFrame, *, confidence_floor: float = CONFIDENCE_FLOOR
) -> pd.Series:
    """The binary model state before the trend gate is applied."""
    state = np.zeros(len(signals), dtype=float)
    previous = 0.0
    side = signals["side"].to_numpy(dtype=float)
    confidence = signals["confidence"].to_numpy(dtype=float)
    for offset in range(len(signals)):
        if not np.isfinite(side[offset]):
            previous = 0.0
        elif confidence[offset] >= confidence_floor:
            previous = 1.0 if side[offset] > 0.0 else 0.0
        state[offset] = previous
    return pd.Series(state, index=signals.index, name="position")


def fee_sensitivity(
    positions: pd.Series,
    frame: pd.DataFrame,
    *,
    grid: tuple[float, ...] = FEE_SENSITIVITY_GRID,
) -> list[dict[str, float | None]]:
    """Re-price one position path across a grid of per-side commissions."""
    rows = []
    for commission in grid:
        result = continuous_backtest(positions, frame, commission_per_side=commission)
        rows.append(
            {
                "commission_per_side": commission,
                "total_return": result["total_return"],
                "sharpe": result["sharpe"],
            }
        )
    return rows
