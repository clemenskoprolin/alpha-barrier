"""Turn a drift forecast into a position by maximizing expected log wealth.

The whole design follows from one line. If a bar's excess return has mean
``mu`` and variance ``sigma^2``, the weight that maximizes the expected growth
rate of wealth is ``w = mu / sigma^2``. Everything ``alphabarrier`` bolted on by
hand is either derived from that or removed by it:

* Volatility scaling is not an overlay. It is the denominator.
* The sizing map is not a squashed log-odds offset with a hand-set floor. It is
  linear in the forecast, and it reaches zero on its own when the forecast does.
* Direction is not a threshold decision. A weak call is a small weight.

Two things Kelly does not supply are supplied by measurement rather than by
choice. The *level* comes from a per-fold risk budget: the position is scaled so
that on the fold's own purged calibration window it runs at the same volatility
as buying and holding the asset. That makes "more return" mean more return at
equal risk rather than more return from more leverage, and it removes both the
Kelly fraction and the volatility target as free constants. The *no-trade band*
comes from balancing the growth given up by tracking the target loosely against
the commissions charged for tracking it tightly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphakelly.data import HORIZONS


# The ladder weights horizon h by 1/sqrt(h), so the forecast it produces has an
# effective horizon of sum(sqrt(h)) / sum(1/sqrt(h)) bars. For (3, 6, 12, 24,
# 48) that is exactly 12.0. It is a property of the ladder rather than a fitted
# quantity, and it is what the no-trade band's holding period is scaled to.
EFFECTIVE_HORIZON = float(np.sqrt(HORIZONS).sum() / (1.0 / np.sqrt(HORIZONS)).sum())
# Smoothing spans. Sweeping a single span produces no plateau -- 6 bars beats
# both 1 and 12 in every period -- which is exactly the shape that produced
# alphabarrier's 720-bar trend filter, and picking the peak would repeat that
# mistake. The spans are therefore averaged, and they are the forecast horizons
# themselves, so the ladder introduces no numbers of its own.
SMOOTHING_LADDER = tuple(float(horizon) for horizon in HORIZONS)
# Same volatility as holding the asset. 1.0 is the point of the exercise: beat
# buy and hold on return without taking more risk than buy and hold does.
RISK_BUDGET = 1.0
# Nothing here runs at more than four times notional. The cap exists so that a
# collapsing volatility estimate cannot ask for infinite size; the share of bars
# on which it binds is reported rather than assumed small.
LEVERAGE_CAP = 4.0
BAND_MODES = ("cost_aware", "fixed", "none")
# Inherited unchanged from alphabarrier, where it was swept. Used only by the
# "fixed" comparison band.
FIXED_BAND_FRACTION = 0.10
COMMISSION_PER_SIDE = 0.001


@dataclass(frozen=True)
class FoldScaling:
    """Everything about a fold's sizing that is measured, not chosen."""

    fold: int
    level: float
    step_sd: float
    calibration_mean_weight: float
    calibration_cap_share: float
    calibration_position_sd: float
    calibration_asset_sd: float


def smooth(values: pd.Series, spans: tuple[float, ...] | float) -> pd.Series:
    """Average an EWMA ladder, or apply a single span when one is given."""
    ladder = (spans,) if isinstance(spans, (int, float)) else tuple(spans)
    if not ladder:
        raise ValueError("spans must not be empty")
    if any(span < 1.0 for span in ladder):
        raise ValueError("spans must be at least 1")
    return sum(
        values.ewm(span=span, adjust=False).mean() if span > 1.0 else values
        for span in ladder
    ) / float(len(ladder))


def smooth_by_fold(
    size: pd.Series, folds: pd.Series, spans: tuple[float, ...] | float
) -> pd.Series:
    """Smooth within each fold, so one fold's calls never leak into the next."""
    parts = [
        smooth(part, spans)
        for _, part in size.groupby(folds.ne(folds.shift()).cumsum(), sort=False)
    ]
    return pd.concat(parts).sort_index()


def clip_weights(weights: np.ndarray, *, cap: float, allow_short: bool) -> np.ndarray:
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    return np.clip(weights, -cap if allow_short else 0.0, cap)


def cost_aware_band(
    volatility: np.ndarray,
    step_sd: np.ndarray | float,
    *,
    commission_per_side: float = COMMISSION_PER_SIDE,
) -> np.ndarray:
    """Half-width of the no-trade band that minimizes total growth lost.

    The target weight is not a fixed point to be reached once; it wanders as the
    forecast decays and refreshes, so this is a tracking problem. Let ``s`` be
    the per-bar standard deviation of changes in the target and ``b`` the band
    half-width. Two costs trade off against each other:

    * Tracking error. Inside the band the gap to the target behaves like a
      reflected random walk, with mean square roughly ``b^2 / 3``. Being off
      target by ``d`` gives up ``sigma^2 d^2 / 2`` of expected log wealth per
      bar, so the drag is about ``sigma^2 b^2 / 6``.
    * Commission. A random walk of step ``s`` inside a band of half-width ``b``
      touches an edge often enough to trade about ``s^2 / (2b)`` of notional per
      bar, costing ``c s^2 / (2b)``.

    Their sum is minimized at ``b = (3 c s^2 / (2 sigma^2))^(1/3)``, which is
    the cube-root band familiar from proportional-cost control problems. It
    introduces no constant: ``c`` is the fee schedule, ``s`` is measured on each
    fold's own purged calibration window, and ``sigma`` is the same volatility
    estimate the Kelly weight already divides by. It also behaves correctly on
    its own terms, widening when a quiet market makes the growth surface flat
    and tightening when a violent one makes it steep.
    """
    if commission_per_side < 0.0:
        raise ValueError("commission_per_side must be non-negative")
    steps = np.asarray(step_sd, dtype=float)
    if np.any(~np.isfinite(steps)) or np.any(steps < 0.0):
        raise ValueError("step_sd must be finite and non-negative")
    variance = np.maximum(np.asarray(volatility, dtype=float), 1e-8) ** 2
    return np.cbrt(1.5 * commission_per_side * steps**2 / variance)


def apply_rebalance_band(target: pd.Series, band: np.ndarray | float) -> pd.Series:
    """Hold the current weight until the target drifts past a per-bar band."""
    values = np.nan_to_num(target.to_numpy(dtype=float), nan=0.0)
    widths = np.broadcast_to(np.asarray(band, dtype=float), values.shape)
    if np.any(~np.isfinite(widths)) or np.any(widths < 0.0):
        raise ValueError("band must be finite and non-negative")
    held = np.empty(len(values), dtype=float)
    current = 0.0
    for offset in range(len(values)):
        if abs(values[offset] - current) >= widths[offset]:
            current = values[offset]
        held[offset] = current
    return pd.Series(held, index=target.index, name="position")


def _position_sd(
    smoothed: np.ndarray,
    returns: np.ndarray,
    level: float,
    *,
    cap: float,
    allow_short: bool,
) -> float:
    weights = clip_weights(smoothed * level, cap=cap, allow_short=allow_short)
    realized = weights * returns
    return float(realized.std(ddof=1)) if len(realized) > 1 else 0.0


def solve_risk_level(
    smoothed: np.ndarray,
    returns: np.ndarray,
    *,
    risk_budget: float = RISK_BUDGET,
    cap: float = LEVERAGE_CAP,
    allow_short: bool = False,
    iterations: int = 80,
) -> float:
    """Scale a Kelly path so the calibration window runs at a target volatility.

    ``risk_budget`` is measured in units of the asset's own volatility over the
    same window, so 1.0 means "take exactly the risk that buying and holding
    takes". The solve runs on the clipped path, so the long-only truncation and
    the leverage cap sit inside the target rather than leaking out of it.
    """
    if risk_budget <= 0.0:
        raise ValueError("risk_budget must be positive")
    usable = np.isfinite(smoothed) & np.isfinite(returns)
    smoothed, returns = smoothed[usable], returns[usable]
    if len(smoothed) < 30:
        return 0.0
    target = float(np.std(returns, ddof=1)) * risk_budget
    if not target > 0.0:
        return 0.0
    low, high = 1e-8, 1e8
    if _position_sd(smoothed, returns, high, cap=cap, allow_short=allow_short) <= target:
        return high
    if _position_sd(smoothed, returns, low, cap=cap, allow_short=allow_short) >= target:
        return low
    for _ in range(iterations):
        middle = float(np.sqrt(low * high))
        if _position_sd(
            smoothed, returns, middle, cap=cap, allow_short=allow_short
        ) < target:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def fold_scaling(
    fold: int,
    raw: np.ndarray,
    returns: np.ndarray,
    *,
    risk_budget: float = RISK_BUDGET,
    cap: float = LEVERAGE_CAP,
    allow_short: bool = False,
    smoothing_spans: tuple[float, ...] | float = SMOOTHING_LADDER,
) -> FoldScaling:
    """Measure a fold's level and target-step size on its calibration window.

    Both quantities depend on the risk budget, the cap and the smoothing, so
    every sweep re-solves them here instead of rescaling a stored answer.
    """
    smoothed = smooth(pd.Series(raw), smoothing_spans).to_numpy()
    level = solve_risk_level(
        smoothed,
        returns,
        risk_budget=risk_budget,
        cap=cap,
        allow_short=allow_short,
    )
    usable = np.isfinite(smoothed) & np.isfinite(returns)
    smoothed, returns = smoothed[usable], returns[usable]
    weights = clip_weights(smoothed * level, cap=cap, allow_short=allow_short)
    steps = np.diff(weights)
    return FoldScaling(
        fold=fold,
        level=level,
        step_sd=float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0,
        calibration_mean_weight=float(np.mean(weights)) if len(weights) else 0.0,
        calibration_cap_share=(
            float(np.mean(np.abs(weights) >= cap - 1e-9)) if len(weights) else 0.0
        ),
        calibration_position_sd=_position_sd(
            smoothed, returns, level, cap=cap, allow_short=allow_short
        ),
        calibration_asset_sd=(
            float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        ),
    )


def build_positions(
    signals: pd.DataFrame,
    scalings: dict[int, FoldScaling],
    *,
    cap: float = LEVERAGE_CAP,
    allow_short: bool = False,
    smoothing_spans: tuple[float, ...] | float = SMOOTHING_LADDER,
    band_mode: str = "cost_aware",
    band_fraction: float = FIXED_BAND_FRACTION,
    commission_per_side: float = COMMISSION_PER_SIDE,
    trend_overlay: pd.Series | None = None,
) -> pd.Series:
    """Assemble the traded weight path from a walk-forward's signal frame.

    ``signals`` carries the raw Kelly weight, the volatility estimate and the
    fold identifier for every test bar; ``scalings`` carries the two numbers
    measured on each fold's calibration window. Smoothing happens before the
    clip, so the leverage cap stays a hard bound rather than an averaged one.
    """
    if band_mode not in BAND_MODES:
        raise ValueError(f"band_mode must be one of {BAND_MODES}")
    required = {"raw_kelly", "volatility", "fold"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")

    folds = signals["fold"]
    levels = folds.map(lambda key: scalings[key].level).astype(float)
    smoothed = smooth_by_fold(signals["raw_kelly"] * levels, folds, smoothing_spans)
    weights = pd.Series(
        clip_weights(
            np.nan_to_num(smoothed.to_numpy(dtype=float), nan=0.0),
            cap=cap,
            allow_short=allow_short,
        ),
        index=signals.index,
    )
    if trend_overlay is not None:
        weights = weights * trend_overlay.reindex(signals.index).fillna(0.0)
    if band_mode == "none":
        return weights.rename("position")
    if band_mode == "fixed":
        band = band_fraction * float(np.mean(np.abs(weights.to_numpy(dtype=float))))
    else:
        band = cost_aware_band(
            signals["volatility"].to_numpy(dtype=float),
            folds.map(lambda key: scalings[key].step_sd).to_numpy(dtype=float),
            commission_per_side=commission_per_side,
        )
    return apply_rebalance_band(weights, band)
