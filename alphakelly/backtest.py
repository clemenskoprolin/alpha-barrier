"""Backtest a weight path that is allowed to leave [0, 1].

``alphabarrier.execution`` charges commission on turnover and nothing else,
which is right for a rule whose weight never exceeds one. This one can lever, so
two things have to be paid for that the classifier never owed: financing on
borrowed notional, and, in the long/short variant, borrow on the short leg.
Leaving them out is the standard way a leveraged backtest invents money.

Reported metrics lead with growth rather than with accuracy or Sharpe, because
growth is the objective: terminal wealth multiple, compound annual growth rate,
and the annualized log return that is literally the quantity Kelly maximizes.
Sharpe, drawdown and the Ulcer index are reported next to them so that a return
bought with risk is visible as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


COMMISSION_PER_SIDE = 0.001
# Charged on borrowed notional only. Crypto margin and perpetual funding have
# averaged well inside this over the sample; the sensitivity sweep covers zero
# to four times it.
FINANCING_ANNUAL = 0.10
BAR_HOURS = 4
BARS_PER_YEAR = 365.25 * 24 / BAR_HOURS
FEE_SENSITIVITY_GRID = (0.0, 0.0005, 0.001, 0.002, 0.004)
FINANCING_SENSITIVITY_GRID = (0.0, 0.05, 0.10, 0.20, 0.40)


def position_returns(
    positions: pd.Series,
    frame: pd.DataFrame,
    *,
    commission_per_side: float = COMMISSION_PER_SIDE,
    financing_annual: float = FINANCING_ANNUAL,
    bar_hours: int = BAR_HOURS,
) -> tuple[pd.Series, np.ndarray, dict[str, float]]:
    """Per-bar net log returns of a weight path, with its weights and costs.

    A weight established at a bar close earns the following close-to-close
    return. The final weight is explicitly closed. Borrowed notional is
    ``max(w - 1, 0)`` long and ``max(-w, 0)`` short, and is charged pro rata.
    Slippage, spread, funding basis, margin calls and liquidation are not
    modeled; at four times notional the last two are not negligible in reality.
    """
    if commission_per_side < 0.0:
        raise ValueError("commission_per_side must be non-negative")
    if financing_annual < 0.0:
        raise ValueError("financing_annual must be non-negative")
    aligned = positions.reindex(frame.index)
    offsets = np.flatnonzero(aligned.notna().to_numpy())
    if not len(offsets):
        raise ValueError("no position timestamps overlap frame")
    first, last = int(offsets[0]), int(offsets[-1])
    closes = frame["raw_close"].astype(float)
    next_return = (closes.shift(-1) / closes - 1.0).to_numpy()
    if last + 1 >= len(frame) or not np.isfinite(next_return[last]):
        raise ValueError("frame needs one valid close after its final position")

    weight = aligned.fillna(0.0).to_numpy(dtype=float)
    weight[:first] = 0.0
    weight[last + 1 :] = 0.0
    previous = np.r_[0.0, weight[:-1]]
    traded = np.abs(weight - previous)
    borrowed = np.maximum(weight - 1.0, 0.0) + np.maximum(-weight, 0.0)
    financing = borrowed * financing_annual * bar_hours / (365.25 * 24)
    simple = weight * next_return - commission_per_side * traded - financing
    closing = abs(weight[last])
    simple[last] -= commission_per_side * closing

    span = np.arange(len(frame))
    usable = (span >= first) & (span <= last) & np.isfinite(simple)
    returns = pd.Series(np.log1p(simple[usable]), index=frame.index[usable])
    costs = {
        "fee_sides": float(traded[usable].sum() + closing),
        "commission_paid": float(
            commission_per_side * (traded[usable].sum() + closing)
        ),
        "financing_paid": float(financing[usable].sum()),
    }
    return returns, weight[usable], costs


def growth_metrics(returns: pd.Series, weights: np.ndarray) -> dict[str, float | None]:
    values = returns.to_numpy()
    stamps = returns.index
    years = max((stamps[-1] - stamps[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    bars_per_year = len(values) / years
    total_log = float(values.sum())
    equity = np.exp(np.cumsum(values))
    peaks = np.maximum.accumulate(np.r_[1.0, equity])
    drawdown = np.r_[1.0, equity] / peaks - 1.0
    deviation = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    max_drawdown = float(drawdown.min())
    annualized_log = total_log / years
    compound_annual = float(np.expm1(annualized_log))
    return {
        "terminal_multiple": float(np.exp(total_log)),
        "total_return": float(np.expm1(total_log)),
        "compound_annual_growth": compound_annual,
        "annualized_log_return": float(annualized_log),
        "annualized_volatility": (
            float(deviation * np.sqrt(bars_per_year)) if deviation > 0 else None
        ),
        "sharpe": (
            float(values.mean() / deviation * np.sqrt(bars_per_year))
            if deviation > 0
            else None
        ),
        "sortino": (
            float(values.mean() / downside * np.sqrt(bars_per_year))
            if downside > 0
            else None
        ),
        "calmar": (
            float(compound_annual / abs(max_drawdown)) if max_drawdown < 0 else None
        ),
        "max_drawdown": max_drawdown,
        "ulcer_index": float(np.sqrt(np.mean(drawdown**2))),
        "bar_exposure": float(np.mean(weights != 0.0)),
        "mean_weight": float(np.mean(weights)),
        "mean_absolute_weight": float(np.mean(np.abs(weights))),
        "max_weight": float(np.max(weights)),
        "min_weight": float(np.min(weights)),
        "bars": int(len(values)),
        "years": float(years),
    }


def backtest(
    positions: pd.Series,
    frame: pd.DataFrame,
    *,
    commission_per_side: float = COMMISSION_PER_SIDE,
    financing_annual: float = FINANCING_ANNUAL,
) -> dict[str, float | None]:
    """Growth, risk and cost summary for one weight path."""
    if positions.empty:
        return {"terminal_multiple": None, "total_return": None, "sharpe": None}
    returns, weights, costs = position_returns(
        positions,
        frame,
        commission_per_side=commission_per_side,
        financing_annual=financing_annual,
    )
    summary = growth_metrics(returns, weights)
    summary.update(costs)
    summary["commission_per_side"] = commission_per_side
    summary["financing_annual"] = financing_annual
    return summary


def headline(result: dict[str, float | None]) -> dict[str, float | None]:
    """The five numbers every comparison table in this project reports."""
    return {
        key: result.get(key)
        for key in (
            "total_return",
            "compound_annual_growth",
            "sharpe",
            "max_drawdown",
            "mean_weight",
        )
    }


def cost_sensitivity(
    positions: pd.Series,
    frame: pd.DataFrame,
    *,
    fee_grid: tuple[float, ...] = FEE_SENSITIVITY_GRID,
    financing_grid: tuple[float, ...] = FINANCING_SENSITIVITY_GRID,
) -> dict[str, list[dict[str, float | None]]]:
    """Re-price one path across commission and financing grids, one at a time."""
    fees = [
        {
            "commission_per_side": commission,
            **{
                key: value
                for key, value in backtest(
                    positions, frame, commission_per_side=commission
                ).items()
                if key in {"total_return", "compound_annual_growth", "sharpe"}
            },
        }
        for commission in fee_grid
    ]
    financing = [
        {
            "financing_annual": rate,
            **{
                key: value
                for key, value in backtest(
                    positions, frame, financing_annual=rate
                ).items()
                if key in {"total_return", "compound_annual_growth", "sharpe"}
            },
        }
        for rate in financing_grid
    ]
    return {"commission": fees, "financing": financing}
