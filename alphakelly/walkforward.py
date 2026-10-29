"""Purged walk-forward that emits weights instead of calls.

The protocol is ``alphabarrier``'s, unchanged where it does not have to change,
because that protocol is the part of the older project most worth keeping:
anchored training, a purged twelve-month calibration window ahead of each test
quarter, a label-horizon embargo on both sides of it, and the whole thing re-run
on every calendar shift of the fold grid so that the spread across shifts can be
read as a noise floor.

Three things differ. The purge and embargo are the longest forecast horizon,
48 bars, rather than the 12-bar label horizon, because the targets reach
further. The calibration window measures a forecast scale and a risk level
instead of a decision threshold. And there is no threshold at all, so there is
no accuracy: a fold's output is a weight for every bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphabarrier.evaluation import Window, make_windows
from alphakelly.data import (
    HORIZONS,
    MAX_HORIZON,
    SPOT_TARGET,
    VOL_COLUMN,
    next_bar_return,
)
from alphakelly.forecast import (
    RIDGE_ALPHA_LADDER,
    calibrate_forecast_scale,
    fit_drift_forecaster,
)
from alphakelly.sizing import (
    LEVERAGE_CAP,
    RISK_BUDGET,
    SMOOTHING_LADDER,
    FoldScaling,
    fold_scaling,
)


BAR_HOURS = 4
STEP_MONTHS = 3
CALIBRATION_MONTHS = 12
FOLD_GRID_OFFSETS = tuple(range(STEP_MONTHS))
PURGE_BARS = MAX_HORIZON
EMBARGO = pd.Timedelta(hours=BAR_HOURS * PURGE_BARS)
MIN_TEST_BARS = 50
DEVELOPMENT_START = pd.Timestamp("2021-07-01")
LATER_EVALUATION_START = pd.Timestamp("2025-07-01")
DEVELOPMENT_SIGNAL_END = LATER_EVALUATION_START - EMBARGO
REPLICATION_START = pd.Timestamp("2014-01-01")


@dataclass(frozen=True)
class ForecastSpec:
    """One fully specified way of turning features into a raw Kelly weight.

    Every field here is an ablation the study reports, and each one is a
    complete re-run: dropping the intercept or the volatility denominator
    changes what the calibration window measures, so the forecast scale and the
    risk level are re-derived rather than rescaled.
    """

    horizons: tuple[int, ...] = HORIZONS
    alphas: tuple[float, ...] = RIDGE_ALPHA_LADDER
    use_intercept: bool = True
    volatility_scaling: bool = True

    def label(self) -> str:
        parts = [f"h{'-'.join(str(h) for h in self.horizons)}"]
        if not self.use_intercept:
            parts.append("no_intercept")
        if not self.volatility_scaling:
            parts.append("no_vol_scaling")
        return "+".join(parts)


@dataclass(frozen=True)
class FoldReport:
    """What each fold measured, before any position was taken."""

    fold: int
    test_start: str
    train_rows: int
    calibration_rows: int
    forecast_scale: float
    calibration_information_coefficient: float
    stood_aside: bool
    test_bars: int


@dataclass
class WalkForward:
    """Signals on the test bars plus the calibration arrays behind them."""

    folds: list[FoldReport]
    signals: pd.DataFrame
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]]

    def scalings(
        self,
        *,
        risk_budget: float = RISK_BUDGET,
        cap: float = LEVERAGE_CAP,
        allow_short: bool = False,
        smoothing_spans: tuple[float, ...] | float = SMOOTHING_LADDER,
    ) -> dict[int, FoldScaling]:
        """Re-solve every fold's level and step size for one configuration."""
        return {
            fold: fold_scaling(
                fold,
                raw,
                returns,
                risk_budget=risk_budget,
                cap=cap,
                allow_short=allow_short,
                smoothing_spans=smoothing_spans,
            )
            for fold, (raw, returns) in self.calibrations.items()
        }


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index < end)]


def offset_windows(
    month_offset: int,
    *,
    first_test: pd.Timestamp = DEVELOPMENT_START,
    last_test: pd.Timestamp = DEVELOPMENT_SIGNAL_END,
    step_months: int = STEP_MONTHS,
) -> list[Window]:
    """The walk-forward grid shifted by an arbitrary whole number of months."""
    return make_windows(
        first_test=str(first_test + pd.DateOffset(months=month_offset)),
        last_test=last_test,
        step_months=step_months,
        purge_bars=PURGE_BARS,
        bar_hours=BAR_HOURS,
    )


def raw_kelly_weight(
    drift: np.ndarray,
    volatility: np.ndarray,
    scale: float,
    *,
    volatility_scaling: bool = True,
) -> np.ndarray:
    """``w = mu / sigma^2`` written in the units the forecast actually has.

    ``drift`` is a per-bar expected move in volatility units, so
    ``mu = scale * drift * sigma`` and the Kelly weight is
    ``scale * drift / sigma``. A non-positive ``scale`` means the fold's own
    calibration window says the forecast is inverted, and the fold stands aside.

    With ``volatility_scaling`` off the denominator is dropped and the weight is
    proportional to the forecast alone. That is not Kelly, and it is reported
    because on this asset the difference between the two is the single largest
    effect in the study.
    """
    if scale <= 0.0:
        return np.zeros(len(drift), dtype=float)
    if not volatility_scaling:
        return scale * drift
    safe = np.where(np.isfinite(volatility) & (volatility > 1e-8), volatility, np.nan)
    return scale * drift / safe


def evaluate(
    frame: pd.DataFrame,
    features: list[str],
    windows: list[Window],
    *,
    spec: "ForecastSpec | None" = None,
    calibration_months: int = CALIBRATION_MONTHS,
) -> WalkForward:
    """Fit, calibrate and size every bar of every test window."""
    spec = spec or ForecastSpec()
    horizons, alphas = spec.horizons, spec.alphas
    forward = next_bar_return(frame)
    reports: list[FoldReport] = []
    signal_parts: list[pd.DataFrame] = []
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for index, window in enumerate(windows):
        calibration_start = window.test_start - pd.DateOffset(months=calibration_months)
        calibration_end = window.test_start - EMBARGO
        inner_train = _slice(frame, window.train_start, calibration_start - EMBARGO)
        calibration = _slice(frame, calibration_start, calibration_end)
        outer_train = _slice(frame, window.train_start, window.train_end)
        test = _slice(frame, window.test_start, window.test_end)
        if len(test) < MIN_TEST_BARS or len(calibration) < 100:
            continue
        inner = fit_drift_forecaster(
            inner_train, features, horizons=horizons, alphas=alphas
        )
        outer = fit_drift_forecaster(
            outer_train, features, horizons=horizons, alphas=alphas
        )
        if inner is None or outer is None:
            continue

        calibration_drift = inner.drift(calibration, use_intercept=spec.use_intercept)
        scale, information = calibrate_forecast_scale(
            calibration_drift, calibration[SPOT_TARGET].to_numpy(dtype=float)
        )
        calibration_raw = raw_kelly_weight(
            calibration_drift,
            calibration[VOL_COLUMN].to_numpy(dtype=float),
            scale,
            volatility_scaling=spec.volatility_scaling,
        )
        calibrations[index] = (
            calibration_raw,
            forward.reindex(calibration.index).to_numpy(dtype=float),
        )

        test_drift = outer.drift(test, use_intercept=spec.use_intercept)
        test_volatility = test[VOL_COLUMN].to_numpy(dtype=float)
        signal_parts.append(
            pd.DataFrame(
                {
                    "fold": index,
                    "drift": test_drift,
                    "volatility": test_volatility,
                    "forecast_scale": scale,
                    "raw_kelly": raw_kelly_weight(
                        test_drift,
                        test_volatility,
                        scale,
                        volatility_scaling=spec.volatility_scaling,
                    ),
                },
                index=test.index,
            )
        )
        reports.append(
            FoldReport(
                fold=index,
                test_start=str(window.test_start),
                train_rows=outer.rows,
                calibration_rows=int(np.isfinite(calibration_drift).sum()),
                forecast_scale=float(scale),
                calibration_information_coefficient=float(information),
                stood_aside=bool(scale <= 0.0),
                test_bars=len(test),
            )
        )

    if not reports:
        raise RuntimeError("no walk-forward windows were evaluated")
    return WalkForward(
        folds=reports,
        signals=pd.concat(signal_parts).sort_index(),
        calibrations=calibrations,
    )
