"""Multi-horizon ridge drift forecaster.

``alphabarrier`` fits one balanced logistic regression at a hand-picked ``C``.
Two things change here. The output has to be a magnitude rather than a class,
so the model is a ridge; and the penalty is averaged over a ladder rather than
chosen, which is safe in a way it was not for the classifier: a per-fold scale
is measured downstream on a purged calibration window, so the penalty only sets
the *shape* of the forecast and the calibration sets its size. Over-shrinking is
undone; picking wrong is not.

Every horizon is fit on the identical design matrix, so one singular value
decomposition serves the whole ladder and every penalty on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphakelly.data import FORWARD_PREFIX, HORIZONS


# Multiples of the row count, so the penalty means the same thing whether a
# fold trains on two years or on thirteen. Two decades of shrinkage.
RIDGE_ALPHA_LADDER = (0.01, 0.03, 0.1, 0.3, 1.0)
MIN_TRAIN_ROWS = 500


@dataclass(frozen=True)
class DriftForecaster:
    """A fitted ladder of ridge regressions, one per forecast horizon."""

    features: tuple[str, ...]
    horizons: tuple[int, ...]
    medians: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    rows: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Forecast every horizon for every row, imputing gaps with train medians."""
        values = frame.loc[:, list(self.features)].to_numpy(dtype=float).copy()
        missing = np.where(~np.isfinite(values))
        values[missing] = np.take(self.medians, missing[1])
        standardized = (values - self.center) / self.scale
        return self.intercept + standardized @ self.coefficients.T

    def drift(self, frame: pd.DataFrame, *, use_intercept: bool = True) -> np.ndarray:
        """Collapse the horizon ladder into one per-bar drift, in sigma units.

        ``y_h`` is a forward return over ``h`` bars measured in ``sigma*sqrt(h)``,
        so ``y_h / sqrt(h)`` is the per-bar drift over sigma however long the
        horizon. With ``use_intercept`` off the training window's unconditional
        mean is removed first, leaving a pure timing forecast with no structural
        long tilt. The ladder members are estimates of one quantity, and this is
        their equal-weight average. A horizon that carries no information is
        shrunk toward zero by its own ridge before it reaches this average, so
        equal weighting does not hand an uninformative horizon equal say.
        """
        predictions = self.predict(frame)
        if not use_intercept:
            predictions = predictions - self.intercept
        return combine_horizons(predictions, self.horizons)


def combine_horizons(predictions: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    if predictions.shape[1] != len(horizons):
        raise ValueError("predictions and horizons disagree in width")
    return np.mean(predictions / np.sqrt(horizons), axis=1)


def target_columns(horizons: tuple[int, ...] = HORIZONS) -> list[str]:
    return [f"{FORWARD_PREFIX}{horizon}" for horizon in horizons]


def training_rows(
    frame: pd.DataFrame, features: list[str], horizons: tuple[int, ...]
) -> pd.DataFrame:
    """Rows with every feature and every horizon's target observed.

    Requiring all horizons keeps the design matrix identical across the ladder,
    which is what lets one decomposition serve every fit. It also purges the
    training tail by the longest horizon rather than the shortest, which is
    stricter than purging each horizon separately.
    """
    return frame.dropna(subset=list(features) + target_columns(horizons))


def fit_drift_forecaster(
    frame: pd.DataFrame,
    features: list[str],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    alphas: tuple[float, ...] = RIDGE_ALPHA_LADDER,
    min_rows: int = MIN_TRAIN_ROWS,
) -> DriftForecaster | None:
    """Fit every horizon at every penalty, and average the penalties.

    Averaging predictions across penalties is the same as averaging
    coefficients, so the ladder costs one extra matrix product rather than one
    extra fit.
    """
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must be positive")
    train = training_rows(frame, features, horizons)
    if len(train) < min_rows:
        return None
    design = train.loc[:, features].to_numpy(dtype=float)
    targets = train.loc[:, target_columns(horizons)].to_numpy(dtype=float)
    medians = np.nanmedian(design, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    center = design.mean(axis=0)
    scale = design.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (design - center) / scale
    intercept = targets.mean(axis=0)

    left, singular, right = np.linalg.svd(standardized, full_matrices=False)
    projected = left.T @ (targets - intercept)
    coefficients = np.zeros((len(horizons), len(features)))
    for alpha in alphas:
        filtered = singular / (singular**2 + alpha * len(standardized))
        coefficients += (right.T @ (filtered[:, None] * projected)).T
    coefficients /= len(alphas)
    return DriftForecaster(
        features=tuple(features),
        horizons=tuple(horizons),
        medians=medians,
        center=center,
        scale=scale,
        intercept=intercept,
        coefficients=coefficients,
        rows=len(train),
    )


def calibrate_forecast_scale(
    drift: np.ndarray, spot_target: np.ndarray
) -> tuple[float, float]:
    """Regress the realized one-bar move on the forecast, through the origin.

    A ridge forecast is shrunk by construction, so its own magnitude is not an
    expected return and cannot be fed to a Kelly formula. One scalar per fold,
    measured on the purged calibration window, restores the units: after it,
    ``scale * drift`` is an estimate of next-bar return over volatility.

    Its sign is also the only check this design has on itself. A fold whose
    calibration window says the forecast is inverted returns a non-positive
    scale, and the caller stands aside for that quarter.
    """
    usable = np.isfinite(drift) & np.isfinite(spot_target)
    if usable.sum() < 30:
        return 0.0, float("nan")
    forecast, realized = drift[usable], spot_target[usable]
    energy = float(forecast @ forecast)
    if energy <= 1e-18:
        return 0.0, float("nan")
    scale = float(forecast @ realized) / energy
    deviation = forecast.std(ddof=1) * realized.std(ddof=1)
    correlation = (
        float(np.mean((forecast - forecast.mean()) * (realized - realized.mean())))
        / deviation
        if deviation > 0
        else float("nan")
    )
    return scale, correlation
