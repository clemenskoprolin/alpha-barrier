"""Purged walk-forward evaluation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(
    *,
    first_test: str,
    last_test: pd.Timestamp,
    step_months: int = 3,
    purge_bars: int = 12,
    bar_hours: int = 4,
    min_train_start: str = "2012-01-01",
) -> list[Window]:
    windows: list[Window] = []
    test_start = pd.Timestamp(first_test)
    while test_start < last_test:
        test_end = test_start + pd.DateOffset(months=step_months)
        windows.append(
            Window(
                train_start=pd.Timestamp(min_train_start),
                train_end=test_start - pd.Timedelta(hours=bar_hours * purge_bars),
                test_start=test_start,
                test_end=min(test_end, last_test),
            )
        )
        test_start = test_end
    return windows


def normalize_price_features(
    frame: pd.DataFrame, columns: list[str], window: int = 1_000
) -> pd.DataFrame:
    """Apply a causal rolling z-score to price features."""
    output = frame.copy()
    for column in columns:
        series = frame[column]
        mean = series.rolling(window, min_periods=window // 4).mean()
        std = series.rolling(window, min_periods=window // 4).std(ddof=0)
        output[column] = ((series - mean) / std.replace(0.0, np.nan)).clip(-8, 8)
    return output


def _wilson(successes: int, samples: int) -> tuple[float, float]:
    if samples == 0:
        return float("nan"), float("nan")
    probability, z = successes / samples, 1.96
    denominator = 1 + z * z / samples
    center = (probability + z * z / (2 * samples)) / denominator
    margin = z * (
        (
            probability * (1 - probability) / samples
            + z * z / (4 * samples * samples)
        )
        ** 0.5
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def block_bootstrap_ci(
    correct: np.ndarray,
    *,
    block: int = 36,
    draws: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Moving-block interval for overlapping 12-bar outcomes."""
    samples = len(correct)
    if samples < block * 3:
        return _wilson(int(correct.sum()), samples)
    rng = np.random.default_rng(seed)
    starts = np.arange(0, samples - block + 1)
    blocks = int(np.ceil(samples / block))
    means = np.empty(draws)
    for draw in range(draws):
        sampled_starts = rng.choice(starts, size=blocks)
        indices = (
            sampled_starts[:, None] + np.arange(block, dtype=int)[None, :]
        ).ravel()[:samples]
        means[draw] = correct[indices].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_ppf(probability: float) -> float:
    """Acklam's rational approximation to the standard normal quantile."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low, high = 0.02425, 1.0 - 0.02425
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if probability > high:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = probability - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
        ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


def deflated_sharpe_ratio(
    bar_returns: np.ndarray,
    *,
    trials: int,
    trial_sharpe_variance: float,
) -> dict[str, float | int | None]:
    """Bailey and Lopez de Prado's deflated Sharpe ratio.

    ``trials`` is how many configurations were screened and
    ``trial_sharpe_variance`` the variance of the per-bar Sharpe across them.
    Together they set the Sharpe a lucky random search would be expected to
    reach, which the observed Sharpe must clear to mean anything.
    """
    values = np.asarray(bar_returns, dtype=float)
    values = values[np.isfinite(values)]
    samples = len(values)
    if samples < 30 or trials < 2 or trial_sharpe_variance <= 0.0:
        return {"deflated_sharpe_ratio": None, "expected_max_sharpe": None,
                "observed_sharpe": None, "trials": int(trials), "samples": samples}
    deviation = values.std(ddof=1)
    if deviation <= 0.0:
        return {"deflated_sharpe_ratio": None, "expected_max_sharpe": None,
                "observed_sharpe": None, "trials": int(trials), "samples": samples}
    observed = float(values.mean() / deviation)
    centred = (values - values.mean()) / deviation
    skewness = float(np.mean(centred ** 3))
    kurtosis = float(np.mean(centred ** 4))
    threshold = math.sqrt(trial_sharpe_variance) * (
        (1.0 - EULER_MASCHERONI) * _normal_ppf(1.0 - 1.0 / trials)
        + EULER_MASCHERONI * _normal_ppf(1.0 - 1.0 / (trials * math.e))
    )
    denominator = 1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed ** 2
    if denominator <= 0.0:
        return {"deflated_sharpe_ratio": None, "expected_max_sharpe": threshold,
                "observed_sharpe": observed, "trials": int(trials), "samples": samples}
    statistic = (observed - threshold) * math.sqrt(samples - 1) / math.sqrt(denominator)
    return {
        "deflated_sharpe_ratio": float(_normal_cdf(statistic)),
        "expected_max_sharpe": float(threshold),
        "observed_sharpe": observed,
        "trials": int(trials),
        "samples": samples,
    }
