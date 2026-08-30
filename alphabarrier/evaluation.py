"""Purged walk-forward evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    step_months: int = 6,
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
