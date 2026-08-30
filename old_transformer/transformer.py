"""Train and evaluate the Alpha Barrier direction classifier.

The evaluation protocol deliberately separates model selection from the final
out-of-sample claim:

1. purged, expanding-window cross-validation is run on the development period;
2. the median best epoch is selected without looking at the holdout period;
3. one model is fit on the full development period and evaluated once on the
   chronologically last holdout period.

All model inputs are causal and stationary (returns, ratios, rolling statistics and
calendar cycles). The feature sequence includes the signal bar and labels only
inspect bars strictly after it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# Public constants are kept for the backtesting module and existing users.
ENABLED_SYMBOLS = ["XBTUSD"]
SEQUENCE_LENGTH = 64
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 40
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.25
N_SPLITS = 5
NHEAD = 4
DIM_FEEDFORWARD = 128
SEED = 42
DIRECTION_LOGISTIC_C = 1.0
DIRECTION_ENSEMBLE_TRANSFORMER_WEIGHT = 0.30
DIRECTION_FEATURE_PATTERNS = (
    "log_return",
    "close_to_ema",
    "rsi",
    "macd",
    "bollinger",
)

# A fixed horizon makes every label represent the same 48-hour opportunity on
# 240-minute data. The old ATR-ratio formula rounded to zero and therefore
# silently used its four-bar minimum for almost every sample.
LABEL_HORIZON_BARS = 12
HORIZON_MULTIPLIER = LABEL_HORIZON_BARS  # backwards-compatible import
PROFIT_TAKE_ATR_MULT = 1.5
STOP_LOSS_ATR_MULT = 1.5
HOLDOUT_FRACTION = 0.15

DOWN, SIDEWAYS, UP = 0, 1, 2
CLASS_NAMES = ["DOWN", "SIDEWAYS", "UP"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    nhead: int = NHEAD
    dropout: float = DROPOUT
    dim_feedforward: int = DIM_FEEDFORWARD
    n_classes: int = 3


@dataclass(frozen=True)
class DataConfig:
    sequence_length: int = SEQUENCE_LENGTH
    label_horizon_bars: int = LABEL_HORIZON_BARS
    profit_take_atr_mult: float = PROFIT_TAKE_ATR_MULT
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT
    holdout_fraction: float = HOLDOUT_FRACTION


def set_seed(seed: int) -> None:
    """Seed every RNG used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    relative_strength = _safe_divide(gain, loss)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def get_days_since_last_halving(timestamp: pd.Timestamp) -> int:
    """Return days since the latest known Bitcoin halving at ``timestamp``."""
    halving_dates = (
        pd.Timestamp("2009-01-03"),
        pd.Timestamp("2012-11-28"),
        pd.Timestamp("2016-07-09"),
        pd.Timestamp("2020-05-11"),
        pd.Timestamp("2024-04-19"),
    )
    previous = [date for date in halving_dates if date <= timestamp]
    return 0 if not previous else (timestamp - previous[-1]).days


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal, mostly stationary features to one OHLCVT frame.

    Raw price levels and cumulative indicators are intentionally not model
    features: their scale reveals the historical regime and generalises poorly.
    Every rolling calculation uses the current bar and older bars only.
    """
    result = df.copy()
    close = result["close"].astype(float)
    open_ = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    volume = result["volume"].astype(float)
    log_return_1 = np.log(close).diff()

    atr = _atr(result)
    result["feature_atr_pct"] = _safe_divide(atr, close)
    result["feature_range_pct"] = _safe_divide(high - low, close)
    result["feature_body_pct"] = _safe_divide(close - open_, open_)
    result["feature_upper_wick_pct"] = _safe_divide(high - np.maximum(open_, close), close)
    result["feature_lower_wick_pct"] = _safe_divide(np.minimum(open_, close) - low, close)
    result["feature_close_location"] = _safe_divide(close - low, high - low).clip(0.0, 1.0)

    for lag in (1, 2, 3, 6, 12, 24, 48):
        result[f"feature_log_return_{lag}"] = np.log(_safe_divide(close, close.shift(lag)))

    for window in (6, 12, 24, 48, 96):
        rolling_returns = log_return_1.rolling(window, min_periods=window)
        result[f"feature_realized_vol_{window}"] = rolling_returns.std(ddof=0)
        result[f"feature_return_z_{window}"] = _safe_divide(
            log_return_1 - rolling_returns.mean(), rolling_returns.std(ddof=0)
        )
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

    log_volume = np.log1p(volume)
    for window in (12, 24, 72):
        volume_mean = log_volume.rolling(window, min_periods=window).mean()
        volume_std = log_volume.rolling(window, min_periods=window).std(ddof=0)
        result[f"feature_volume_z_{window}"] = _safe_divide(log_volume - volume_mean, volume_std)
    result["feature_volume_change"] = log_volume.diff()
    # Trade count is available in Kraken OHLCVT but absent from the long
    # Bitstamp OHLCV archive.  Keep the venue-specific features when the field
    # exists; a model trained without them remains portable to either schema.
    if "trades" in result and result["trades"].notna().any():
        trades = result["trades"].astype(float)
        result["feature_trade_count_change"] = np.log1p(trades).diff()
        result["feature_average_trade_size"] = _safe_divide(volume, trades).pipe(np.log1p).diff()

    for window in (12, 36, 84, 168):
        resistance = high.rolling(window, min_periods=window).max()
        support = low.rolling(window, min_periods=window).min()
        result[f"feature_dist_to_resistance_{window}"] = _safe_divide(resistance - close, close)
        result[f"feature_dist_to_support_{window}"] = _safe_divide(close - support, close)
        result[f"feature_range_position_{window}"] = _safe_divide(close - support, resistance - support)

    return result


def add_support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper; support/resistance is part of feature engineering."""
    return add_technical_features(df)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    index = result.index
    cycles = (
        (index.hour.to_numpy(), 24, "hour"),
        (index.dayofweek.to_numpy(), 7, "day_of_week"),
        (index.month.to_numpy() - 1, 12, "month"),
    )
    for values, period, name in cycles:
        angle = 2 * np.pi * values / period
        result[f"feature_{name}_sin"] = np.sin(angle)
        result[f"feature_{name}_cos"] = np.cos(angle)

    days_since_halving = np.asarray([get_days_since_last_halving(ts) for ts in index])
    halving_angle = 2 * np.pi * (days_since_halving % 1461) / 1461
    result["feature_halving_cycle_sin"] = np.sin(halving_angle)
    result["feature_halving_cycle_cos"] = np.cos(halving_angle)
    return result


def get_triple_barrier_labels(
    prices: pd.DataFrame,
    atr_series: pd.Series,
    horizon_mult: float = LABEL_HORIZON_BARS,
    pt_mult: float = PROFIT_TAKE_ATR_MULT,
    sl_mult: float = STOP_LOSS_ATR_MULT,
    *,
    horizon_bars: int | None = None,
) -> pd.Series:
    """Return forward-looking triple-barrier labels.

    ``horizon_mult`` is retained for API compatibility but is interpreted as a
    number of bars. Earlier code multiplied ATR/price by this value, which
    rounded to zero and made the horizon effectively constant at four bars.

    A sample is excluded when one bar touches both barriers because OHLC data
    cannot reveal which level was hit first. The final ``horizon_bars``
    observations also remain missing rather than receiving artificially
    shortened horizons.
    """
    horizon = int(round(horizon_mult if horizon_bars is None else horizon_bars))
    if horizon < 1:
        raise ValueError("horizon_bars must be at least 1")
    if pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("barrier multipliers must be positive")

    labels = pd.Series(np.nan, index=prices.index, dtype=float)
    close = prices["close"].to_numpy(dtype=float)
    high = prices["high"].to_numpy(dtype=float)
    low = prices["low"].to_numpy(dtype=float)
    atr = atr_series.to_numpy(dtype=float)

    for entry_idx in range(0, len(prices) - horizon):
        if not np.isfinite(close[entry_idx]) or not np.isfinite(atr[entry_idx]):
            continue
        upper = close[entry_idx] + pt_mult * atr[entry_idx]
        lower = close[entry_idx] - sl_mult * atr[entry_idx]
        outcome = 0
        for path_idx in range(entry_idx + 1, entry_idx + horizon + 1):
            hit_upper = high[path_idx] >= upper
            hit_lower = low[path_idx] <= lower
            if hit_upper and hit_lower:
                outcome = np.nan  # intrabar ordering is unknowable
                break
            if hit_upper:
                outcome = 1
                break
            if hit_lower:
                outcome = -1
                break
        labels.iloc[entry_idx] = outcome
    return labels


def read_ohlcvt(path: str | os.PathLike[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] not in (6, 7):
        raise ValueError(f"{path} has {frame.shape[1]} columns; expected OHLCV (6) or OHLCVT (7)")
    frame.columns = ["time", "open", "high", "low", "close", "volume", "trades"][: frame.shape[1]]
    numeric_time = pd.to_numeric(frame.pop("time"), errors="coerce")
    time_unit = "ms" if numeric_time.dropna().median() > 10_000_000_000 else "s"
    frame.index = pd.to_datetime(numeric_time, unit=time_unit, utc=True).dt.tz_localize(None)
    frame.index.name = "time"
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model features in deterministic artifact order."""
    return sorted(column for column in df.columns if "feature_" in column)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the compact momentum feature set selected on development folds."""
    return [
        column
        for column in get_feature_columns(df)
        if any(pattern in column for pattern in DIRECTION_FEATURE_PATTERNS)
    ]


def load_and_engineer(
    path_dir: str | os.PathLike[str],
    *,
    include_target: bool = True,
    data_config: DataConfig | None = None,
) -> pd.DataFrame:
    """Load configured symbols and build the causal model frame."""
    config = data_config or DataConfig()
    frames: list[pd.DataFrame] = []
    raw_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in ENABLED_SYMBOLS:
        path = Path(path_dir) / f"{symbol}_240.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Training data not found: {path}. Download Kraken {symbol} 240-minute OHLCVT data first."
            )
        raw = read_ohlcvt(path)
        raw_by_symbol[symbol] = raw
        frames.append(add_technical_features(raw).add_prefix(f"{symbol}_"))

    combined = pd.concat(frames, axis=1, join="inner").sort_index()
    combined = add_calendar_features(combined)

    if include_target:
        primary = raw_by_symbol[ENABLED_SYMBOLS[0]].reindex(combined.index)
        labels = get_triple_barrier_labels(
            primary[["high", "low", "close"]],
            _atr(primary),
            horizon_bars=config.label_horizon_bars,
            pt_mult=config.profit_take_atr_mult,
            sl_mult=config.stop_loss_atr_mult,
        )
        combined["target"] = labels.map({-1: DOWN, 0: SIDEWAYS, 1: UP})

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    required = get_feature_columns(combined)
    # Keep rows with unavailable labels as sequence context. Dataset target
    # indices filter them out, so ambiguous samples and the final horizon are
    # never trained or scored but their observed features are not erased.
    return combined.dropna(subset=required).copy()


class TimeSeriesDataset(Dataset):
    """Sequences ending on (and including) each requested signal bar."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        target_col: str,
        seq_len: int,
        target_indices: Iterable[int] | None = None,
    ) -> None:
        self.X = df.loc[:, features].to_numpy(dtype=np.float32)
        self.y = pd.to_numeric(df.loc[:, target_col], errors="coerce").to_numpy(dtype=float)
        self.seq_len = seq_len
        candidates = range(seq_len - 1, len(df)) if target_indices is None else target_indices
        self.indices = np.asarray(
            [
                idx
                for idx in candidates
                if seq_len - 1 <= idx < len(df) and np.isfinite(self.y[idx])
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_idx = int(self.indices[idx])
        start_idx = target_idx - self.seq_len + 1
        x = self.X[start_idx : target_idx + 1]
        return torch.from_numpy(x), torch.tensor(int(self.y[target_idx]), dtype=torch.long)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first tensors."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        nhead: int,
        dropout: float,
        n_classes: int = 3,
        dim_feedforward: int = DIM_FEEDFORWARD,
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.pos_encoder(self.embedding(x))
        encoded = self.transformer(embedded)
        return self.fc(self.final_norm(encoded[:, -1]))


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=3).astype(float)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_loop(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total_loss += criterion(model(xb), yb).item() * len(yb)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for xb, yb in loader:
        logits = model(xb.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(yb.numpy())
    probability_array = np.concatenate(probabilities)
    label_array = np.concatenate(labels)
    return label_array, probability_array.argmax(axis=1), probability_array


def _wilson_interval(successes: int, samples: int) -> tuple[float | None, float | None]:
    if samples == 0:
        return None, None
    proportion = successes / samples
    z = 1.96
    denominator = 1.0 + z**2 / samples
    center = (proportion + z**2 / (2 * samples)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / samples + z**2 / (4 * samples**2))
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def calculate_metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Calculate explicit metrics so a headline number cannot hide coverage."""
    directional_truth = y_true != SIDEWAYS
    actionable = y_pred != SIDEWAYS
    majority_accuracy = np.bincount(y_true, minlength=3).max() / len(y_true)
    directional_samples = int(directional_truth.sum())
    actionable_samples = int(actionable.sum())
    directional_accuracy = (
        float(np.mean(y_pred[directional_truth] == y_true[directional_truth]))
        if directional_truth.any()
        else None
    )
    signal_accuracy = (
        float(np.mean(y_pred[actionable] == y_true[actionable])) if actionable.any() else None
    )
    directional_baseline = None
    directional_ci_low, directional_ci_high = _wilson_interval(
        int(np.sum(y_pred[directional_truth] == y_true[directional_truth])), directional_samples
    )
    if directional_samples:
        directional_labels = y_true[directional_truth]
        directional_baseline = float(
            max(np.sum(directional_labels == DOWN), np.sum(directional_labels == UP))
            / directional_samples
        )
    metrics: dict[str, float | int | None] = {
        "samples": int(len(y_true)),
        "directional_samples": directional_samples,
        "actionable_samples": actionable_samples,
        "three_class_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy,
        "directional_accuracy_ci95_low": directional_ci_low,
        "directional_accuracy_ci95_high": directional_ci_high,
        "signal_accuracy": signal_accuracy,
        "signal_coverage": float(actionable.mean()),
        "majority_baseline_accuracy": float(majority_accuracy),
        "directional_majority_baseline_accuracy": directional_baseline,
    }
    if probabilities is not None:
        if probabilities.shape != (len(y_true), 3):
            raise ValueError(
                f"Expected probability shape {(len(y_true), 3)}, got {probabilities.shape}"
            )
        # Conventional direction-only score: for outcomes that resolved DOWN
        # or UP, compare those two probabilities directly.  SIDEWAYS outcomes
        # are outside this metric and remain included in three-class accuracy.
        forced_direction = np.where(probabilities[:, UP] >= probabilities[:, DOWN], UP, DOWN)
        forced_successes = int(
            np.sum(forced_direction[directional_truth] == y_true[directional_truth])
        )
        forced_low, forced_high = _wilson_interval(forced_successes, directional_samples)
        metrics.update(
            {
                "forced_directional_accuracy": (
                    forced_successes / directional_samples if directional_samples else None
                ),
                "forced_directional_accuracy_ci95_low": forced_low,
                "forced_directional_accuracy_ci95_high": forced_high,
            }
        )
    return metrics


def _direction_probability_metrics(
    y_true: np.ndarray, probability_up: np.ndarray, prefix: str
) -> dict[str, float | int | None]:
    if probability_up.shape != (len(y_true),):
        raise ValueError(f"Expected {(len(y_true),)} direction probabilities, got {probability_up.shape}")
    directional = y_true != SIDEWAYS
    samples = int(directional.sum())
    predicted_up = probability_up >= 0.5
    correct = int(np.sum(predicted_up[directional] == (y_true[directional] == UP)))
    low, high = _wilson_interval(correct, samples)
    return {
        f"{prefix}_accuracy": correct / samples if samples else None,
        f"{prefix}_accuracy_ci95_low": low,
        f"{prefix}_accuracy_ci95_high": high,
        f"{prefix}_samples": samples,
    }


def _fit_direction_head(
    scaled: pd.DataFrame,
    feature_cols: Sequence[str],
    target_indices: np.ndarray,
    seed: int,
) -> LogisticRegression:
    targets = scaled.iloc[target_indices]["target"].to_numpy()
    directional = targets != SIDEWAYS
    model = LogisticRegression(
        C=DIRECTION_LOGISTIC_C,
        class_weight="balanced",
        max_iter=1_000,
        random_state=seed,
    )
    model.fit(
        scaled.iloc[target_indices[directional]][list(feature_cols)],
        (targets[directional] == UP).astype(int),
    )
    return model


def _combine_direction_probabilities(
    transformer_probabilities: np.ndarray, direction_probability_up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    transformer_probability_up = transformer_probabilities[:, UP] / (
        transformer_probabilities[:, DOWN] + transformer_probabilities[:, UP]
    )
    hybrid_probability_up = (
        DIRECTION_ENSEMBLE_TRANSFORMER_WEIGHT * transformer_probability_up
        + (1.0 - DIRECTION_ENSEMBLE_TRANSFORMER_WEIGHT) * direction_probability_up
    )
    return transformer_probability_up, hybrid_probability_up


def calculate_metrics(model: nn.Module, loader: DataLoader) -> tuple[float, str]:
    """Compatibility API returning exact accuracy and a class report."""
    y_true, y_pred, _ = predict(model, loader)
    return accuracy_score(y_true, y_pred), classification_report(
        y_true, y_pred, labels=[DOWN, SIDEWAYS, UP], target_names=CLASS_NAMES, zero_division=0
    )


def _scaled_frame(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    fit_indices: Sequence[int],
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    fit_positions = np.asarray(fit_indices)
    # Early feature rows are part of the first training sequences, even though
    # they are not target positions themselves.
    scaler.fit(df.iloc[: int(fit_positions.max()) + 1][list(feature_cols)])
    scaled = df.copy()
    scaled.loc[:, feature_cols] = scaler.transform(df.loc[:, feature_cols]).astype(np.float32)
    return scaled, scaler


def _new_model(n_features: int, config: ModelConfig) -> TransformerClassifier:
    return TransformerClassifier(n_features=n_features, **asdict(config)).to(device)


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_weights: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    patience: int = 6,
) -> tuple[nn.Module, int, float]:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    wait = 0
    for epoch in range(1, epochs + 1):
        training_loss = train_loop(model, train_loader, criterion, optimizer)
        validation_loss = evaluate_loss(model, validation_loader, criterion)
        scheduler.step(validation_loss)
        print(
            f"  epoch {epoch:02d} | train {training_loss:.5f} | validation {validation_loss:.5f}"
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def run_purged_cross_validation(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    development_end: int,
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_splits: int,
    seed: int,
) -> tuple[list[dict[str, object]], int]:
    """Tune only within the development period using a label-horizon purge."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=data_config.label_horizon_bars)
    development_positions = np.arange(development_end)
    fold_results: list[dict[str, object]] = []

    for fold, (train_relative, validation_relative) in enumerate(
        splitter.split(development_positions), start=1
    ):
        train_indices = development_positions[train_relative]
        validation_indices = development_positions[validation_relative]
        train_indices = train_indices[train_indices >= data_config.sequence_length - 1]
        if len(train_indices) == 0:
            raise ValueError("Not enough observations for the requested sequence length and folds")

        scaled, _ = _scaled_frame(df, feature_cols, train_indices)
        train_dataset = TimeSeriesDataset(
            scaled, feature_cols, "target", data_config.sequence_length, train_indices
        )
        validation_dataset = TimeSeriesDataset(
            scaled, feature_cols, "target", data_config.sequence_length, validation_indices
        )
        train_loader = _make_loader(train_dataset, batch_size, shuffle=True, seed=seed + fold)
        validation_loader = _make_loader(
            validation_dataset, batch_size, shuffle=False, seed=seed + fold
        )

        set_seed(seed + fold)
        model = _new_model(len(feature_cols), model_config)
        weights = _class_weights(df.iloc[train_dataset.indices]["target"].to_numpy())
        print(
            f"\nFold {fold}/{n_splits}: train={len(train_dataset):,}, "
            f"validation={len(validation_dataset):,}"
        )
        model, best_epoch, best_loss = fit_with_early_stopping(
            model,
            train_loader,
            validation_loader,
            weights,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        y_true, y_pred, probabilities = predict(model, validation_loader)
        metrics = calculate_metric_dict(y_true, y_pred, probabilities)
        direction_head = _fit_direction_head(
            scaled, feature_cols, train_dataset.indices, seed + fold
        )
        direction_probability_up = direction_head.predict_proba(
            scaled.iloc[validation_dataset.indices][list(feature_cols)]
        )[:, 1]
        _, hybrid_probability_up = _combine_direction_probabilities(
            probabilities, direction_probability_up
        )
        metrics.update(
            _direction_probability_metrics(
                y_true, direction_probability_up, "direction_head"
            )
        )
        metrics.update(
            _direction_probability_metrics(
                y_true, hybrid_probability_up, "hybrid_directional"
            )
        )
        directional_text = (
            "n/a"
            if metrics["directional_accuracy"] is None
            else f"{metrics['directional_accuracy']:.2%}"
        )
        print(
            f"  fold accuracy={metrics['three_class_accuracy']:.2%}, "
            f"directional={directional_text}, "
            f"baseline={metrics['majority_baseline_accuracy']:.2%}"
        )
        fold_results.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "validation_start": str(df.index[validation_dataset.indices[0]]),
                "validation_end": str(df.index[validation_dataset.indices[-1]]),
                **metrics,
            }
        )

    selected_epochs = max(1, int(round(np.median([row["best_epoch"] for row in fold_results]))))
    return fold_results, selected_epochs


def fit_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    for epoch in range(1, epochs + 1):
        loss = train_loop(model, train_loader, criterion, optimizer)
        scheduler.step()
        print(f"  final epoch {epoch:02d}/{epochs:02d} | train {loss:.5f}")
    return model


def _read_dataset_manifest(data_dir: Path) -> dict[str, object]:
    manifest_path = data_dir / "dataset.json"
    if not manifest_path.exists():
        return {"data_dir": str(data_dir.resolve()), "manifest": None}
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def train_and_evaluate(
    data_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    epochs: int = EPOCHS,
    n_splits: int = N_SPLITS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LR,
    seed: int = SEED,
    data_config: DataConfig | None = None,
    model_config: ModelConfig | None = None,
    evaluation_data_dir: str | os.PathLike[str] | None = None,
    development_end_time: str | None = None,
) -> dict[str, object]:
    data_config = data_config or DataConfig()
    model_config = model_config or ModelConfig()
    set_seed(seed)
    print(f"Using device: {device}")
    df = load_and_engineer(data_dir, data_config=data_config)
    feature_cols = get_model_feature_columns(df)

    if evaluation_data_dir is not None:
        if development_end_time is None:
            raise ValueError("--development-end is required with --evaluation-data-dir")
        cutoff = pd.Timestamp(development_end_time)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)
        cutoff_position = int(df.index.searchsorted(cutoff, side="right"))
        development_end = cutoff_position - data_config.label_horizon_bars
        holdout_start = None
    else:
        holdout_start = int(len(df) * (1.0 - data_config.holdout_fraction))
        development_end = holdout_start - data_config.label_horizon_bars
    minimum = (n_splits + 1) * data_config.sequence_length
    if development_end < minimum:
        raise ValueError(
            f"Only {len(df):,} usable rows; need at least {minimum:,} for {n_splits} folds"
        )
    print(
        f"Rows={len(df):,}, features={len(feature_cols)}, development_end={development_end:,}, "
        + (
            f"external_evaluation={evaluation_data_dir}"
            if evaluation_data_dir is not None
            else f"untouched_holdout={len(df) - int(holdout_start):,}"
        )
    )
    print("Class distribution:")
    print(
        df.iloc[:development_end]["target"]
        .value_counts(normalize=True)
        .sort_index()
        .rename(index=dict(enumerate(CLASS_NAMES)))
    )

    fold_results, selected_epochs = run_purged_cross_validation(
        df,
        feature_cols,
        development_end,
        data_config=data_config,
        model_config=model_config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_splits=n_splits,
        seed=seed,
    )
    print(f"\nSelected final training duration: {selected_epochs} epochs (CV median)")

    final_train_indices = np.arange(data_config.sequence_length - 1, development_end)
    scaled, scaler = _scaled_frame(df, feature_cols, final_train_indices)
    train_dataset = TimeSeriesDataset(
        scaled, feature_cols, "target", data_config.sequence_length, final_train_indices
    )
    train_loader = _make_loader(train_dataset, batch_size, shuffle=True, seed=seed)

    if evaluation_data_dir is not None:
        holdout_frame = load_and_engineer(evaluation_data_dir, data_config=data_config)
        missing_features = sorted(set(feature_cols) - set(get_feature_columns(holdout_frame)))
        if missing_features:
            raise ValueError(f"Evaluation data is missing trained features: {missing_features}")
        holdout_scaled = holdout_frame.copy()
        holdout_scaled.loc[:, feature_cols] = scaler.transform(
            holdout_frame.loc[:, feature_cols]
        ).astype(np.float32)
        holdout_indices = np.arange(data_config.sequence_length - 1, len(holdout_frame))
        holdout_dataset = TimeSeriesDataset(
            holdout_scaled,
            feature_cols,
            "target",
            data_config.sequence_length,
            holdout_indices,
        )
    else:
        holdout_frame = df
        holdout_indices = np.arange(int(holdout_start), len(df))
        holdout_dataset = TimeSeriesDataset(
            scaled, feature_cols, "target", data_config.sequence_length, holdout_indices
        )
    holdout_loader = _make_loader(holdout_dataset, batch_size, shuffle=False, seed=seed)

    set_seed(seed)
    final_model = _new_model(len(feature_cols), model_config)
    weights = _class_weights(df.iloc[train_dataset.indices]["target"].to_numpy())
    final_model = fit_fixed_epochs(
        final_model,
        train_loader,
        weights,
        epochs=selected_epochs,
        learning_rate=learning_rate,
    )
    direction_head = _fit_direction_head(scaled, feature_cols, train_dataset.indices, seed)

    y_true, y_pred, probabilities = predict(final_model, holdout_loader)
    holdout_metrics = calculate_metric_dict(y_true, y_pred, probabilities)
    direction_probability_up = direction_head.predict_proba(
        holdout_scaled.iloc[holdout_dataset.indices][feature_cols]
        if evaluation_data_dir is not None
        else scaled.iloc[holdout_dataset.indices][feature_cols]
    )[:, 1]
    transformer_probability_up, hybrid_probability_up = _combine_direction_probabilities(
        probabilities, direction_probability_up
    )
    holdout_metrics.update(
        _direction_probability_metrics(y_true, direction_probability_up, "direction_head")
    )
    holdout_metrics.update(
        _direction_probability_metrics(y_true, hybrid_probability_up, "hybrid_directional")
    )
    holdout_metrics.update(
        {
            "start": str(holdout_frame.index[holdout_dataset.indices[0]]),
            "end": str(holdout_frame.index[holdout_dataset.indices[-1]]),
            "class_report": classification_report(
                y_true,
                y_pred,
                labels=[DOWN, SIDEWAYS, UP],
                target_names=CLASS_NAMES,
                zero_division=0,
                output_dict=True,
            ),
        }
    )
    cv_summary = {
        metric: (
            float(np.mean([float(fold[metric]) for fold in fold_results if fold[metric] is not None]))
            if any(fold[metric] is not None for fold in fold_results)
            else None
        )
        for metric in (
            "three_class_accuracy",
            "balanced_accuracy",
            "directional_accuracy",
            "forced_directional_accuracy",
            "direction_head_accuracy",
            "hybrid_directional_accuracy",
            "signal_accuracy",
            "signal_coverage",
            "majority_baseline_accuracy",
            "directional_majority_baseline_accuracy",
        )
    }
    results: dict[str, object] = {
        "protocol": (
            "purged expanding-window CV through the development cutoff plus future "
            "cross-venue external holdout"
            if evaluation_data_dir is not None
            else "purged expanding-window CV plus untouched chronological holdout"
        ),
        "seed": seed,
        "data_config": asdict(data_config),
        "model_config": asdict(model_config),
        "feature_count": len(feature_cols),
        "direction_model": {
            "type": "logistic_regression",
            "C": DIRECTION_LOGISTIC_C,
            "transformer_ensemble_weight": DIRECTION_ENSEMBLE_TRANSFORMER_WEIGHT,
            "feature_patterns": list(DIRECTION_FEATURE_PATTERNS),
        },
        "selected_epochs": selected_epochs,
        "development_end_time": development_end_time,
        "training_data": _read_dataset_manifest(Path(data_dir)),
        "evaluation_data": (
            _read_dataset_manifest(Path(evaluation_data_dir))
            if evaluation_data_dir is not None
            else _read_dataset_manifest(Path(data_dir))
        ),
        "cross_validation": {"folds": fold_results, "mean": cv_summary},
        "holdout": holdout_metrics,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "feature_cols": feature_cols,
            "model_config": asdict(model_config),
            "data_config": asdict(data_config),
            "selected_epochs": selected_epochs,
            "holdout_start": str(holdout_frame.index[holdout_dataset.indices[0]]),
            "training_data": results["training_data"],
            "evaluation_data": results["evaluation_data"],
            "direction_feature_cols": feature_cols,
            "direction_ensemble_transformer_weight": DIRECTION_ENSEMBLE_TRANSFORMER_WEIGHT,
        },
        output_path / "final_transformer_model.pt",
    )
    joblib.dump(scaler, output_path / "final_scaler.pkl")
    joblib.dump(direction_head, output_path / "direction_classifier.pkl")
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    np.savez_compressed(
        output_path / "holdout_predictions.npz",
        timestamp=np.asarray(
            holdout_frame.index[holdout_dataset.indices].astype(str), dtype=str
        ),
        y_true=y_true,
        y_pred=y_pred,
        probability=probabilities,
        transformer_probability_up=transformer_probability_up,
        direction_probability_up=direction_probability_up,
        hybrid_probability_up=hybrid_probability_up,
        hybrid_direction_prediction=np.where(hybrid_probability_up >= 0.5, UP, DOWN),
    )

    print("\nUntouched out-of-sample holdout")
    print(json.dumps({key: value for key, value in holdout_metrics.items() if key != "class_report"}, indent=2))
    print(f"Artifacts written to {output_path.resolve()}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./training_data")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--evaluation-data-dir",
        help="Optional distinct future dataset used only for final evaluation",
    )
    parser.add_argument(
        "--development-end",
        help="Last timestamp allowed in development data; required for external evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_and_evaluate(
        arguments.data_dir,
        arguments.output_dir,
        epochs=arguments.epochs,
        n_splits=arguments.splits,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        evaluation_data_dir=arguments.evaluation_data_dir,
        development_end_time=arguments.development_end,
    )
