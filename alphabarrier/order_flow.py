"""Causal Binance taker-flow features used by the current strategy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PREFIX = "flow_"
ORDER_FLOW_COLUMNS = (
    "flow_spot_imbalance",
    "flow_spot_imbalance_mean_6",
    "flow_spot_imbalance_mean_24",
    "flow_spot_imbalance_mean_72",
    "flow_spot_imbalance_z_240",
    "flow_spot_trade_size_z",
    "flow_spot_trades_z",
    "flow_eth_imbalance",
    "flow_perp_imbalance",
    "flow_perp_imbalance_mean_6",
    "flow_perp_imbalance_mean_24",
    "flow_perp_imbalance_mean_72",
)


def _z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 4).mean()
    std = series.rolling(window, min_periods=window // 4).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-8, 8)


def _imbalance(frame: pd.DataFrame) -> pd.Series:
    return (2.0 * frame["taker_buy_base"] - frame["volume"]) / frame[
        "volume"
    ].replace(0.0, np.nan)


def build_order_flow_features(
    index: pd.DatetimeIndex, market_dir: str | Path
) -> pd.DataFrame:
    """Build the frozen 12-feature order-flow block on ``index``."""
    market_dir = Path(market_dir)
    spot = pd.read_pickle(market_dir / "spot_btcusdt_4h.pkl")
    eth = pd.read_pickle(market_dir / "spot_ethusdt_4h.pkl")
    perp = pd.read_pickle(market_dir / "perp_btcusdt_4h.pkl")
    output = pd.DataFrame(index=index)

    spot_imbalance = _imbalance(spot)
    output["flow_spot_imbalance"] = spot_imbalance.reindex(index)
    for window in (6, 24, 72):
        output[f"flow_spot_imbalance_mean_{window}"] = spot_imbalance.rolling(
            window, min_periods=window // 2
        ).mean().reindex(index)
    output["flow_spot_imbalance_z_240"] = _z(spot_imbalance, 240).reindex(index)
    trade_size = spot["volume"] / spot["trades"].replace(0, np.nan)
    output["flow_spot_trade_size_z"] = _z(np.log(trade_size), 240).reindex(index)
    output["flow_spot_trades_z"] = _z(np.log1p(spot["trades"]), 240).reindex(index)

    output["flow_eth_imbalance"] = _imbalance(eth).reindex(index)

    perp_imbalance = _imbalance(perp)
    output["flow_perp_imbalance"] = perp_imbalance.reindex(index)
    for window in (6, 24, 72):
        output[f"flow_perp_imbalance_mean_{window}"] = perp_imbalance.rolling(
            window, min_periods=window // 2
        ).mean().reindex(index)

    return output.loc[:, ORDER_FLOW_COLUMNS].replace([np.inf, -np.inf], np.nan)
