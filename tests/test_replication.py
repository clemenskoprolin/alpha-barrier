import numpy as np
import pandas as pd

from alphabarrier.execution import trend_gate
from alphabarrier.replication import (
    PERIODS,
    REPLICATION_START,
    off_trend_participation,
    period_table,
    replication_features,
    replication_windows,
    short_sleeve,
    sleeve_table,
    yearly_sharpe_table,
)
from alphabarrier.strategy import LABEL_EMBARGO


def _frame(years: int = 4) -> pd.DataFrame:
    periods = years * 366 * 6
    index = pd.date_range("2014-01-01", periods=periods, freq="4h")
    drift = np.linspace(0.0, 1.2, periods)
    wobble = 0.25 * np.sin(np.arange(periods) / 400.0)
    return pd.DataFrame({"raw_close": 100.0 * np.exp(drift + wobble)}, index=index)


def _signals(frame: pd.DataFrame) -> pd.DataFrame:
    index = frame.index[2_500:-1]
    rng = np.random.default_rng(0)
    probability = rng.uniform(0.35, 0.65, len(index))
    return pd.DataFrame(
        {
            "side": np.where(probability >= 0.5, 1, -1),
            "confidence": np.abs(probability - 0.5),
            "probability": probability,
            "threshold": 0.5,
            "logit_scale": 0.3,
            "signal_span": 5.0,
        },
        index=index,
    )


def test_replication_drops_the_feature_blocks_that_do_not_reach_back():
    features = [
        "XBTUSD_feature_rsi_14",
        "flow_spot_imbalance",
        "flow_perp_imbalance",
        "mature_outcome_ewm_9",
    ]
    assert replication_features(features) == [
        "XBTUSD_feature_rsi_14",
        "mature_outcome_ewm_9",
    ]


def test_replication_windows_start_in_2014_and_keep_the_purge():
    frame = _frame()
    base = replication_windows(frame, 0)
    shifted = replication_windows(frame, 2)
    assert base[0].test_start == REPLICATION_START
    assert shifted[0].test_start == REPLICATION_START + pd.DateOffset(months=2)
    assert all(
        window.train_end == window.test_start - LABEL_EMBARGO for window in base
    )
    assert base[-1].test_end <= frame.index[-1]


def test_period_table_prices_every_rule_over_every_period():
    frame = _frame()
    rows = period_table([_signals(frame)], frame)
    assert [row["period"] for row in rows] == [
        "pre_development",
        "development",
        "later",
        "all",
    ]
    priced = rows[0]
    assert {"current", "binary_gate", "trend_only", "buy_and_hold"} <= set(priced)
    assert priced["buy_and_hold"]["mean_weight"] == 1.0
    assert set(priced["current"]) == {
        "total_return",
        "sharpe",
        "max_drawdown",
        "mean_weight",
    }
    # Periods with no overlapping signal are omitted rather than reported empty.
    assert "current" not in rows[2]


def test_yearly_table_counts_years_instead_of_pooling_them():
    frame = _frame()
    table = yearly_sharpe_table([_signals(frame)], frame)
    years = [row["year"] for row in table["years"]]
    assert years == sorted(years)
    paired = {row["comparison"]: row for row in table["paired"]}
    gap = paired["current minus binary_gate"]
    assert gap["years"] == len(years)
    assert 0 <= gap["years_won"] <= gap["years"]
    assert gap["pre_development_years"] == sum(year < 2021 for year in years)


def test_sleeves_are_priced_against_the_current_rule_in_every_period():
    frame = _frame()
    table = sleeve_table([_signals(frame)], frame)
    variants = table["variants"]
    assert set(variants) == {
        "current",
        "short_sleeve_0.25x",
        "short_sleeve_0.5x",
        "short_sleeve_1x",
        "off_trend_participation_0.15x",
        "off_trend_participation_0.5x",
    }
    assert set(variants["current"]) == {name for name, _, _ in PERIODS}


def test_short_sleeve_goes_short_off_trend_while_participation_stays_long():
    frame = _frame()
    signals = _signals(frame)
    sleeve = short_sleeve(signals, frame, 1.0)
    partial = off_trend_participation(signals, frame, 0.5)
    gate = trend_gate(frame).reindex(signals.index).fillna(0.0)
    flat = gate == 0.0
    assert flat.any()
    assert (sleeve.loc[flat] <= 0.0).all()
    assert sleeve.min() < 0.0
    # Participation only ever adds long exposure, never a short one.
    assert (partial >= 0.0).all()
    assert (partial.loc[flat] > 0.0).any()
