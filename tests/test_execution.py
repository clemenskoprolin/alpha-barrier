import numpy as np
import pandas as pd
import pytest

from alphabarrier.execution import build_strategy_positions, continuous_backtest


def test_strategy_holds_through_weak_calls_and_exits_on_confident_down_call():
    index = pd.date_range("2020-01-01", periods=6, freq="4h")
    frame = pd.DataFrame({"raw_close": np.arange(100.0, 106.0)}, index=index)
    signals = pd.DataFrame(
        {
            "side": [1, -1, -1, -1],
            "confidence": [1.0, 0.1, 0.2, 1.0],
        },
        index=index[1:5],
    )
    positions = build_strategy_positions(
        signals,
        frame,
        confidence_floor=0.5,
        trend_ema_span=2,
    )
    np.testing.assert_array_equal(positions.to_numpy(), [1.0, 1.0, 1.0, 0.0])


def test_strategy_is_long_only_when_above_trend_gate():
    index = pd.date_range("2020-01-01", periods=6, freq="4h")
    frame = pd.DataFrame(
        {"raw_close": [100.0, 101.0, 102.0, 101.0, 103.0, 104.0]}, index=index
    )
    signals = pd.DataFrame(
        {"side": [1, 1, -1, 1], "confidence": [1.0, 0.1, 1.0, 1.0]},
        index=index[1:5],
    )
    positions = build_strategy_positions(
        signals,
        frame,
        confidence_floor=0.5,
        trend_ema_span=2,
    )
    assert set(positions.unique()) <= {0.0, 1.0}
    assert positions.iloc[0] == 1.0
    assert positions.iloc[1] == 1.0
    assert positions.iloc[2] == 0.0


def test_continuous_backtest_charges_entry_and_final_exit():
    index = pd.date_range("2020-01-01", periods=5, freq="4h")
    frame = pd.DataFrame({"raw_close": np.full(5, 100.0)}, index=index)
    positions = pd.Series(np.ones(4), index=index[:4])
    result = continuous_backtest(positions, frame, commission_per_side=0.001)
    assert result["fee_sides"] == 2.0
    assert result["total_return"] == pytest.approx(0.999**2 - 1.0)


def test_continuous_backtest_requires_a_close_after_last_position():
    index = pd.date_range("2020-01-01", periods=3, freq="4h")
    frame = pd.DataFrame({"raw_close": [100.0, 101.0, 102.0]}, index=index)
    positions = pd.Series(np.ones(3), index=index)
    with pytest.raises(ValueError, match="one valid close"):
        continuous_backtest(positions, frame)
