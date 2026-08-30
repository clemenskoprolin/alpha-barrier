import numpy as np
import pandas as pd

from alphabarrier.features import (
    add_momentum_features,
    get_triple_barrier_labels,
    read_ohlcv,
)


def _barrier_case(second_high: float, second_low: float) -> pd.Series:
    prices = pd.DataFrame(
        {
            "high": [100.0, second_high, 100.5],
            "low": [100.0, second_low, 99.5],
            "close": [100.0, 100.0, 100.0],
        }
    )
    return get_triple_barrier_labels(
        prices,
        pd.Series([1.0, 1.0, 1.0]),
        pt_mult=1.0,
        sl_mult=1.0,
        horizon_bars=2,
    )


def test_barrier_uses_first_touch_and_excludes_unknown_intrabar_order():
    assert _barrier_case(101.0, 99.5).iloc[0] == 1
    assert _barrier_case(100.5, 99.0).iloc[0] == -1
    assert np.isnan(_barrier_case(101.0, 99.0).iloc[0])


def test_barrier_does_not_shorten_final_horizons():
    assert _barrier_case(101.0, 99.5).iloc[-2:].isna().all()


def test_feature_engineering_is_prefix_invariant():
    size = 220
    index = pd.date_range("2020-01-01", periods=size, freq="4h")
    base = np.linspace(100.0, 150.0, size)
    frame = pd.DataFrame(
        {
            "open": base - 0.2,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.linspace(10.0, 20.0, size),
            "trades": np.arange(size) + 20,
        },
        index=index,
    )
    original = add_momentum_features(frame)
    changed = frame.copy()
    changed.loc[index[-1], ["open", "high", "low", "close", "volume"]] *= 10
    recomputed = add_momentum_features(changed)
    feature_columns = [column for column in original if column.startswith("feature_")]
    pd.testing.assert_series_equal(
        original.loc[index[-2], feature_columns],
        recomputed.loc[index[-2], feature_columns],
        check_names=False,
    )


def test_six_column_ohlcv_is_supported(tmp_path):
    path = tmp_path / "BTCUSD_240.csv"
    path.write_text("100,1,2,0.5,1.5,10\n200,1.5,2.5,1,2,11\n")
    frame = read_ohlcv(path)
    engineered = add_momentum_features(frame)
    assert "trades" not in frame
    assert len([column for column in engineered if "feature_" in column]) == 17
