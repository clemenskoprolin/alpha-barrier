import numpy as np
import pandas as pd
import pytest

from alphabarrier.execution import (
    FEE_SENSITIVITY_GRID,
    SIZING_FLOOR,
    TREND_EMA_LADDER,
    apply_rebalance_band,
    build_scaled_positions,
    build_strategy_positions,
    continuous_backtest,
    fee_sensitivity,
    reference_baselines,
    signal_size,
    trend_gate,
    trend_state,
)


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


def _signals(
    probabilities,
    threshold=0.5,
    logit_scale=0.4,
    start="2020-01-01",
    signal_span=5.0,
):
    index = pd.date_range(start, periods=len(probabilities), freq="4h")
    probabilities = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "side": np.where(probabilities >= threshold, 1, -1),
            "confidence": np.abs(probabilities - threshold),
            "probability": probabilities,
            "threshold": threshold,
            "logit_scale": logit_scale,
            "signal_span": signal_span,
        },
        index=index,
    )


def test_signal_size_is_bounded_monotone_and_neutral_at_the_threshold():
    signals = _signals([0.30, 0.45, 0.50, 0.55, 0.70])
    size = signal_size(signals, ewma_span=1, floor=0.0)
    assert ((size >= 0.0) & (size <= 1.0)).all()
    assert size.is_monotonic_increasing
    assert size.iloc[2] == pytest.approx(0.5)


def test_signal_size_rescales_each_fold_by_its_own_calibration_spread():
    wide = signal_size(_signals([0.55] * 4, logit_scale=2.0), ewma_span=1, floor=0.0)
    narrow = signal_size(_signals([0.55] * 4, logit_scale=0.1), ewma_span=1, floor=0.0)
    assert narrow.iloc[0] > wide.iloc[0] > 0.5


def test_signal_size_floor_reaches_flat_without_touching_strong_calls():
    signals = _signals([0.20, 0.45, 0.50, 0.90])
    unfloored = signal_size(signals, ewma_span=1, floor=0.0)
    floored = signal_size(signals, ewma_span=1)
    # The floor is what the binary gate had over the continuous rule: a weak
    # call reaches zero instead of leaving a residual position on the book.
    assert unfloored.iloc[0] > 0.0
    assert floored.iloc[0] == 0.0
    assert floored.iloc[3] == pytest.approx(
        (unfloored.iloc[3] - SIZING_FLOOR) / (1.0 - SIZING_FLOOR)
    )
    assert (floored <= unfloored).all()


def test_signal_size_smooths_each_fold_over_its_own_measured_span():
    step = [0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    slow = signal_size(_signals(step, signal_span=20.0), floor=0.0)
    quick = signal_size(_signals(step, signal_span=1.0), floor=0.0)
    assert quick.iloc[3] > slow.iloc[3]
    assert quick.iloc[3] == pytest.approx(signal_size(_signals(step), ewma_span=1, floor=0.0).iloc[3])

    mixed = _signals(step, signal_span=20.0)
    mixed.loc[mixed.index[3:], "signal_span"] = 1.0
    # Each fold restarts its own smoothing, so the fast fold reacts at once.
    assert signal_size(mixed, floor=0.0).iloc[3] == pytest.approx(quick.iloc[3])


def test_signal_size_rejects_missing_columns_and_an_out_of_range_floor():
    frame = pd.DataFrame({"side": [1], "confidence": [0.1]}, index=[pd.Timestamp("2020")])
    with pytest.raises(ValueError, match="logit_scale"):
        signal_size(frame)
    with pytest.raises(ValueError, match="signal_span"):
        signal_size(_signals([0.6]).drop(columns="signal_span"))
    with pytest.raises(ValueError, match="floor"):
        signal_size(_signals([0.6]), floor=1.0)


def test_trend_gate_grades_the_ladder_and_reduces_to_one_span():
    index = pd.date_range("2020-01-01", periods=400, freq="4h")
    frame = pd.DataFrame(
        {"raw_close": np.r_[np.linspace(100.0, 200.0, 200), np.linspace(200.0, 150.0, 200)]},
        index=index,
    )
    ladder = trend_gate(frame, (10, 50, 250))
    assert ((ladder >= 0.0) & (ladder <= 1.0)).all()
    assert set(np.round(ladder.unique(), 6)) <= {
        round(share, 6) for share in (0.0, 1 / 3, 2 / 3, 1.0)
    }
    # Partial agreement is the whole point: the short span turns off first.
    assert (ladder.between(0.0, 1.0, inclusive="neither")).any()
    pd.testing.assert_series_equal(
        trend_gate(frame, 50), trend_state(frame, trend_ema_span=50), check_names=False
    )
    with pytest.raises(ValueError, match="must not be empty"):
        trend_gate(frame, ())


def test_rebalance_band_holds_until_the_target_drifts_far_enough():
    target = pd.Series([0.50, 0.55, 0.62, 0.62, 0.30])
    np.testing.assert_allclose(
        apply_rebalance_band(target, 0.10).to_numpy(), [0.50, 0.50, 0.62, 0.62, 0.30]
    )
    pd.testing.assert_series_equal(
        apply_rebalance_band(target, 0.0), target.rename("position"), check_names=False
    )


def test_scaled_positions_stay_flat_below_the_trend_gate():
    signals = _signals([0.9] * 6)
    falling = pd.DataFrame(
        {"raw_close": np.linspace(120.0, 100.0, 7)},
        index=pd.date_range("2020-01-01", periods=7, freq="4h"),
    )
    positions = build_scaled_positions(
        signals, falling, ewma_span=1, rebalance_band=0.0, trend_spans=2
    )
    assert (positions == 0.0).all()


def test_reference_baselines_price_the_do_nothing_comparisons():
    index = pd.date_range("2020-01-01", periods=6, freq="4h")
    frame = pd.DataFrame({"raw_close": np.linspace(100.0, 110.0, 7)},
                         index=pd.date_range("2020-01-01", periods=7, freq="4h"))
    baselines = reference_baselines(index, frame, signals=_signals([0.6] * 6),
                                    trend_spans=2, trend_ema_span=2)
    assert set(baselines) == {
        "buy_and_hold",
        "trend_only",
        "trend_only_single_span",
        "model_only_binary",
        "model_only_scaled",
    }
    assert baselines["buy_and_hold"]["fee_sides"] == 2.0
    assert baselines["buy_and_hold"]["mean_weight"] == pytest.approx(1.0)


def test_fee_sensitivity_return_never_rises_with_the_commission():
    index = pd.date_range("2020-01-01", periods=7, freq="4h")
    frame = pd.DataFrame({"raw_close": np.linspace(100.0, 112.0, 7)}, index=index)
    positions = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=index[:6])
    rows = fee_sensitivity(positions, frame)
    returns = [row["total_return"] for row in rows]
    assert [row["commission_per_side"] for row in rows] == list(FEE_SENSITIVITY_GRID)
    assert returns == sorted(returns, reverse=True)
