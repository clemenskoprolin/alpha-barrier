import numpy as np
import pandas as pd
import torch

from alphabarrier.transformer import (
    PositionalEncoding,
    TimeSeriesDataset,
    add_technical_features,
    calculate_metric_dict,
    get_triple_barrier_labels,
    read_ohlcvt,
)


def _barrier_case(second_high: float, second_low: float) -> pd.Series:
    prices = pd.DataFrame(
        {
            "high": [100.0, second_high, 100.5],
            "low": [100.0, second_low, 99.5],
            "close": [100.0, 100.0, 100.0],
        }
    )
    atr = pd.Series([1.0, 1.0, 1.0])
    return get_triple_barrier_labels(
        prices, atr, pt_mult=1.0, sl_mult=1.0, horizon_bars=2
    )


def test_batch_first_positional_encoding_is_not_batch_dependent():
    encoding = PositionalEncoding(d_model=8, max_len=4)
    result = encoding(torch.zeros(2, 4, 8))

    torch.testing.assert_close(result[0], result[1])
    assert not torch.equal(result[0, 0], result[0, 1])


def test_triple_barrier_handles_first_touch_and_excludes_unknown_intrabar_order():
    assert _barrier_case(second_high=101.0, second_low=99.5).iloc[0] == 1
    assert _barrier_case(second_high=100.5, second_low=99.0).iloc[0] == -1
    assert np.isnan(_barrier_case(second_high=101.0, second_low=99.0).iloc[0])


def test_triple_barrier_does_not_shorten_final_horizons():
    labels = _barrier_case(second_high=101.0, second_low=99.5)

    assert labels.iloc[-2:].isna().all()


def test_sequence_includes_the_current_signal_bar():
    frame = pd.DataFrame({"feature": [10.0, 20.0, 30.0], "target": [0, 1, 2]})
    dataset = TimeSeriesDataset(
        frame, ["feature"], "target", seq_len=2, target_indices=[1]
    )

    features, target = dataset[0]
    torch.testing.assert_close(features[:, 0], torch.tensor([10.0, 20.0]))
    assert target.item() == 1


def test_unlabelled_bar_remains_sequence_context_but_not_a_target():
    frame = pd.DataFrame(
        {"feature": [10.0, 20.0, 30.0], "target": [0.0, np.nan, 2.0]}
    )
    dataset = TimeSeriesDataset(frame, ["feature"], "target", seq_len=2)

    assert dataset.indices.tolist() == [2]
    features, target = dataset[0]
    torch.testing.assert_close(features[:, 0], torch.tensor([20.0, 30.0]))
    assert target.item() == 2


def test_feature_engineering_does_not_change_the_past_when_future_changes():
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
    original = add_technical_features(frame)
    changed = frame.copy()
    changed.loc[index[-1], ["open", "high", "low", "close", "volume"]] *= 10
    recomputed = add_technical_features(changed)

    feature_columns = [column for column in original if column.startswith("feature_")]
    pd.testing.assert_series_equal(
        original.loc[index[-2], feature_columns],
        recomputed.loc[index[-2], feature_columns],
        check_names=False,
    )


def test_metrics_report_accuracy_and_signal_coverage_separately():
    truth = np.array([0, 1, 2, 2])
    prediction = np.array([0, 0, 1, 2])

    metrics = calculate_metric_dict(truth, prediction)

    assert metrics["three_class_accuracy"] == 0.5
    assert metrics["directional_accuracy"] == 2 / 3
    assert metrics["directional_samples"] == 3
    assert metrics["directional_majority_baseline_accuracy"] == 2 / 3
    assert metrics["signal_accuracy"] == 2 / 3
    assert metrics["signal_coverage"] == 0.75


def test_probability_metric_forces_a_direction_without_hiding_three_class_errors():
    truth = np.array([0, 1, 2, 2])
    prediction = np.array([1, 1, 1, 2])
    probabilities = np.array(
        [
            [0.40, 0.50, 0.10],
            [0.20, 0.60, 0.20],
            [0.30, 0.50, 0.20],
            [0.10, 0.20, 0.70],
        ]
    )

    metrics = calculate_metric_dict(truth, prediction, probabilities)

    assert metrics["three_class_accuracy"] == 0.5
    assert metrics["directional_accuracy"] == 1 / 3
    assert metrics["forced_directional_accuracy"] == 2 / 3


def test_six_column_ohlcv_omits_trade_features(tmp_path):
    path = tmp_path / "BTCUSD_240.csv"
    path.write_text("100,1,2,0.5,1.5,10\n200,1.5,2.5,1,2,11\n")

    frame = read_ohlcvt(path)
    engineered = add_technical_features(frame)

    assert "trades" not in frame
    assert "feature_trade_count_change" not in engineered
    assert "feature_average_trade_size" not in engineered
