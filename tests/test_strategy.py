import numpy as np
import pandas as pd
import pytest

from alphabarrier.evaluation import Window
from alphabarrier.features import DOWN, SIDEWAYS, UP
from alphabarrier.strategy import (
    DEVELOPMENT_START,
    DEVELOPMENT_SIGNAL_END,
    LABEL_EMBARGO,
    LATER_EVALUATION_END,
    LATER_EVALUATION_START,
    build_mature_outcome_features,
    choose_decision_threshold,
    evaluate,
)


def test_mature_outcome_state_is_prefix_invariant_and_safely_lagged():
    index = pd.date_range("2020-01-01", periods=100, freq="4h")
    target = pd.Series(
        np.resize(np.array([DOWN, SIDEWAYS, UP, np.nan]), len(index)), index=index
    )
    changed = target.copy()
    observation = 70
    changed.iloc[observation - 12 :] = UP
    original = build_mature_outcome_features(target, spans=(3, 9))
    altered = build_mature_outcome_features(changed, spans=(3, 9))
    pd.testing.assert_series_equal(original.iloc[observation], altered.iloc[observation])
    cutoff = 80
    prefix = build_mature_outcome_features(target.iloc[:cutoff], spans=(3, 9))
    pd.testing.assert_frame_equal(prefix, original.iloc[:cutoff])


def test_mature_outcome_state_rejects_unsafe_configuration():
    target = pd.Series([DOWN, SIDEWAYS, UP] * 10, dtype=float)
    with pytest.raises(ValueError, match="must exceed"):
        build_mature_outcome_features(target, maturity_lag=12, spans=(3,))
    with pytest.raises(ValueError, match="positive integers"):
        build_mature_outcome_features(target, spans=(0,))


def test_threshold_ignores_sideways_and_breaks_ties_toward_half():
    target = pd.Series([DOWN, SIDEWAYS, DOWN, UP, UP, np.nan])
    probability = np.asarray([0.48, 0.99, 0.49, 0.51, 0.52, 0.01])
    threshold, accuracy, rows = choose_decision_threshold(
        probability, target, thresholds=(0.49, 0.50, 0.51)
    )
    assert threshold == 0.50
    assert accuracy == 1.0
    assert rows == 4


def test_final_development_labels_are_embargoed():
    assert LABEL_EMBARGO == pd.Timedelta(hours=48)
    assert LATER_EVALUATION_START == DEVELOPMENT_START + pd.DateOffset(years=4)
    assert LATER_EVALUATION_END == LATER_EVALUATION_START + pd.DateOffset(years=1)
    assert DEVELOPMENT_SIGNAL_END == LATER_EVALUATION_START - LABEL_EMBARGO


def test_evaluation_calls_every_test_row():
    index = pd.date_range("2018-01-01", periods=6_000, freq="4h")
    phase = np.arange(len(index), dtype=float)
    target = pd.Series(
        np.resize(np.array([DOWN, UP, SIDEWAYS, UP, DOWN]), len(index)), index=index
    )
    frame = pd.DataFrame(
        {
            "feature_a": np.sin(phase / 11.0),
            "feature_b": np.cos(phase / 17.0),
            "target": target,
        },
        index=index,
    )
    test_start = index[5_000]
    window = Window(
        train_start=index[0],
        train_end=test_start - LABEL_EMBARGO,
        test_start=test_start,
        test_end=index[5_400],
    )
    scores, correct, signals = evaluate(
        frame, ["feature_a", "feature_b"], [window]
    )
    test = frame.loc[(frame.index >= window.test_start) & (frame.index < window.test_end)]
    directional = test["target"].notna() & test["target"].ne(SIDEWAYS)
    assert len(scores) == 1
    assert len(signals) == len(test)
    assert len(correct) == int(directional.sum())
    assert set(signals["side"].unique()) <= {-1, 1}
