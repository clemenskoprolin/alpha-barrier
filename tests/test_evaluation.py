import numpy as np
import pandas as pd
import pytest

from alphabarrier.evaluation import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    make_windows,
)


def test_walk_forward_window_has_exclusive_full_horizon_purge():
    windows = make_windows(
        first_test="2021-07-01",
        last_test=pd.Timestamp("2022-01-01"),
        purge_bars=12,
    )
    assert len(windows) == 2
    assert [window.test_start for window in windows] == [
        pd.Timestamp("2021-07-01"),
        pd.Timestamp("2021-10-01"),
    ]
    for window in windows:
        assert window.train_end == window.test_start - pd.Timedelta(hours=48)


def test_block_bootstrap_is_deterministic_and_bounded():
    correct = np.resize(np.array([True, True, False]), 180)
    first = block_bootstrap_ci(correct, block=12, draws=200, seed=7)
    second = block_bootstrap_ci(correct, block=12, draws=200, seed=7)
    assert first == second
    assert 0.0 <= first[0] <= correct.mean() <= first[1] <= 1.0


def test_deflated_sharpe_falls_as_the_screen_widens():
    returns = np.random.default_rng(3).normal(0.0004, 0.01, 4_000)
    narrow = deflated_sharpe_ratio(returns, trials=5, trial_sharpe_variance=1e-4)
    wide = deflated_sharpe_ratio(returns, trials=5_000, trial_sharpe_variance=1e-4)
    assert 0.0 <= wide["deflated_sharpe_ratio"] < narrow["deflated_sharpe_ratio"] <= 1.0
    assert wide["expected_max_sharpe"] > narrow["expected_max_sharpe"]
    assert wide["observed_sharpe"] == pytest.approx(narrow["observed_sharpe"])


def test_deflated_sharpe_declines_to_score_degenerate_input():
    for kwargs in (
        {"trials": 320, "trial_sharpe_variance": 0.0},
        {"trials": 1, "trial_sharpe_variance": 1e-4},
    ):
        assert deflated_sharpe_ratio(np.ones(500), **kwargs)["deflated_sharpe_ratio"] is None
    assert (
        deflated_sharpe_ratio(np.ones(500), trials=320, trial_sharpe_variance=1e-4)[
            "deflated_sharpe_ratio"
        ]
        is None
    )
