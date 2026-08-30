import numpy as np
import pandas as pd

from alphabarrier.evaluation import block_bootstrap_ci, make_windows


def test_walk_forward_window_has_exclusive_full_horizon_purge():
    windows = make_windows(
        first_test="2021-07-01",
        last_test=pd.Timestamp("2022-01-01"),
        purge_bars=12,
    )
    assert len(windows) == 1
    assert windows[0].train_end == windows[0].test_start - pd.Timedelta(hours=48)


def test_block_bootstrap_is_deterministic_and_bounded():
    correct = np.resize(np.array([True, True, False]), 180)
    first = block_bootstrap_ci(correct, block=12, draws=200, seed=7)
    second = block_bootstrap_ci(correct, block=12, draws=200, seed=7)
    assert first == second
    assert 0.0 <= first[0] <= correct.mean() <= first[1] <= 1.0
