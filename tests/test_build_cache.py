import pandas as pd

from alphabarrier.features import get_feature_columns
from scripts.build_cache import build


def test_build_cache_respects_frozen_end_and_keeps_only_model_features(tmp_path):
    index = pd.date_range("2020-01-01", periods=40, freq="4h")
    rows = []
    for position, timestamp in enumerate(index):
        close = 100.0 + position
        rows.append(
            f"{int(timestamp.timestamp())},{close - 0.5},{close + 1.0},"
            f"{close - 1.0},{close},10"
        )
    source = tmp_path / "XBTUSD_240.csv"
    source.write_text("\n".join(rows) + "\n")
    destination = tmp_path / "cache.pkl"

    frame = build(source, destination, end=index[30])

    assert destination.exists()
    assert len(frame) == 30
    assert frame.index.max() == index[29]
    assert len(get_feature_columns(frame)) == 17
