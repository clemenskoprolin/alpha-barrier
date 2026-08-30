import io
import zipfile

from scripts.fetch_market_data import KLINE_COLUMNS, _read_archive


def _archive(contents: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-4h.csv", contents)
    return buffer.getvalue()


def test_read_archive_accepts_headered_binance_files():
    values = ["1"] * len(KLINE_COLUMNS)
    payload = _archive(",".join(KLINE_COLUMNS) + "\n" + ",".join(values) + "\n")

    frame = _read_archive(payload)

    assert len(frame) == 1
    assert frame.columns.tolist() == KLINE_COLUMNS


def test_read_archive_accepts_headerless_binance_files():
    values = [str(index) for index in range(len(KLINE_COLUMNS))]
    frame = _read_archive(_archive(",".join(values) + "\n"))

    assert len(frame) == 1
    assert frame.columns.tolist() == KLINE_COLUMNS
