import csv
import zipfile

from scripts.prepare_bitstamp_data import prepare


def test_prepare_aggregates_only_complete_intervals(tmp_path):
    archive_path = tmp_path / "archive.zip"
    output = tmp_path / "bitstamp" / "XBTUSD_2.csv"
    rows = [
        "Timestamp,Open,High,Low,Close,Volume",
        "60,10,12,9,11,1.5",  # incomplete 00:00 bucket (missing timestamp 0)
        "120,11,13,10,12,2.0",
        "180,12,14,11,13,3.0",
        "240,13,15,12,14,4.0",  # incomplete final bucket
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/btcusd_1-min_data.csv", "\n".join(rows) + "\n")

    report = prepare(archive_path, output, interval_minutes=2)

    with output.open(newline="") as handle:
        aggregated = list(csv.reader(handle))
    assert report.source_rows == 4
    assert report.output_rows == 1
    assert report.incomplete_intervals_dropped == 2
    assert aggregated == [["120", "11.0", "14.0", "10.0", "13.0", "5.0"]]
    assert (output.parent / "dataset.json").exists()
