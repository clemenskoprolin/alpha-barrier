"""Aggregate the Kaggle/Bitstamp one-minute BTC/USD archive to OHLCV bars.

The source archive maintained at
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data contains
``btcusd_1-min_data.csv``.  Its updater fetches BTC/USD candles from the
Bitstamp API.  This utility streams that large member, so the 390 MB CSV never
needs to be extracted to disk or loaded into memory in full.

Only complete output intervals are retained. The resulting headerless file uses
six columns: timestamp, open, high, low, close, volume.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


EXPECTED_HEADER = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
DEFAULT_MEMBER = "btcusd_1-min_data.csv"
SOURCE_URL = "https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data"


@dataclass(frozen=True)
class MinuteBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class PreparationReport:
    source_rows: int
    output_rows: int
    incomplete_intervals_dropped: int
    first_timestamp: int
    last_timestamp: int
    interval_minutes: int


def _source_stream(source: Path, member_name: str) -> tuple[TextIO, object | None]:
    if source.suffix.lower() != ".zip":
        return source.open("r", encoding="utf-8", newline=""), None

    archive = zipfile.ZipFile(source)
    matches = [name for name in archive.namelist() if Path(name).name == member_name]
    if len(matches) != 1:
        archive.close()
        raise ValueError(
            f"Expected one {member_name} in {source}, found {len(matches)}: {matches[:5]}"
        )
    binary = archive.open(matches[0])
    return io.TextIOWrapper(binary, encoding="utf-8", newline=""), archive


def iter_minute_bars(source: Path, member_name: str = DEFAULT_MEMBER) -> Iterator[MinuteBar]:
    stream, owner = _source_stream(source, member_name)
    try:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(
                f"Unexpected columns in {source}: {reader.fieldnames}; expected {EXPECTED_HEADER}"
            )
        previous_timestamp: int | None = None
        for row_number, row in enumerate(reader, start=2):
            try:
                bar = MinuteBar(
                    timestamp=int(row["Timestamp"]),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric row at {source}:{row_number}: {row}") from error
            if previous_timestamp is not None and bar.timestamp <= previous_timestamp:
                raise ValueError(
                    f"Timestamps are not strictly increasing at {source}:{row_number}"
                )
            previous_timestamp = bar.timestamp
            yield bar
    finally:
        stream.close()
        if owner is not None:
            owner.close()


def aggregate_complete_intervals(
    rows: Iterator[MinuteBar], interval_minutes: int
) -> tuple[list[list[float | int]], PreparationReport]:
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be positive")
    interval_seconds = interval_minutes * 60
    output: list[list[float | int]] = []
    source_rows = 0
    incomplete = 0
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    bucket: int | None = None
    bucket_rows: list[MinuteBar] = []

    def flush() -> None:
        nonlocal incomplete
        if not bucket_rows or bucket is None:
            return
        complete = (
            len(bucket_rows) == interval_minutes
            and bucket_rows[0].timestamp == bucket
            and bucket_rows[-1].timestamp == bucket + interval_seconds - 60
            and all(
                current.timestamp - previous.timestamp == 60
                for previous, current in zip(bucket_rows, bucket_rows[1:])
            )
        )
        if not complete:
            incomplete += 1
            return
        output.append(
            [
                bucket,
                bucket_rows[0].open,
                max(row.high for row in bucket_rows),
                min(row.low for row in bucket_rows),
                bucket_rows[-1].close,
                sum(row.volume for row in bucket_rows),
            ]
        )

    for row in rows:
        source_rows += 1
        first_timestamp = row.timestamp if first_timestamp is None else first_timestamp
        last_timestamp = row.timestamp
        row_bucket = row.timestamp - row.timestamp % interval_seconds
        if bucket is None:
            bucket = row_bucket
        elif row_bucket != bucket:
            flush()
            bucket_rows = []
            bucket = row_bucket
        bucket_rows.append(row)
    flush()

    if first_timestamp is None or last_timestamp is None:
        raise ValueError("Source contains no data rows")
    report = PreparationReport(
        source_rows=source_rows,
        output_rows=len(output),
        incomplete_intervals_dropped=incomplete,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        interval_minutes=interval_minutes,
    )
    return output, report


def _atomic_csv_write(rows: list[list[float | int]], output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json_write(payload: dict[str, object], output: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def prepare(
    source: Path,
    output: Path,
    *,
    member_name: str = DEFAULT_MEMBER,
    interval_minutes: int = 240,
    force: bool = False,
) -> PreparationReport:
    rows, report = aggregate_complete_intervals(
        iter_minute_bars(source, member_name), interval_minutes
    )
    _atomic_csv_write(rows, output, force)
    metadata = {
        "venue": "Bitstamp",
        "pair": "BTC/USD",
        "source_url": SOURCE_URL,
        "source_file": str(source.resolve()),
        "source_member": member_name,
        "output_schema": ["timestamp", "open", "high", "low", "close", "volume"],
        **asdict(report),
        "output_first_utc": datetime.fromtimestamp(rows[0][0], timezone.utc).isoformat(),
        "output_last_utc": datetime.fromtimestamp(rows[-1][0], timezone.utc).isoformat(),
    }
    _atomic_json_write(metadata, output.parent / "dataset.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Kaggle archive.zip or extracted CSV")
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--interval", type=int, default=240, help="Output interval in minutes")
    parser.add_argument(
        "--output", type=Path, default=Path("training_data/bitstamp/XBTUSD_240.csv")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = prepare(
        arguments.source,
        arguments.output,
        member_name=arguments.member,
        interval_minutes=arguments.interval,
        force=arguments.force,
    )
    first = datetime.fromtimestamp(report.first_timestamp, timezone.utc).isoformat()
    last = datetime.fromtimestamp(report.last_timestamp, timezone.utc).isoformat()
    print(f"Read {report.source_rows:,} one-minute rows: {first} through {last}")
    print(
        f"Wrote {report.output_rows:,} complete {report.interval_minutes}-minute bars to "
        f"{arguments.output.resolve()}"
    )
    print(f"Dropped incomplete edge/gap intervals: {report.incomplete_intervals_dropped:,}")
    print(f"Wrote provenance to {(arguments.output.parent / 'dataset.json').resolve()}")


if __name__ == "__main__":
    main()
