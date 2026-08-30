"""Fetch the three Binance 4-hour datasets used by the current strategy."""

from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_URL = "https://data.binance.vision/data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "market"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _download(url: str) -> bytes | None:
    try:
        request = Request(url, headers={"User-Agent": "alpha-barrier/0.1"})
        with urlopen(request, timeout=60) as response:
            return response.read()
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def _months(start: str, end: str) -> list[str]:
    return [date.strftime("%Y-%m") for date in pd.date_range(start, end, freq="MS")]


def _timestamps(series: pd.Series) -> pd.DatetimeIndex:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    microseconds = values > 1e15
    nanoseconds = np.where(microseconds, values * 1_000.0, values * 1_000_000.0)
    return pd.DatetimeIndex(nanoseconds.astype("int64").astype("datetime64[ns]"))


def _read_archive(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw = archive.read(archive.namelist()[0])
    first_cell = raw[:200].decode("utf-8", "replace").split("\n", 1)[0].split(",")[0]
    has_header = any(character.isalpha() for character in first_cell)
    frame = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    frame = frame.iloc[:, : len(KLINE_COLUMNS)]
    frame.columns = KLINE_COLUMNS
    return frame


def fetch_klines(
    market: str,
    symbol: str,
    start: str,
    end: str,
    *,
    workers: int = 8,
) -> pd.DataFrame:
    prefix = (
        f"{BASE_URL}/spot/monthly"
        if market == "spot"
        else f"{BASE_URL}/futures/um/monthly"
    )
    urls = {
        month: f"{prefix}/klines/{symbol}/4h/{symbol}-4h-{month}.zip"
        for month in _months(start, end)
    }
    with ThreadPoolExecutor(workers) as pool:
        payloads = dict(zip(urls, pool.map(_download, urls.values())))
    frames = [_read_archive(payload) for payload in payloads.values() if payload]
    if not frames:
        raise RuntimeError(f"no data found for {market}/{symbol}")
    frame = pd.concat(frames, ignore_index=True)
    frame.index = _timestamps(frame.pop("open_time"))
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.drop(columns=["close_time", "ignore"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=pd.Timestamp.utcnow().strftime("%Y-%m"))
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "spot_btcusdt_4h.pkl": fetch_klines(
            "spot", "BTCUSDT", "2017-08", args.end
        ),
        "spot_ethusdt_4h.pkl": fetch_klines(
            "spot", "ETHUSDT", "2017-08", args.end
        ),
        "perp_btcusdt_4h.pkl": fetch_klines(
            "futures", "BTCUSDT", "2020-01", args.end
        ),
    }
    for filename, frame in datasets.items():
        frame.to_pickle(OUTPUT_DIR / filename)
        print(f"{filename}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
