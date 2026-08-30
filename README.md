# Alpha Barrier

Alpha Barrier is a Bitcoin direction-classification model. The current model is
a regularized logistic regression using causal price momentum, Binance taker
flow, and already-matured barrier outcomes.

> Research software, not financial advice. The result is exploratory only.

## Main result

The calendar is divided into exactly four years of development followed by one
year of later evaluation.

Development walk-forward, July 2021 through June 2025:

| Metric | Result |
|---|---:|
| Directional accuracy | **53.42%** (3,861 / 7,228) |
| Block-bootstrap 95% interval | **51.31–55.58%** |
| Causal majority baseline | 48.35% |
| Half-years above 50% | 8 / 8 |

At least 270 configurations were inspected. The figure is therefore a
development result and a hypothesis for future testing, not a performance
claim.

## Current model

- **Target:** BTC/USD four-hour bars, 12-bar horizon, symmetric ±1.5 ATR
  barriers.
- **Features:** 17 momentum features, 12 Binance spot/perpetual order-flow
  features, and 6 matured-outcome state features lagged 13 bars.
- **Classifier:** balanced logistic regression (`C=0.5`).
- **Calibration:** the decision threshold comes from the preceding purged
  12-month period before each six-month test window.
- **Evaluation:** anchored walk-forward training, a full label-horizon purge,
  final embargo, and a moving-block bootstrap.

## Backtesting example

One optimized execution example observed in the project converts the
classifier calls into a long-only position:

1. A confident up call enters long and a confident down call exits.
2. Calls within 0.02 of the calibrated boundary leave the previous state
   unchanged to reduce turnover.
3. A long position is allowed only while BTC closes above its causal 720-bar
   exponential moving average.
4. The backtest charges 0.1% commission for every entry and exit.

| Period | Return | Sharpe | Maximum drawdown | Exposure |
|---|---:|---:|---:|---:|
| Jul 2021–Jun 2025 development | **+150.20%** | **0.91** | **−25.02%** | 29.7% |
| Jul 2025–Jun 2026 later evaluation | **+2.51%** | **0.22** | **−8.51%** | 11.2% |

In the later period, directional accuracy was **55.48%** (1,048 / 1,889;
block-bootstrap 95% interval **51.30–59.82%**).

Machine-readable results:

- [`results/walk_forward.json`](results/walk_forward.json)
- [`results/later_evaluation.json`](results/later_evaluation.json)

## Reproduce

Python 3.12 or 3.13 is required. The frozen 2021–2025 inputs are checked in.

```bash
uv sync
uv run pytest -q
uv run python -m alphabarrier.strategy
```

The later evaluator accepts a data directory containing the same cache and
market-data layout extended through the requested test period:

```bash
uv run python -m scripts.evaluate_locked_strategy \
  --data-dir /path/to/extended/data \
  --output results/later_evaluation.json
```

To rebuild the frozen inputs:

```bash
uv run python scripts/prepare_bitstamp_data.py \
  training_data/Bitstamp_BTCUSD_1m_2012_2026.zip \
  --output training_data/bitstamp/XBTUSD_240.csv --force
uv run python -m scripts.build_cache
uv run python -m scripts.fetch_market_data --end 2025-12
```

## Repository layout

```text
alphabarrier/      direction model, evaluation, and backtesting example
data/              frozen processed inputs
results/           development and later evaluation results
scripts/           data preparation and later evaluation
tests/             feature, evaluation, and execution regression tests
```
