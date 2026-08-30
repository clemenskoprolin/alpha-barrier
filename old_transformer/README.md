# Alpha Barrier — archived Transformer version

> Historical snapshot only. The current implementation and supported commands
> live in the repository root. This version is not installed or tested by the
> current package.

Transformer-based Bitcoin direction classifier with volatility-adaptive triple-barrier labels.

Alpha Barrier predicts `DOWN`, `SIDEWAYS`, or `UP` from 4-hour Bitcoin OHLCVT candles. The current pipeline is designed around one requirement: a result must survive a chronological, untouched out-of-sample holdout before it becomes a performance claim.

> **Research software, not financial advice.** Directional accuracy does not imply profitability. Fees, slippage, latency, class balance, signal coverage, and position sizing all matter.

## What changed

The original implementation reported 50.43% average time-series cross-validation accuracy, but that number should not be compared directly with results from the current pipeline. The earlier pipeline had three important defects:

- positional encodings were broadcast over the batch axis instead of the time axis;
- a full-sample volume z-score leaked future distribution statistics;
- labels near validation boundaries could inspect prices inside the validation period.

The current implementation fixes those issues and adds:

- causal, scale-free features instead of raw price-level and cumulative features;
- a symmetric, fixed 12-bar (48-hour) triple-barrier target;
- exclusion of samples where both barriers touch within one OHLC bar and ordering is unknowable;
- a compact regularized Transformer (`64` hidden units, one encoder block);
- a regularized direction head on the 17 momentum/trend features that survived development-fold selection;
- a fixed hybrid direction score (70% direction head, 30% Transformer), selected before external evaluation;
- purged expanding-window cross-validation for epoch selection;
- one untouched final 15% chronological holdout;
- saved predictions, probabilities, baselines, coverage, and class metrics;
- an out-of-sample-only backtest boundary stored in the model artifact.

## The 53.4% target

This repository does **not** claim 53.4% until a run on the full dataset produces it. The authoritative result is `artifacts/metrics.json`, not a hand-copied README number.

### Current measured result (2026-08-30)

The locked hybrid reached **54.05% mean directional accuracy** across five purged Bitstamp development folds through 2023, compared with a **52.24% mean directional-majority baseline**. That development result did **not** survive the separate future/venue test:

> On Kraken from 2024-02-08 through 2025-12-29, the locked hybrid achieved **50.40% directional accuracy** (95% CI **48.75–52.04%**) across **3,536** resolved directional outcomes, versus a **50.08%** directional-majority baseline.

Accordingly, the repository still does **not** support a 53.4% external out-of-sample claim. The failed transfer is retained rather than tuning against the external set after seeing it.

The corresponding out-of-sample strategy diagnostic is also negative: with three-bar signal confirmation and 0.1% commission on entry and exit, the hybrid returned **−75.46%** while buy-and-hold returned **+95.83%** over the same backtest window. This is evidence against deployment, not a parameter-tuning invitation on the now-observed test period.

The interval above is a Wilson binomial interval and assumes independent observations. That assumption is optimistic here because adjacent labels share most of the same 48-hour future path; holdout correctness has 0.57 lag-one autocorrelation. A lag-12 Newey–West sensitivity interval for the hybrid's edge over the majority baseline is much wider (**−5.26 to +5.88 percentage points**). The point estimate is unchanged, but the effective evidence is smaller than 3,536 independent coin flips.

### Positioning-data follow-up (2026-08-30)

A follow-up experiment tested whether genuinely new information helps more than additional technical indicators:

- all 57 existing OHLCV features, nonlinear tree models, rolling windows, and recency weighting failed to improve the development result;
- 16 features reconstructed from the one-minute path inside each four-hour Bitstamp candle also failed to improve it;
- Binance taker-flow context was mildly helpful on restricted 2020–2023 development folds, but the benefit did not persist in the final fold;
- an 11-feature model using only open-interest dynamics and the global account long/short ratio reached **54.03% mean directional accuracy** on five purged 2020–2023 development folds, versus a **53.92%** mean fold-majority baseline.

That positioning model was then locked and evaluated once on a new Bitstamp/Binance holdout from **2026-01-01 through 2026-07-31**. It achieved **53.32%** (578/1,084; independence-assuming 95% CI **50.34–56.27%**) versus a **54.15%** directional-majority baseline. A lag-12 Newey–West sensitivity interval was **47.77–58.87%**. This is closer to 54%, but it is not an edge and does not rescue the external claim. Full results are in `artifacts/positioning_experiment.json`.

### Causally calibrated order-flow follow-up (2026-08-30)

The strongest post-hoc development candidate combines the 17 momentum features with 12 Binance spot/perpetual taker-flow features. A new nested walk-forward experiment fixes the confidence threshold on the preceding purged 12-month calibration period, then scores each six-month test period without using its confidence distribution. It also corrects an inclusive-boundary error in the research harness: training and test endpoints are now explicitly exclusive.

Across nine development windows from **2021-07-01 through 2025-12-31**, the always-call refit model reached only **52.32%** directional accuracy. Abstaining on low-confidence bars crossed 54%, but changed the question materially:

| requested calibration coverage | realized coverage | directional accuracy | block-bootstrap 95% CI | fee-aware return | profitable half-years |
|---|---:|---:|---:|---:|---:|
| 100% | 100.00% | 52.32% | 50.36–54.17% | −94.55% | 1/9 |
| 50% | 43.91% | 55.66% | 52.63–58.50% | −37.11% | 5/9 |
| 30% | 26.37% | 58.14% | 54.70–62.14% | +28.59% | 4/9 |

The backtest uses one non-overlapping position, a 12-bar ±1.5 ATR bracket, adverse ordering when both barriers touch in one candle, and 0.1% commission per side. The 30% result has only a **0.20 trade Sharpe**, is profitable in fewer than half the test windows, excludes slippage, and trails buy-and-hold; it is therefore fragile development evidence, not a deployable strategy result. It also does not meet an always-call 54% target because it abstains on about three quarters of bars.

Most importantly, a separate 2026 diagnostic did not confirm the order-flow family: after correcting its inclusive training endpoint, the price-plus-flow model scored **49.91%** at full coverage and **50.86%** at 37.5% selective coverage on 1,090 directional bars from January through July. The corresponding selective backtest averaged **−22.19 bps per trade** after commission. Because those 2026 labels have now been inspected, neither the stronger price-only arm nor a revised flow model may be claimed as out-of-sample on that interval. The next honest confirmation must begin after the last inspected timestamp.

A later boundary audit found one qualification: the final 12 development labels before `SEARCH_END` use the first 48 hours of the 2026 price path. The diagnostic was therefore not literally untouched, although this tiny overlap cannot rescue its negative result. New experiments must embargo a full label horizon before any claimed holdout; `research/exp12_adaptive_outcome_state.py` does so.

The evidence now says that more copies of BTC spot OHLCV are unlikely to be enough. The best data priorities are:

1. **Positioning across derivatives venues:** open interest, changes in open interest, global and top-trader long/short ratios, funding, and spot/perpetual basis from Binance plus a second long-history venue. This was the only tested category that approached 54% consistently enough to justify more work.
2. **Signed order flow and forced-flow events:** taker-buy/sell imbalance from aggregate trades, liquidations, and stable order-book imbalance snapshots. Four-hour candle volume discards who initiated the trade and where resting liquidity sat.
3. **Options state:** implied-volatility term structure, 25-delta skew, put/call open interest, and expiry concentration from Deribit or a normalized vendor. These may distinguish trend and crash-risk regimes that identical spot momentum features cannot.
4. **Cross-asset risk context:** ETH/BTC relative strength, Nasdaq, dollar, rates, and gold—using point-in-time release timestamps. These are cheaper to add but were weaker in the initial ETH-relative test.
5. **On-chain flows:** exchange inflow/outflow, spent-output profit measures, and stablecoin supply/flows. Daily frequency and publication latency make this a lower priority for a 48-hour target than derivatives positioning and order flow.

Kraken history is still useful for provenance and execution modeling, but it is not the missing predictive dataset: aligned Kraken/Bitstamp four-hour returns correlate **99.87%**, and their resolved directional labels agree **98.90%**. Multiple venues during the same dates also do not create independent target outcomes.

At this horizon, roughly 1,200 independent directional outcomes are needed to distinguish 54% from 50% with conventional 5% significance and 80% power. Because labels overlap for 12 bars, that means on the order of **5–8 years of genuinely forward, regime-diverse evidence**, not merely denser candles from the same period. A faster research path is to add independent information, pre-register the next model, and accumulate non-overlapping event blocks.

Metrics are intentionally separated:

- `three_class_accuracy`: exact `DOWN` / `SIDEWAYS` / `UP` accuracy on every holdout row;
- `directional_accuracy`: accuracy on true `DOWN` or `UP` outcomes; a `SIDEWAYS` prediction is wrong;
- `forced_directional_accuracy`: on those same directional outcomes, choose whichever of the model's `DOWN` and `UP` probabilities is larger; this is the conventional always-call direction metric;
- `directional_majority_baseline_accuracy`: always predicting the more common direction on those same rows;
- `signal_accuracy`: accuracy when the model predicts `DOWN` or `UP`; a true sideways outcome is wrong;
- `signal_coverage`: the fraction of holdout rows on which the model predicts a direction;
- `direction_head_accuracy`: forced up/down accuracy from the regularized linear momentum head;
- `hybrid_directional_accuracy`: forced up/down accuracy from the predeclared 70/30 direction-head/Transformer blend;
- `majority_baseline_accuracy`: the no-skill majority-class comparator.

A defensible result statement should include the holdout dates, sample count, metric definition, baseline, and—when quoting signal accuracy—coverage. For example:

> On the untouched chronological holdout from **[start]** through **[end]**, Alpha Barrier achieved **[x]% directional accuracy** (95% CI **[low–high]**) across **[n] directional outcomes**, versus a **[b]% directional-majority baseline**. Directional signals covered **[c]%** of all holdout bars.

Do not fill those placeholders from cross-validation folds or after repeatedly tuning against the holdout.

## Architecture

```text
Input sequence (64 × N features)
  → linear projection + layer normalization
  → batch-first sinusoidal positional encoding
  → 1 × Transformer encoder (4 heads, GELU, pre-norm)
  → final-token representation
  → regularized 3-class head

17 selected momentum features at the signal bar
  → regularized logistic direction head

direction score = 70% logistic + 30% Transformer up/down probability
```

The old 512-unit, four-layer network had far more capacity than this dataset warranted. The smaller model is faster to iterate and less prone to memorizing a market regime.

## Setup

This workspace includes a local `.python-runtime` and populated `.venv`, so the
commands below do not depend on a temporary interpreter. To rebuild the
environment elsewhere, the project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Data sources

### Long Bitstamp history (recommended for model development)

The [Bitcoin Historical Data dataset on Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) contains one-minute Bitstamp BTC/USD OHLCV from January 2012 onward. Its [public updater](https://github.com/mczielinski/kaggle-bitcoin) fetches new candles from Bitstamp's API. Prepare it without extracting the 390 MB CSV:

```bash
.venv/bin/python scripts/prepare_bitstamp_data.py \
  training_data/Bitstamp_BTCUSD_1m_2012_2026.zip
```

This streams `btcusd_1-min_data.csv`, retains only complete 4-hour intervals, and writes `training_data/bitstamp/XBTUSD_240.csv` plus a provenance manifest. Bitstamp does not provide the Kraken trade-count column, so the pipeline omits trade-count features for this source.

Do not splice Bitstamp and Kraken candles into one time series. Venue changes create artificial price/volume discontinuities and can invalidate an out-of-sample claim.

### Kraken archive

Kraken's download page describes a complete OHLCVT archive from each market's inception. However, the archive observed on 2026-08-30 contained `XBTUSD_240.csv` only from 2024-01-01 through 2025-12-31. Always trust the preparation script's reported dates instead of the download label.

The complete ZIP is several gigabytes. It is unnecessary to unpack the entire archive. Pass the ZIPs to the included extractor in oldest-to-newest order; it reads only `XBTUSD_240.csv`, merges quarterly updates, and reports coverage and gaps:

```bash
.venv/bin/python scripts/prepare_kraken_data.py \
  ~/Downloads/Kraken_OHLCVT.zip \
  ~/Downloads/Kraken_OHLCVT_Q4_2024.zip \
  ~/Downloads/Kraken_OHLCVT_Q1_2025.zip \
  --output training_data/kraken/XBTUSD_240.csv
```

You can also pass a directory containing ZIPs, but explicit chronological arguments are safer because later files replace duplicate timestamps. Use `--force` only when you intend to replace an existing merged CSV.

Kraken omits intervals in which no trades occurred. The preparation script reports non-240-minute gaps and leaves them unfilled; for liquid XBT/USD history these should be investigated rather than silently synthesized.

### Binance positioning and derivatives context

The [Binance public archive](https://github.com/binance/binance-public-data) provides monthly spot/futures klines and funding, plus daily five-minute positioning files. The preparation script derives only causal, scale-free context: taker imbalance, ETH-relative momentum, perpetual basis/premium/funding, open-interest changes, and long/short ratios.

```bash
uv run python scripts/prepare_binance_context.py \
  --start 2019-09 \
  --end 2026-08 \
  --positioning-start 2020-09-01 \
  --positioning-end 2026-08-29 \
  --output training_data/binance/context_240.csv
```

Reproduce the locked positioning experiment with:

```bash
uv run python -m scripts.evaluate_positioning_model \
  --context training_data/binance/context_240.csv \
  --development-end 2023-12-29T20:00:00 \
  --holdout-start 2026-01-01T00:00:00 \
  --output artifacts/positioning_experiment.json
```

If the feature set, regularization, target, or preprocessing changes after reading the 2026 result, 2026 is development data and the next claim requires a later holdout.

## Train and evaluate

```bash
.venv/bin/python -m alphabarrier.transformer \
  --data-dir training_data/bitstamp \
  --output-dir artifacts
```

For the stricter future, cross-venue protocol used in this repository, tune and train on Bitstamp only through 2023, then evaluate once on Kraken 2024–2025:

```bash
.venv/bin/python -m alphabarrier.transformer \
  --data-dir training_data/bitstamp \
  --development-end 2023-12-31T20:00:00 \
  --evaluation-data-dir training_data/kraken \
  --output-dir artifacts
```

The run writes:

- `artifacts/final_transformer_model.pt` — weights plus model/data metadata and holdout boundary;
- `artifacts/final_scaler.pkl` — scaler fit only on the development period;
- `artifacts/direction_classifier.pkl` — regularized momentum direction head;
- `artifacts/metrics.json` — fold results and untouched holdout metrics;
- `artifacts/holdout_predictions.npz` — timestamps, labels, predictions, and probabilities.

The final holdout is evaluated once. If you tune features or hyperparameters after reading it, it has become development data; move the holdout forward before making a new out-of-sample claim. The checked-in configuration is frozen at the 2026-08-30 external result above.

For a quick pipeline check before a full run:

```bash
.venv/bin/python -m alphabarrier.transformer --epochs 2 --splits 3
```

## Backtest

```bash
.venv/bin/python -m testing.transformer_backtesting
```

The backtester loads feature/model metadata from the artifact and restricts its output to the untouched holdout period. It uses 4-hour candles consistently; the former unused one-minute-data instruction has been removed.

Classification and strategy evaluation answer different questions. Treat the backtest as a second out-of-sample diagnostic, and compare it with buy-and-hold after fees and slippage before interpreting its return.
