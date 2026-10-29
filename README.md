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
| Directional accuracy | **53.06%** (3,835 / 7,228) |
| Block-bootstrap 95% interval | **51.05–55.09%** |
| Causal majority baseline | 48.74% |
| Quarters above 50% | 13 / 16 |

At least 320 configurations were inspected. The figure is therefore a
development result and a hypothesis for future testing, not a performance
claim.

## The execution rule the development window chose did not replicate

The four development years are where every execution choice was made, so they
cannot also be the evidence that the choices were right. The cache reaches back
to 2012. Only the order-flow block does not: momentum and matured-outcome state
cover the whole span, and since the question is what to do with a call rather
than how the call is produced, that reduced 23-feature model is enough to ask
it. It buys seven and a half extra years and three more bear markets.

The previous release replaced a binary trend-gated switch with continuous
sizing and reported +0.31 Sharpe for the change. Run the same comparison on the
history before the development window and the sign flips. Mean Sharpe across
fold-grid shifts:

| Rule | 2014–2021 (before) | 2021–2025 (development) | 2014–2025 |
|---|---:|---:|---:|
| Binary gate | **1.659** | 0.920 | 1.408 |
| Continuous sizing (previous) | 1.341 | **1.062** | 1.213 |
| Trend filter only | 1.120 | 0.836 | 0.987 |
| Buy and hold | 0.685 | 0.525 | 0.591 |

Counting calendar years rather than pooling them, the continuous rule beat the
binary gate it replaced in **2 of 12 years**, mean gap −0.26 Sharpe, t = −2.47 —
and in **0 of the 7 years before the development window**. The two years it won
were 2023 and 2024, both inside the window where it was selected. That is the
signature of a selection effect, not of a better rule.

The diagnosis is narrow and it is not "continuous sizing is wrong". The binary
gate can go **fully flat**; the continuous rule floors out at a small residual
long position and never leaves the market inside an uptrend. Grading the size
is worth something, and being able to reach zero is worth more.

## What changed, and how it was decided

Three changes. Each replaces a constant that was chosen by hand with either a
measurement or an average over the range the constant could have taken, so none
of them is justified by the Sharpe it produces:

1. **The trend filter averages a ladder instead of picking a lookback.** The
   720-bar span was never swept, and it moved the published number more than
   anything else in the pipeline. Sweeping it produces no plateau. The gate is
   now the share of a geometric ladder — 90, 180, 360, 720, 1440, 2160 bars,
   fifteen days to a year — that the close sits above, which grades
   participation and removes the choice rather than making it better.
2. **The signal is smoothed over the label's measured lifetime, not its
   horizon.** A ±1.5 ATR barrier is actually touched after **4.93 bars on
   average** (median 4); the 12-bar horizon is only the ceiling. Each fold now
   measures that number on its own purged calibration window — the same window
   that already supplies the threshold and the log-odds scale — and gets 4.7 to
   5.2 bars, a stable property of the label rather than a fitted quantity.
3. **A weak call sizes to zero.** The size map is a sigmoid of the log-odds
   offset from the fold threshold, divided by that fold's calibration spread.
   Sizes below `sigmoid(-1) = 0.269` — one full calibration spread below the
   threshold — now map to flat, and the remainder is rescaled. The floor is
   stated in the units the map already emits, so it introduces no new constant.

Together they close most of the gap the falsification opened, and reverse the
comparison against the rule they replace:

| Comparison | Years won | Mean gap | t | Pre-development years won |
|---|---:|---:|---:|---:|
| Previous continuous vs binary gate | 2 / 12 | −0.259 | −2.47 | **0 / 7** |
| Current vs binary gate | 5 / 12 | −0.135 | −0.70 | 2 / 7 |
| Current vs previous continuous | **11 / 12** | +0.124 | +0.74 | **7 / 7** |

| Rule | 2014–2021 | 2021–2025 | 2014–2025 |
|---|---:|---:|---:|
| Binary gate | **1.659** | 0.920 | 1.408 |
| Continuous sizing (previous) | 1.341 | 1.062 | 1.213 |
| **Current** | 1.588 | **1.190** | **1.423** |

The current rule is no longer distinguishable from the binary gate on the
history it was tested against (t = −0.70), and beats the rule it replaces in
eleven of twelve years and in all seven pre-development years.

**This is not a clean out-of-sample claim, and the distinction matters.** The
2014–2021 history is genuine out-of-development evidence *against* the previous
rule, because that rule was fixed before the history was looked at. It is not
evidence *for* the current rule: the floor was introduced after seeing that
history, so those years are in-sample with respect to this design step. What
can be claimed is that the current rule is not contradicted where its
predecessor was, which is a weaker and more accurate statement.

## Read every execution number against the baselines

A classifier layered on a trend filter can look good while contributing
nothing, so `walk_forward.json` always emits the do-nothing comparisons over
the identical index and fee model. Development period, 0.1% per side:

| Strategy | Return | Sharpe | Max drawdown | Mean weight | Fee sides |
|---|---:|---:|---:|---:|---:|
| Buy and hold | +213.6% | 0.53 | −77.1% | 1.00 | 2 |
| Trend ladder only, no model | +221.8% | 0.84 | −38.1% | 0.58 | 194 |
| Single 720-bar gate only, no model | +281.3% | 0.89 | −34.0% | 0.58 | 72 |
| Model only, no trend filter | +130.3% | 1.21 | −26.2% | 0.29 | 343 |
| Binary gate variant | +151.6% | 0.91 | −32.7% | 0.28 | 640 |
| **Backtesting example** | **+117.0%** | **1.48** | **−12.3%** | **0.20** | **137** |

The strategy still earns less than simply holding through the trend filter.
What it buys is two thirds off the drawdown and 0.64 of Sharpe over the gate it
sits on, at a fifth of the turnover of the binary gate it replaces.

Turnover roughly doubled against the previous release, because the shorter
smoothing span reacts faster. It buys more than it costs, and the result still
does not depend on the fee assumption:

| Fee per side | Binary gate | Backtesting example | Trend ladder only |
|---|---:|---:|---:|
| 0.00% | +377.3% / 1.54 | +148.8% / 1.75 | +290.7% / 0.98 |
| 0.05% | +246.6% / 1.23 | +132.4% / 1.61 | +254.6% / 0.91 |
| 0.10% | +151.6% / 0.91 | +117.0% / 1.48 | +221.8% / 0.84 |
| 0.20% | +32.6% / 0.28 | +89.3% / 1.22 | +165.0% / 0.70 |
| 0.40% | −63.3% / −0.97 | +44.0% / 0.70 | +79.7% / 0.42 |

The binary gate stops working somewhere below 0.2% per side. The current rule
still beats its own trend filter at 0.4%.

## The trend filter was the largest unswept knob in the pipeline

The gate does most of the work, and until now its lookback had never been
swept, so the *level* of every published execution number rested on it.
`walk_forward.json` now emits the sweep next to the fee table. Single spans,
development period, each against its own trend-only baseline:

| EMA span | Strategy Sharpe | Trend-only Sharpe | Uplift |
|---|---:|---:|---:|
| 90 | 1.212 | 0.860 | +0.35 |
| 180 | 1.485 | 1.029 | +0.46 |
| 270 | **1.609** | 0.827 | +0.78 |
| 360 | 1.431 | 0.643 | +0.79 |
| 540 | 1.489 | 0.864 | +0.63 |
| 720 | 1.422 | 0.894 | +0.53 |
| 900 | 1.227 | 0.742 | +0.49 |
| 1200 | 0.898 | 0.415 | +0.48 |
| 1440 | 1.022 | 0.397 | +0.62 |
| 2160 | 1.195 | 0.592 | +0.60 |
| **Ladder (all six)** | **1.484** | 0.837 | **+0.65** |

Two things to read here. The strategy column has a standard deviation of
**0.221** across single spans and no plateau — that spread is the cost of
choosing one lookback, and it is four times the fold-grid noise floor. The
uplift column has a standard deviation of **0.130** and never goes near zero:
the model's contribution is stable no matter which filter it sits on, while the
level is not. The ladder is deliberately **not** the best row; 270 is. Picking
270 would repeat the mistake that produced 720.

## What did not work

Recorded because the negative results are the more reliable half of this
exercise.

### Execution and sizing

Measured on the full 2014–2025 history against the current rule, so a change
that only wins on the development window shows up as one. The noise floor is
±0.06:

| Change | 2014–2021 | 2021–2025 | 2014–2025 |
|---|---:|---:|---:|
| *(current rule)* | 1.588 | **1.190** | 1.423 |
| Short sleeve on trend-off bars, 0.25× | 1.612 | 1.124 | 1.431 |
| Short sleeve, 0.5× | 1.512 | 0.927 | 1.307 |
| Short sleeve, 1.0× | 1.267 | 0.571 | 1.023 |
| Partial long participation off-trend, 0.15× | 1.627 | 1.169 | 1.445 |
| Partial long participation off-trend, 0.5× | 1.627 | 1.010 | 1.406 |

Bars the gate flattens are not edgeless, which made them look like the one
place genuinely new return could come from. They are not, and the reason is
worth stating because it is the reason both sleeves fail. `long_history.json`
emits the measurement:

| Bars the gate forces flat | Share of bars | Up calls | Down calls |
|---|---:|---:|---:|
| Single 720-bar gate | 40.4% | +6.49 bp | −1.58 bp |
| **Ladder gate (current)** | **18.1%** | +8.86 bp | **+0.25 bp** |

The graded ladder already participates on most of what the old binary gate
threw away — 18% of bars are fully flat rather than 40% — so more than half the
supposed opportunity was collected by the trend-filter change itself, before
any sleeve was added. What is left has no short edge in it at all: the model's
down calls on those bars average **+0.25 bp**, not a negative number. Shorting a
positive expectancy is why every short row degrades monotonically with size.

The long side does have an edge there, but not a harvestable one. Quintiling
the flat bars by the model's own size gives −0.2, +3.0, −6.6, +0.7, +8.5 bp:
only the top fifth pays, the ordering is not monotone, and 8.5 bp against a 20
bp round trip does not clear its own costs. The 0.15× and 0.25× rows are inside
the ±0.06 noise floor — indistinguishable from doing nothing — and there is no
size at which either sleeve pays.

### Model class and label horizon

Screened over the same long history through the identical purge, calibration
and embargo, on a single fold-grid offset, reproducible with
`scripts/screen_alternatives.py`:

| Candidate | Accuracy | 2014–2021 | 2021–2025 | 2014–2025 |
|---|---:|---:|---:|---:|
| *(published logistic)* | 53.85% | 1.588 | **1.193** | 1.429 |
| Bagged logistic, 5 penalties × 3 bootstraps | 53.72% | 1.583 | 1.190 | 1.424 |
| Histogram gradient boosting, depth 3 | 52.73% | 1.634 | 1.121 | 1.440 |
| Label horizon 6 bars | 53.59% | **1.723** | 1.084 | **1.480** |
| Label horizon 24 bars | 53.37% | 1.457 | 1.067 | 1.295 |
| Horizon ladder, 6/12/24 averaged | — | 1.631 | 0.981 | 1.406 |

Bagging is the change that should have worked — variance reduction without
added capacity, the kind that tends to survive out of sample. It moves nothing
in any period, for fourteen times the compute. Gradient boosting buys
interactions at the cost of 1.1 points of accuracy, and its +0.011 on the full
history is a fifth of the noise floor.

The horizon ladder is the same trick that fixed the trend filter, applied to
the equally arbitrary 12-bar label horizon, and it does not transfer: averaging
lands between the single horizons rather than above them. Unlike the EMA spans,
the horizons are not disagreeing about one question — they define different
questions, and the average of three targets is not a better target.

The tempting row is the 6-bar horizon, which posts the best full-history Sharpe
in the table. It gets there by giving up 0.11 in the development window and
0.94 in the later half-year, and the only reason to prefer it is that one
column. Choosing it on that basis is exactly the error that produced the
720-bar trend filter, so it stays a recorded observation rather than a change.

### Model and feature changes

Measured earlier, under the **previous** execution rule, against its baseline
of 53.15% accuracy and 1.224 Sharpe with a noise floor of ±0.056. The accuracy
column is unaffected by the execution change; the Sharpe column is not
comparable to the tables above.

| Change | Accuracy | Sharpe |
|---|---:|---:|
| *(baseline)* | 53.15% | 1.224 |
| Uniqueness weighting, `∝ 1/concurrency` | 52.34% | 1.129 |
| Weighting by realized move size | 51.41% | 1.067 |
| Isotonic calibration of the probability | — | 1.096 |
| Platt calibration of the probability | — | 1.034 |
| Inverse-volatility scaling from `raw_atr` | — | 1.186 |
| Keeping SIDEWAYS rows in training | 52.70% | 1.142 |
| Smoothing the threshold grid before the argmax | 53.05% | 1.209 |
| Rolling 5-year training instead of anchored | 53.09% | 1.218 |
| Add perpetual basis (3 features) | 52.92% | 1.247 |
| Add volume and realized-volatility state (5) | 52.89% | 1.146 |
| Add time-of-day and day-of-week (4) | 53.08% | 1.201 |
| Add all twelve new features | 52.54% | 1.086 |

Nothing here earns its place. The two López de Prado sample weightings both
lower accuracy; calibration destroys the fine ranking the sizing map depends
on; and of the new features only the perpetual basis moves Sharpe at all, by
less than half the noise floor while costing accuracy.

## Feature ablation

The six `mature_outcome` features are the label itself, lagged 13 bars and
smoothed, which invites the question of whether they are carrying the result.
They are not, but they are not redundant either:

| Feature block | Accuracy | Sharpe |
|---|---:|---:|
| All 35 | 53.15% | 1.224 |
| Momentum + order flow (29) | 51.68% | 1.110 |
| Momentum only (17) | 52.34% | 1.043 |
| Matured outcomes only (6) | **48.64%** | 0.710 |

Alone the matured outcomes are worse than a coin flip, yet removing them costs
1.5 points of accuracy. All three blocks contribute. Sharpe here is on the
previous execution rule, as above.

## Current model

- **Target:** BTC/USD four-hour bars, 12-bar horizon, symmetric ±1.5 ATR
  barriers.
- **Features:** 17 momentum features, 12 Binance spot/perpetual order-flow
  features, and 6 matured-outcome state features lagged 13 bars.
- **Classifier:** balanced logistic regression (`C=0.5`).
- **Calibration:** the decision threshold, the log-odds scale, and the signal
  smoothing span all come from the preceding purged 12-month period before each
  three-month test window.
- **Evaluation:** anchored walk-forward training, a full label-horizon purge,
  final embargo, and a moving-block bootstrap.

## Backtesting example

The execution example converts the classifier calls into a long-only position:

1. Each call is mapped to a continuous size in `[0, 1]`, neutral at the
   calibrated threshold, scaled by that fold's calibration log-odds spread.
2. The size is EWMA-smoothed over the fold's own measured barrier lifetime
   (≈4.9 bars), not over the 12-bar horizon ceiling.
3. Sizes within one calibration spread below the threshold map to flat.
4. The position is scaled by the share of the six-span EMA ladder that BTC
   closes above.
5. The position is held until the target drifts by 0.10.
6. The backtest charges 0.1% commission for every side actually traded.

| Period | Return | Sharpe | Maximum drawdown | Mean weight |
|---|---:|---:|---:|---:|
| Jul 2021–Jun 2025 development | **+117.00%** | **1.48** | **−12.28%** | 0.20 |
| Jul–Dec 2025 out of development | **+2.60%** | **0.75** | **−3.92%** | 0.13 |

The half-year outside the development period no longer disagrees with it. In it
BTC fell 17.8%, the trend filter lost 8.6%, the binary gate returned +4.87% at
Sharpe 0.64 by sitting flat, and the current rule returned +2.60% at Sharpe
0.75 — less money, better risk-adjusted, and a −3.9% drawdown against the
gate's −6.8%. Directional
accuracy over the window was **58.83%** (563 / 957; block-bootstrap 95%
interval 54.55–64.79%). This is one half-year and one observation; it is
reported because it is the only data outside the development period, and it
carries far less weight than the eleven-year replication above.

The deflated Sharpe ratio, using 320 screened configurations and the observed
variance across the execution sensitivity sweep, is **0.98**. That variance
estimate moves one knob at a time and so understates the true trial variance,
which makes the figure optimistic.

Machine-readable results:

- [`results/walk_forward.json`](results/walk_forward.json) — development
  walk-forward, baselines, fee and knob sensitivity, trend-span sweep.
- [`results/long_history.json`](results/long_history.json) — the 2014–2025
  replication and the per-year rule comparison.
- [`results/alternatives.json`](results/alternatives.json) — the model-class
  and label-horizon negatives.
- [`results/later_evaluation_2025h2.json`](results/later_evaluation_2025h2.json)
- [`results/later_evaluation.json`](results/later_evaluation.json) — the full
  Jul 2025–Jun 2026 window, produced with an extended data directory and the
  **previous** binary-gate execution rule. Regenerate it with the command below
  to bring it onto the current rule.

## Reproduce

Python 3.12 or 3.13 is required. The frozen 2021–2025 inputs are checked in.

```bash
uv sync
uv run pytest -q
uv run python -m alphabarrier.strategy
```

The replication over the history before the development window, and the
alternative-model screen:

```bash
uv run python -m alphabarrier.replication
uv run python -m scripts.screen_alternatives
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
alphabarrier/      direction model, evaluation, backtesting example, replication
data/              frozen processed inputs
results/           development, replication, and later evaluation results
scripts/           data preparation, later evaluation, alternative-model screen
tests/             feature, evaluation, execution, and replication tests
```
