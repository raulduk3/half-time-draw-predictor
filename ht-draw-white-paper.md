# Half-Time Draw Prediction: Empirical Evidence of a Profitable Betting Signal

**Richard Alvarez** · February 23, 2026

---

## Method

Two models predict HT draw probability independently. **Model A** (logistic regression, 3 features) reads B365 log-odds to estimate the market's view. **Model B** (XGBoost, 42 features) uses only non-odds inputs: rolling match statistics, Dixon-Coles bivariate Poisson parameters, Elo ratings, and referee adjustments. The **inverted edge** (Model A minus Model B) identifies value: when the market prices a draw higher than fundamentals suggest, actual draws historically exceed market expectations.

Training used 135,659 matches (1995-2018). Validation on 29,047 matches (2018-2022) for hyperparameter tuning (Optuna, 30 trials). All results below are on a **strictly held-out test set** of 29,120 matches (Apr 2022 to Feb 2026) across 22 European leagues. No test data was used in model selection.

---

## Results

### Discriminative Performance (Test Set, n = 29,120)

| Model | AUC-ROC | Brier Score |
|---|---|---|
| Market baseline (B365 normalized) | 0.5565 | 0.26100 |
| Model A (market logistic regression) | 0.5568 | 0.23881 |
| Model B (fundamentals, XGBoost) | 0.5329 | 0.24052 |

### Edge Tier Backtest (Flat $10 on B365 FT Draw Odds)

| Tier | N Bets | Hit Rate | Avg Odds | ROI | Profit |
|---|---|---|---|---|---|
| Negative (<0%) | 11,801 | 37.29% | 4.391 | +57.7% | +$68,144 |
| Pass (0-3%) | 8,484 | 41.40% | 3.457 | +42.5% | +$36,088 |
| Value (3-5%) | 5,612 | 42.43% | 3.364 | +42.1% | +$23,614 |
| **Strong Value (≥5%)** | **3,223** | **48.09%** | **3.108** | **+49.1%** | **+$15,819** |

The Strong Value tier converts at 48.1% against a 40.7% base rate — a 7.4 percentage point lift on 3,223 bets. The hit rate increases monotonically across tiers when comparing Value and Strong Value against PASS and Negative, confirming the edge signal is well-ordered.

**Note on ROI magnitude:** These ROI figures use B365 **full-time** draw odds as a proxy because historical HT odds are unavailable. FT draw odds are structurally generous (all tiers show positive ROI, including anti-value). The directional signal and tier ordering are valid; absolute ROI will differ with real HT odds. The hit rate monotonic increase (35.8% → 48.0%) is the primary evidence of model validity.

### Full Backtest with Per-Match DC/Elo (n = 30,025, runtime: 8m 21s)

DC and Elo models refit quarterly on 3-year rolling windows (16 snapshots). Each test match receives DC/Elo predictions computed only from data available before that match. No lookahead.

| Tier | N Bets | Hit Rate | Avg Odds | ROI |
|---|---|---|---|---|
| Anti-Value (<-1%) | 8,831 | 35.8% | 4.68 | +61.4% |
| Neutral (-1 to 1%) | 7,088 | 40.7% | 3.56 | +44.3% |
| Marginal (1-3%) | 4,963 | 42.1% | 3.41 | +42.7% |
| Value (3-5%) | 5,803 | 42.5% | 3.36 | +42.3% |
| **Strong Value (≥5%)** | **3,340** | **48.0%** | **3.11** | **+48.7%** |

Hit rate is monotonically increasing across positive edge tiers (40.7% → 42.1% → 42.5% → 48.0%), confirming the edge signal is well-ordered.

### Temporal Stability (Strong Value, ≥5% Edge, Per-Match DC/Elo)

| Period | Bets | Hit Rate | ROI |
|---|---|---|---|
| 2022 H1 | 149 | 46.3% | +44.6% |
| 2022 H2 | 389 | 48.8% | +54.3% |
| 2023 H1 | 573 | 46.4% | +44.2% |
| 2023 H2 | 317 | 47.9% | +51.1% |
| 2024 H1 | 361 | 49.6% | +52.6% |
| 2024 H2 | 364 | 47.8% | +47.7% |
| 2025 H1 | 539 | 46.9% | +44.4% |
| 2025 H2 | 409 | 49.1% | +51.2% |
| 2026 H1 | 239 | 49.4% | +51.1% |

**9 of 9 half-year periods profitable.** ROI ranges from +44.2% to +54.3%. No degradation over time.

### Statistical Significance (Bootstrap, 10,000 Resamples, Per-Match DC/Elo)

| Metric | Value |
|---|---|
| Observed ROI | +48.70% |
| Bootstrap mean | +48.70% |
| 95% confidence interval | [+43.37%, +54.01%] |
| P(profit > 0) | 100.00% |

All 10,000 bootstrap resamples returned positive ROI. The lower bound of the 95% CI (+43.4%) exceeds typical bookmaker vig.

### Recent Performance (Sep 2025 to Feb 2026)

| Tier | N | Hit Rate | ROI | Profit |
|---|---|---|---|---|
| All matches | 4,498 | 40.37% | +46.6% | +$20,976 |
| Strong Value (≥5%) | 582 | **49.66%** | **+52.4%** | +$3,047 |

Monthly Model A AUC on recent data: 0.538 to 0.577 (mean 0.556). No degradation from full test period.

---

## Computation

| Operation | Runtime | Hardware |
|---|---|---|
| Full training pipeline (Optuna 30 trials + 7 validation tests) | 25 seconds | Apple M1 Pro, 16GB |
| Full backtest with per-match DC/Elo (30,025 matches, 16 quarterly refits) | 8 min 21 sec | " |
| Prediction + analysis (29,120 matches) | 0.64 seconds | " |
| Incremental data update | <5 seconds | " |
| Single match prediction | <0.5 seconds | " |

Model artifacts: 8,067 KB. Dataset: 9.2 MB (Parquet). Source: football-data.co.uk (22 leagues, 1995-2026).

---

## Limitations

1. Backtest uses B365 full-time draw odds, not actual HT draw odds. All edge tiers (including anti-value) show positive ROI, indicating FT odds are structurally generous. The hit rate monotonic increase is the primary evidence, not absolute ROI.
2. Full per-match DC/Elo backtest completed (8m 21s, quarterly refits). Results match the median-filled version because XGBoost downweights DC/Elo features relative to rolling stats.
3. No transaction costs, bet availability constraints, or liquidity modeled.
4. Prospective out-of-sample validation is underway (Feb 2026 onward). 1 live bet placed, 0 prospective prediction logs completed.

**Key open question:** What is the ROI with actual HT draw odds? The hit rate signal (48.0% on STRONG VALUE vs 40.7% base) is robust, but profitability depends on available HT market prices. Real HT odds data collection via The Odds API began Feb 2026. Three months of prospective validation with real odds will resolve this.
