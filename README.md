# Half-Time Draw Predictor — V4

Predicts whether a football match will be level (HT draw: HTHG == HTAG) at half-time. Uses a two-model architecture that separates market signal from fundamental statistics signal to identify mispriced bets.

## The Key Finding

**Inverted Edge** — when a fundamentals-only model predicts *fewer* HT draws than the bookmaker's odds imply, the actual HT draw rate historically *exceeds* the market estimate.

This counter-intuitive signal (+6.4% ROI at ≥3% edge in backtesting, 99%+ bootstrap profitable) is the core of V4.

## Architecture

```
MODEL A (Market)       — LogisticRegression on 3 B365 log-odds features
                         Essentially mirrors what B365 implies about P(HT draw)

MODEL B (Fundamentals) — XGBoost on 42 non-odds features:
                         39 rolling match stats (shots, fouls, corners, cards...)
                         + Dixon-Coles Poisson draw probability
                         + Elo rating system draw probability
                         + Referee draw-rate adjustment

INVERTED EDGE = Model A − Model B
  Positive = market prices HT draw HIGHER than fundamentals say
  → actual HT draws historically EXCEED the market prediction → value bet
```

### Why the inversion works

Model B's AUC (0.535–0.540) is below the market (0.557). This means Model B's raw draw predictions are *less accurate* than the market. But when Model B disagrees with the market in a specific direction — predicting fewer draws — the market is reliably over-correcting. The result: actual draw rates in those cases *beat* what the market priced.

## Performance

| Model | AUC | vs Market |
|-------|-----|-----------|
| Market (B365 normalized) | 0.5566 | — |
| Model A (market LR) | ~0.557 | ~+0.000 |
| Model B (fundamentals XGB) | ~0.535 | −0.021 |

### Backtest results (test period ~Apr 2022 – Feb 2026, using B365D as FT-draw proxy)

| Inverted edge threshold | Bets | Win% | Flat ROI | vs Baseline |
|------------------------|------|------|----------|-------------|
| > 1% | ~8,700 | ~44.5% | +0.44 | +0.00 |
| > 3% | ~5,500 | ~46.0% | +0.64 | +0.14 |
| > 5% | ~2,350 | ~47.0% | +0.70 | +0.21 |

Bootstrap (1,000 resamples, edge > 3%): **≥99% of resamples profitable**, 95% CI [+1.6%, +11.3%]

> ⚠️ **Caveat**: B365D = full-time draw odds. HT draws occur at ~42%, FT draws at ~28%. The backtest uses FT odds as a proxy — the ROI pattern is directionally valid but the absolute figures are inflated. Real HT draw markets (Betfair, Asian books) would show lower but more accurate returns.

## Data

- **203,995 matches** from 2003–2026
- 22 European leagues (Championship, Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga, and more)
- MLS 2012–2025 (5,816 matches — FT only, DC/Elo fitted using FT score proxy)
- Source: [football-data.co.uk](https://www.football-data.co.uk/)
- Dataset: `data/processed/mega_dataset_v2.parquet`

## Setup

```bash
git clone <repo>
cd half-time
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Rebuild all models from scratch (~15–30 minutes for Optuna tuning):

```bash
source .venv/bin/activate
python src/train_v4.py
```

Saves everything to `models/v4/`.

## Predicting a Match

```bash
# --odds H/D/A  (convenience flag)
python src/predict_v4.py 'Everton' 'Man United' --odds 3.70/3.75/1.91

# With league and referee
python src/predict_v4.py 'Leeds United' 'Sheffield Utd' \
    --odds 2.10/3.40/3.50 --league E1 --referee 'T Kettle'

# MLS match
python src/predict_v4.py 'LA Galaxy' 'NYCFC' \
    --odds 2.25/3.60/2.80 --league USA_MLS

# Legacy individual flag format
python src/predict_v4.py 'Alaves' 'Girona' \
    --b365h 2.30 --b365d 3.00 --b365a 3.50 --league SP1
```

### Sample output

```
════════════════════════════════════════════════════════════════
  Everton vs Man United  [E0]
════════════════════════════════════════════════════════════════
  ODDS: Home 3.70  |  Draw 3.75  |  Away 1.91
  Market implied P(HT draw): 38.9%

  ┌─ MODEL A — Market Estimate ───────────────────────┐
  │  P(HT draw) = 39.8%  (logistic on B365 log-odds)    │
  └────────────────────────────────────────────────────┘

  ┌─ MODEL B — Fundamentals Estimate ─────────────────┐
  │  P(HT draw) = 38.5%  (team stats only — NO odds)    │
  │  Components:                                       │
  │    Dixon-Coles:  41.2%    Elo ratings:  43.1%      │
  │    Referee adj: ×1.000                             │
  └────────────────────────────────────────────────────┘

  INVERTED EDGE (A − B): +1.39%  ← MARGINAL ★
  BET RATING: MARGINAL ★

  Kelly staking (draw @ 3.75, calibrated hit rate 44.0%):
    Quarter Kelly:  0.82% of bankroll  ← recommended
════════════════════════════════════════════════════════════════
```

## Scanning Upcoming Fixtures

```bash
# Fetch live from football-data.co.uk (default)
python src/scan_v4.py

# Demo mode (no network needed)
python src/scan_v4.py --demo

# From a CSV file
python src/scan_v4.py --fixtures my_fixtures.csv
```

## Tracking Bets

```bash
# First-time setup: backfill the Galaxy/NYCFC opening bet
python src/tracker.py backfill-galaxy

# Add a new bet
python src/tracker.py add \
    --home "Everton" --away "Man United" \
    --league E0 --odds 3.75 --stake 0.5 \
    --model-a 0.398 --model-b 0.385 --edge 0.0139 --rating MARGINAL

# Record result
python src/tracker.py result --id 1 --outcome win

# List bets / stats
python src/tracker.py list
python src/tracker.py stats
```

## Kelly Sizing

V4 uses backtest-calibrated hit rates per edge bucket:

| Rating | Edge | Hit rate | Full Kelly (@ 3.5 odds) |
|--------|------|----------|--------------------------|
| STRONG VALUE | ≥ 5% | 47.0% | ~8.4% |
| VALUE | 3–5% | 46.0% | ~6.3% |
| MARGINAL | 1–3% | 44.0% | ~2.5% |

Recommended: **Quarter Kelly** for real-money bets.

## Project Structure

```
half-time/
├── src/
│   ├── train_v4.py          ← Training pipeline (canonical)
│   ├── predict_v4.py        ← Single-match prediction CLI
│   ├── scan_v4.py           ← Upcoming fixture scanner
│   ├── tracker.py           ← Bet tracking system
│   ├── dixon_coles.py       ← Dixon-Coles Poisson model
│   ├── elo.py               ← Elo rating system
│   ├── referee_model.py     ← Referee draw-rate profiles
│   ├── features.py          ← Rolling feature engineering
│   ├── utils.py             ← Team name matching utilities
│   ├── build_mega_dataset.py ← Dataset builder
│   └── archive/             ← Old V1/V2/V3 scripts (preserved)
├── models/
│   └── v4/                  ← Canonical model directory
│       ├── v4_paths.json    ← Master paths manifest
│       ├── v4_metrics.json  ← Evaluation metrics + backtest
│       └── plots/
├── data/
│   ├── processed/mega_dataset_v2.parquet  ← 203k matches
│   └── bets.json                          ← Bet tracker
└── README.md
```

## Sub-models

**Dixon-Coles**: Per-league bivariate Poisson on HT scores. MLS uses FT scores ×0.45 to approximate HT rates. Time-decay weighting (ξ=0.003).

**Elo**: Rolling team strength updates after each HT result. K=16, home advantage=50 points. P(draw) via logistic regression on Elo differential.

**Referee Model**: Per-referee Bayesian draw-rate adjustment. Only applied when referee provided and effect is statistically significant (chi-squared p < 0.10).

## Limitations

1. **B365D proxy**: Backtest uses full-time draw odds. Real HT draw markets would give different absolute ROI.
2. **MLS expansion teams**: Post-2018 teams (Austin FC, Inter Miami, etc.) use global draw rate in DC; Elo handles them via extension.
3. **No real-time form**: Form stats come from the last recorded match in the historical dataset.
4. **Distribution shift**: Tactical styles evolve; models may underperform on very recent data.
