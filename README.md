# Machine Learning Education Project: Predicting Half-Time Draws in the English Championship

## Overview
This project walks through a complete, end-to-end example of machine learning in sports analytics:  
**predicting whether a soccer match will be a draw at half-time** in England’s second division, the **EFL Championship**.

It is designed for two curious learners exploring how data, modeling, and reasoning intersect in AI systems.  
The goal is educational clarity and reproducibility—not production infrastructure.

The workflow is minimal and organized into **two Jupyter notebooks**, supported by a small data ingestion layer.

---

## Learning Goals
1. Acquire and prepare real-world sports data.  
2. Create temporal and contextual features.  
3. Train and evaluate both a logistic baseline and an LSTM sequence model.  
4. Interpret probabilities, calibration, and generalization.  
5. Build a reusable structure for future sports ML projects.

---

## Why the English Championship
- **Data-rich:** historical records with half-time and full-time scores since the early 2000s.  
- **High volume:** 552 matches per season, 24 clubs.  
- **Competitive balance:** frequent draws create strong modeling signal.  
- **Availability:** public CSVs from [Kaggle](https://www.kaggle.com/) and [football-data.co.uk](https://www.football-data.co.uk/).

---

## Project Philosophy
Simplicity, transparency, and reproducibility.  
Work locally. Keep everything versioned and small.  
Favor understanding over optimization.  
Each artifact—a dataset, plot, or model—should clearly demonstrate one ML concept.

---

## Directory Layout
````
soccer-htd/
data/
raw/          # raw CSVs from Kaggle or football-data.co.uk
processed/    # final cleaned dataset
models/
notebooks/
01_build_dataset.ipynb
02_train_eval.ipynb
src/
features.py
utils.py
env.yaml
README.md
````
---

## Step 1 — Collect Raw Data
Use **one clean, consistent source** such as the Kaggle dataset for the EFL Championship (a mirror of football-data.co.uk).  
This provides every field needed for our initial model.

Required columns:  
`Date, HomeTeam, AwayTeam, HTHG, HTAG, FTHG, FTAG, B365H, B365D, B365A`

All subsequent features are computed from these raw fields—no APIs required.

---

## Step 2 — Build the Dataset
In **`01_build_dataset.ipynb`**:
1. Load the raw Kaggle CSVs.  
2. Normalize column names and parse dates.  
3. Label target: `y_ht_draw = 1 if HTHG == HTAG else 0`.  
4. Compute rolling form for each team (last five matches, first-half goals).  
5. Add odds and rest-day features.  
6. Save the processed dataset to `data/processed/dataset.parquet`.

---

## Step 3 — Train and Evaluate Models
In **`02_train_eval.ipynb`**:
1. Split matches chronologically into train, validation, and test sets.  
2. Train a **logistic regression** baseline (interpretable, calibrated).  
3. Train a compact **LSTM** using per-team sequences of recent matches.  
4. Compare ROC-AUC, Brier score, and calibration reliability.  
5. Save trained weights and a `metadata.json` describing feature schema.

---

## Step 4 — Reflect and Iterate
- Validate that no features leak future information.  
- Inspect calibration over different seasons.  
- Experiment with window size (3, 5, 10 matches).  
- Add new features incrementally (xG, weather, possession).

---

## Core Feature Schema (v1.5 MVP)
All computed from the raw Kaggle dataset.

| Category | Feature | Description |
|-----------|----------|-------------|
| **Form (rolling mean of last 5 matches)** | `home_gf_r5`, `home_ga_r5`, `home_gd_r5`, `away_gf_r5`, `away_ga_r5`, `away_gd_r5` | First-half scoring form and defensive strength |
| **Odds (log-transformed)** | `log_home_win_odds`, `log_draw_odds`, `log_away_win_odds` | Market expectations |
| **Rest / Recency** | `home_days_since_last`, `away_days_since_last` | Fatigue rhythm |
| **Context** | `month` | Seasonality indicator |
| **Target** | `y_ht_draw` | 1 if tied at half-time |

→ **12 total features**, all derivable from CSV data.

---

## Inference Workflow
To score an upcoming match:
1. Gather each team’s **last 5 completed matches** from the same dataset.  
2. Recompute all 12 features exactly as during training.  
3. Feed the resulting row into the trained model.  
4. The output is a single scalar `p_ht_draw` — the predicted probability of a half-time draw.  

Example output:
```json
{
  "date": "2026-02-14",
  "home_team": "Leeds United",
  "away_team": "Norwich City",
  "p_ht_draw": 0.34
}
````

At minimum, inference requires only the latest five fixtures per team and the current bookmaker odds.

---

## Educational Takeaways

* **Data realism:** most effort lies in cleaning and feature computation.
* **Temporal awareness:** train on past → predict the future.
* **Calibration over accuracy:** trustworthiness matters more than raw score.
* **Reproducibility:** small, deterministic notebooks outperform ad-hoc scripts.

---

## Environment

Dependencies:

* Python ≥ 3.11
* pandas, numpy, matplotlib, seaborn
* scikit-learn
* PyTorch (CPU)
* pyarrow, requests, ipykernel

Setup:
````bash
conda env create -f env.yaml
conda activate soccer-htd
````

---

## Recommended Workflow

1. Download the Kaggle EFL Championship dataset.
2. Run **`01_build_dataset.ipynb`** to create `dataset.parquet`.
3. Run **`02_train_eval.ipynb`** to train and evaluate models.
4. Generate predictions for new fixtures.
5. Iterate and expand features as desired.

---

### Epilogue

This project demonstrates how data becomes insight through simple, transparent modeling.
Predicting half-time draws offers a compact, realistic microcosm of modern machine learning: data engineering, experimentation, validation, and reflection.

---

## Phase 3 — Mega Dataset Results (2026-02-22)

Scaling from one league (EFL Championship, ~10k matches, 12 features) to a **22-league European dataset** (193k matches, 42 features, 1995–2026) with a full model suite.

### Dataset

| Property | Value |
|---|---|
| Source | `data/processed/mega_dataset.parquet` |
| Leagues | 22 (England, Spain, France, Germany, Italy, Scotland, Netherlands, Turkey, Belgium, Portugal, Greece) |
| Matches | 193,637 |
| Date range | 1995-07-19 → 2026-02-18 |
| HT draw rate | **42.3%** (vs ~28% FT draw rate) |
| Features used | 42 (all had ≥60% coverage; none dropped) |

**Temporal split:** 70/15/15 chronological
- Train: 135,545 matches (through Aug 2018)
- Val: 29,045 matches (Aug 2018 → Apr 2022)
- Test: 29,047 matches (Apr 2022 → Feb 2026)

### Model Results

| Model | Train AUC | Val AUC | Test AUC | Test Brier | vs Market |
|---|---|---|---|---|---|
| OLD: LR (12-feat, EFL only) | — | — | 0.5252 | 0.2415 | −0.031 |
| OLD: LSTM seq1 (12-feat, EFL) | — | — | 0.5007 | 0.2436 | −0.056 |
| **NEW: LR (mega, 42 feat)** | 0.5374 | 0.5462 | **0.5561** | 0.2390 | −0.000 |
| NEW: XGBoost Optuna (mega) | 0.5588 | 0.5494 | 0.5534 | 0.2394 | −0.003 |
| NEW: LightGBM (mega) | 0.5917 | 0.5476 | 0.5479 | 0.2398 | −0.009 |
| NEW: LGB + Platt calibration | — | — | 0.5479 | 0.2396 | −0.009 |
| **MARKET: B365 implied P(draw)** | — | — | **0.5565** | 0.2610 | — |

**Key finding:** The mega Logistic Regression (AUC 0.5561) is essentially market-equivalent, just −0.0004 behind B365. No model beats the market by any meaningful margin. Scaling to 22 leagues and 42 features closed the gap dramatically vs the original 12-feature EFL model (−0.031 below market), but the market remains the ceiling.

### Top Features (SHAP, LightGBM)

The model is almost entirely driven by market signals:

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `log_draw_odds` | 0.0575 |
| 2 | `league_ht_draw_rate_historical` | 0.0125 |
| 3 | `log_home_win_odds` | 0.0113 |
| 4 | `league_encoded` | 0.0108 |
| 5 | `log_away_win_odds` | 0.0083 |
| 6 | `home_hst_r5` | 0.0080 |
| 7 | `home_hs_r5` | 0.0072 |

The three log-odds features plus league-level draw rate dominate by a large margin. Team form features contribute marginally.

### Backtest Caveat (important)

The backtest shows ~+49% ROI across all edge thresholds — **this is an artifact, not real edge.**

The model predicts HT draw probability (~42% base rate). The "market implied probability" is derived from B365D (bookmaker full-time draw odds, pricing a ~28% event). Since 42% >> 28%, the model appears to find large "edge" on almost every match (99%+ bet rate at all thresholds). This is a pricing mismatch, not genuine alpha.

To run a valid backtest, you would need odds specifically priced on the half-time draw market (e.g., Betfair in-play pre-kickoff HT draw lines). The B365D column cannot be used to simulate HT draw betting.

### Educational Takeaways from Phase 3

- **Market efficiency holds at scale:** doubling the dataset size and feature set got us from −0.031 to −0.0004 vs market — essentially breaking even — but no model beats B365.
- **Logistic regression benefits most from scale:** LR improved by +0.031 AUC when given 22× more data. Tree models overfit more readily.
- **Odds features dominate:** log(draw_odds) alone explains far more variance than all 39 rolling-form features combined.
- **LSTM is not worth the compute cost:** sequence building on 193k rows is O(n²) and takes hours. The small-dataset phase already showed LSTM AUC ≤ baseline; the mega phase confirms this.
- **Leakage is subtle:** always verify that rolling features are computed from strictly pre-match data. An audit script (`src/audit_data.py`) was written to spot-check this.

### Outputs

```
models/mega/
  metrics_mega.json       # full results
  lr_model.pkl            # logistic regression
  xgb_model.json          # XGBoost (Optuna-tuned)
  lgbm_model.txt          # LightGBM
  scaler.pkl              # StandardScaler (fit on train)
  medians.json            # imputation medians (fit on train)
  lgb_calibrator.pkl      # Platt calibration model for LGB
  xgb_calibrator.pkl      # Platt calibration model for XGB
  plots/
    roc_curves.png
    shap_importance.png
    calibration.png
    backtest.png
```

---

## Phase 4 — V2 Ensemble: Dixon-Coles, Elo, Referee, Multi-Book Market

Extends Phase 3 with five new signal sources and a stacking meta-learner.

### Architecture

```
Raw Data Signals                Learned Sub-models
─────────────────               ───────────────────
HTHG/HTAG (HT scores)  ──▶  Dixon-Coles bivariate Poisson  ──▶ P(HT draw) per match
HT win/draw/loss       ──▶  Elo Rating System               ──▶ P(HT draw) per match
Referee column         ──▶  Referee Impact Model             ──▶ draw rate multiplier
Multi-book odds        ──▶  Market Consensus (8+ books)      ──▶ consensus P(draw)
XGBoost / LightGBM     ──▶  Pre-trained mega models         ──▶ P(HT draw) per match
                                        │
                                        ▼
                           Stacking Meta-Learner (LR)     ← trained on VAL set only
                                        │
                                        ▼
                           Isotonic Regression Calibration
                                        │
                                        ▼
                             Calibrated P(HT draw) + 90% CI
```

### Sub-models

**1. Dixon-Coles (`src/dixon_coles.py`)**
- Bivariate Poisson model fitted on **half-time scores** (HTHG/HTAG)
- Per-league parameter sets: attack[team], defense[team], home_advantage, rho
- `rho` correction amplifies 0-0 and 1-1 scoreline probabilities (low-score excess)
- Time-decay weighting: `weight = exp(−ξ × days_ago)`, ξ=0.003 (≈231-day half-life)
- Optimised with L-BFGS-B (scipy.optimize.minimize), identifiability via soft constraint
- P(HT draw) = Σ P(HTHG=k, HTAG=k) for k=0..4
- Coverage: all 22 leagues, fitted independently

**2. Elo Rating System (`src/elo.py`)**
- All teams initialised at 1500, updated after every match in chronological order
- Updates based on **HT result**: draw=0.5, home win=1.0, away win=0.0
- K=16 (tuned via grid search on English leagues validation set)
- Home advantage: 50 effective Elo points added to home expected score
- P(HT draw) mapped via logistic regression on (elo_diff, |elo_diff|, elo_diff², E_home)

**3. Referee Impact Model (`src/referee_model.py`)**
- Parses `Referee` column from all EFL Championship raw CSVs
- Per-referee stats: HT draw rate, avg HT goals, avg cards/fouls
- Statistical significance: chi-squared test (draw rate vs baseline)
- Bayesian smoothing: `smoothed_rate = (ref_draws + α×base_rate) / (n + α)`
- Adjustment factor capped at [0.6, 1.7] — only applied if p < 0.10
- Significant referees found (train set):
  - **T Kettle**: draw rate 59.7% vs 42.9% baseline → adj=1.316 (p=0.007)
  - **I Williamson**: draw rate 30.9% → adj=0.781 (p=0.071)

**4. Multi-Bookmaker Market Model (`src/market_model.py`)**
- Parses 8 bookmakers: B365, BW, IW, LB, WH, VC, PS (Pinnacle), SJ
- Opening odds (PSH/D/A) + closing odds (PSCH/D/CA) for line movement
- Consensus features: mean, max, std, range of normalized implied P(draw)
- Line movement: Pinnacle closing vs opening draw probability
- Inter-book disagreement (std across individual books) = uncertainty signal

**5. Stacking Ensemble (`src/ensemble_predictor.py`)**
- Meta-learner: Logistic Regression trained on **validation set predictions only** (no leakage)
- Calibration: Isotonic Regression applied to meta-learner raw output
- Bootstrap confidence intervals via signal perturbation (200 samples)
- Key drivers: ranked list of factors pushing toward/away from draw

### V2 Test Set Results

| Model | Test AUC | Test Brier | vs Market |
|---|---|---|---|
| Dixon-Coles (HT Poisson) | 0.5144 | 0.2478 | −0.0421 |
| Elo (K=16, HT updates) | 0.5236 | 0.2411 | −0.0329 |
| XGBoost (mega, 42 feat) | 0.5534 | 0.2392 | −0.0031 |
| LightGBM (mega, 42 feat) | 0.5479 | 0.2396 | −0.0086 |
| Market consensus (multi-book) | 0.5565 | 0.2610 | baseline |
| **Ensemble V2 (stacked)** | **0.5531** | **0.2392** | **−0.0034** |

**Key finding:** The ensemble achieves better Brier score (0.2392 vs 0.2610) than the market, meaning its **calibration** is substantially better — but its ranking (AUC) remains just below the market consensus signal. This makes it useful for calibrated probability estimates, even if it doesn't outrank the sharp bookmaker consensus in AUC terms.

### Meta-Learner Signal Weights (Logistic Regression coefficients)
The meta-learner assigns highest weight to XGBoost, then market consensus, then LightGBM, then Elo, then Dixon-Coles. The referee adjustment is additive.

### Interactive Tools

**Single-match prediction (`src/predict_match.py`)**
```bash
python src/predict_match.py "Leeds United" "Sheffield Utd" \
    --league E1 --referee "M Dean" \
    --b365h 2.30 --b365d 3.30 --b365a 3.20 \
    --psh 2.35 --psd 3.35 --psa 3.15

# Output:
#   P(HT Draw)  =  43.0%
#   90% CI      = 42.0% – 46.9%
#   Signal breakdown with progress bars
#   Key drivers ranked by impact
#   Recent form (last 5 matches per team)
```

**Live scanner (`src/live_scanner.py`)**
```bash
# Demo mode (no API key needed)
python src/live_scanner.py --demo

# Live mode (requires The Odds API key)
ODDS_API_KEY=<your_key> python src/live_scanner.py

# Scan specific date
python src/live_scanner.py --date 2025-12-26
```

The scanner:
1. Fetches fixtures from the-odds-api.com (free tier, 500 req/month) or uses historical demo fixtures
2. Runs the full ensemble on each match
3. Fetches weather from wttr.in for the home team city
4. Computes edge = model P(draw) − market implied P(draw)
5. Flags matches where |edge| ≥ 3%
6. Outputs ranked table with EV and quarter-Kelly stake recommendations

> **Important caveat on "edge":** B365D is priced on *full-time* draws (~28% base rate). Our model predicts *half-time* draws (~42% base rate). The large apparent edges in the scanner output reflect this pricing mismatch, not genuine alpha. To validate real edge you need HT-specific odds from Betfair or specialist bookmakers.

### Training the V2 Ensemble
```bash
source .venv/bin/activate
python src/train_v2.py
```
Runtime: ~5–10 minutes on Apple Silicon. Saves all models to `models/v2/`.

### V2 Outputs
```
models/v2/
  dixon_coles.pkl           # DixonColesEnsemble (22 leagues)
  elo.pkl                   # EloRatingSystem (K=16, all teams)
  referee_model.pkl         # RefereeModel (87 EFL referees)
  referee_model_profiles.json
  market_model.pkl          # MarketModel (multi-book medians)
  ensemble_weights.pkl      # meta-learner + isotonic calibrator
  ensemble_metrics.json     # full evaluation results
  submodel_paths.json       # path manifest for load_full()
  scan_YYYYMMDD.json        # scanner output per run
```