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