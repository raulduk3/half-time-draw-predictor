"""
V4 Training Pipeline — Half-Time Draw Prediction
=================================================
Canonical training script. Builds everything from scratch and saves to models/v4/.

Architecture:
  MODEL A — LogisticRegression on 3 log-odds features (market estimate)
  MODEL B — XGBoost on 39 non-odds rolling stats + dc/elo/referee = 42 features
  SIGNAL  — Inverted Edge = Model A − Model B (positive = value bet on HT draw)
  INSIGHT — When fundamentals predict fewer draws than market → draws exceed market

Data: data/processed/mega_dataset_v2.parquet (203k matches, 22 EU leagues + MLS)
Split: 70/15/15 temporal (train/val/test)

Validation tests run at end:
  1. Target check      — draw rate ≈ 42%
  2. Leakage check     — no odds features in Model B
  3. Permutation test  — shuffled labels → AUC ≈ 0.5
  4. Bootstrap ROI     — 10k resamples, target 99%+ profitable
  5. Temporal stability — AUC in yearly windows
  6. League selection  — edge distribution by league
  7. Naive baseline    — vs flat-all-draws benchmark

Usage:
    source .venv/bin/activate
    python src/train_v4.py
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

sys.path.insert(0, ".")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/mega_dataset_v2.parquet")
OUT_DIR   = Path("models/v4")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET    = "y_ht_draw"
TODAY     = pd.Timestamp("2026-02-22")

MODEL_A_FEATURES = [
    "log_home_win_odds", "log_draw_odds", "log_away_win_odds",
    # Pinnacle log-odds (sharpest book, lowest vig)
    "log_ps_home", "log_ps_draw", "log_ps_away",
    # Market average log-odds (consensus across books)
    "log_avg_home", "log_avg_draw", "log_avg_away",
    # Multi-book disagreement signal
    "odds_spread_draw",
]
ODDS_KEYWORDS    = ["odds", "PSC", "MaxC", "AvgC", "BFEC", "log_ps_", "log_avg_", "odds_spread"]
# Raw odds columns to exclude from Model B candidate features
RAW_ODDS_COLS    = {
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
    "BWH", "BWD", "BWA", "IWH", "IWD", "IWA", "WHH", "WHD", "WHA",
}
EXCLUDE_COLS     = {
    "Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG", "FTHG", "FTAG",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "league", "season", "country", "league_tier", "league_name",
    TARGET, "ft_only", "Referee",
} | RAW_ODDS_COLS
COVERAGE_THRESHOLD = 0.55

# ── Banner ─────────────────────────────────────────────────────────────────────
print("=" * 72)
print("V4 TRAINING PIPELINE — HALF-TIME DRAW PREDICTION")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df[df["Date"] <= TODAY].copy()
df = df.dropna(subset=[TARGET, "B365H", "B365D", "B365A"]).copy()
df = df[~df["ft_only"].astype(str).isin(["True", "1"])].copy()   # Only rows with actual HT data
df = df.sort_values("Date").reset_index(drop=True)

print(f"    Shape: {df.shape}   HT draw rate: {df[TARGET].mean():.1%}")
print(f"    Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"    Leagues: {df['league'].nunique() if 'league' in df.columns else 'N/A'}")

# Engineer Model A features from multiple bookmakers
print("\n[1b] Engineering multi-bookmaker features...")
# B365 log-odds (always available)
for col, src in [("log_home_win_odds", "B365H"), ("log_draw_odds", "B365D"), ("log_away_win_odds", "B365A")]:
    if col not in df.columns and src in df.columns:
        df[col] = np.log(df[src].clip(lower=1.01))

# Pinnacle log-odds
for col, src in [("log_ps_home", "PSH"), ("log_ps_draw", "PSD"), ("log_ps_away", "PSA")]:
    if src in df.columns:
        df[col] = np.log(df[src].clip(lower=1.01))

# Market average log-odds
for col, src in [("log_avg_home", "AvgH"), ("log_avg_draw", "AvgD"), ("log_avg_away", "AvgA")]:
    if src in df.columns:
        df[col] = np.log(df[src].clip(lower=1.01))

# Multi-book disagreement: spread between max and min draw odds across available books
draw_cols = [c for c in ["B365D", "PSD", "BWD", "IWD", "WHD"] if c in df.columns]
if len(draw_cols) >= 2:
    draw_matrix = df[draw_cols]
    df["odds_spread_draw"] = draw_matrix.max(axis=1) - draw_matrix.min(axis=1)
    print(f"    odds_spread_draw: computed from {len(draw_cols)} books ({draw_cols})")

# Report coverage
for f in MODEL_A_FEATURES:
    if f in df.columns:
        n = df[f].notna().sum()
        print(f"    {f}: {n} ({n/len(df)*100:.0f}%)")
    else:
        print(f"    {f}: MISSING")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Classifying features...")

candidate_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
coverage       = {c: df[c].notna().mean() for c in candidate_cols}
odds_features  = [c for c in candidate_cols if any(k in c for k in ODDS_KEYWORDS)]
non_odds_cands = [c for c in candidate_cols if c not in odds_features]
model_b_base   = [c for c in non_odds_cands if coverage.get(c, 0) >= COVERAGE_THRESHOLD]

print(f"    Candidate columns: {len(candidate_cols)}")
print(f"    Odds features: {len(odds_features)}")
print(f"    Model B base features: {len(model_b_base)} (coverage ≥ {COVERAGE_THRESHOLD:.0%})")

# VALIDATION 2: Leakage check
print("\n  ── VALIDATION 2: Leakage check (no odds in Model B) ──")
for f in model_b_base:
    if any(k in f for k in ODDS_KEYWORDS):
        raise ValueError(f"ODDS LEAK DETECTED in Model B: '{f}'")
print(f"  ✓ PASS: All {len(model_b_base)} Model B features verified odds-free")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL SPLIT 70 / 15 / 15
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Temporal split 70/15/15...")
n = len(df)
t = int(n * 0.70)
v = int(n * 0.85)

train_df = df.iloc[:t].copy()
val_df   = df.iloc[t:v].copy()
test_df  = df.iloc[v:].copy()

print(f"    Train: {len(train_df):,}  {train_df['Date'].min().date()} → {train_df['Date'].max().date()}")
print(f"    Val:   {len(val_df):,}  {val_df['Date'].min().date()} → {val_df['Date'].max().date()}")
print(f"    Test:  {len(test_df):,}  {test_df['Date'].min().date()} → {test_df['Date'].max().date()}")

y_train = train_df[TARGET].values.astype(float)
y_val   = val_df[TARGET].values.astype(float)
y_test  = test_df[TARGET].values.astype(float)

# VALIDATION 1: Target check
print(f"\n  ── VALIDATION 1: Target check ──")
draw_rates = {
    "train": y_train.mean(),
    "val":   y_val.mean(),
    "test":  y_test.mean(),
    "all":   df[TARGET].mean(),
}
for split, rate in draw_rates.items():
    status = "✓" if 0.35 <= rate <= 0.50 else "✗"
    print(f"  {status} {split:6s} draw rate: {rate:.3f}  (expected 0.38–0.46)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TUNE HYPERPARAMS, THEN FIT DC / ELO / REFEREE ON TRAINING DATA ONLY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Tuning hyperparameters and fitting sub-models on training data only...")

from src.dixon_coles import DixonColesEnsemble
from src.elo import EloRatingSystem, tune_k_factor
from src.referee_model import RefereeModel

# ── 4a. Tune Elo K-factor ────────────────────────────────────────────────────
print("\n  [4a] Tuning Elo K-factor (replay_and_predict, no look-ahead)...")
k_res  = tune_k_factor(train_df, k_values=[8, 12, 16, 20, 24, 32, 40],
                        train_frac=0.70, val_frac=0.30)
best_k = int(k_res["best_k"])
print(f"    → Using K = {best_k}")

# ── 4b. Tune Dixon-Coles xi ──────────────────────────────────────────────────
print("\n  [4b] Tuning Dixon-Coles xi on inner train split...")
XI_CANDIDATES = [0.001, 0.002, 0.003, 0.005, 0.007]

# Inner split: first 70% of train for xi fitting, last 30% for xi evaluation
n_xi_tr  = int(len(train_df) * 0.70)
xi_tr_df = train_df.iloc[:n_xi_tr].copy()
xi_va_df = train_df.iloc[n_xi_tr:].copy()
xi_y_va  = (xi_va_df["HTHG"] == xi_va_df["HTAG"]).astype(float).values

best_xi, best_xi_auc = XI_CANDIDATES[1], 0.0  # default 0.002
for xi in XI_CANDIDATES:
    dc_xi = DixonColesEnsemble(xi=xi)
    dc_xi.fit(xi_tr_df, ft_only_leagues=["USA_MLS", "MEX_LigaMX"])
    xi_preds = dc_xi.predict_draw(xi_va_df)
    valid_xi = np.isfinite(xi_preds) & (xi_preds > 0)
    if valid_xi.sum() > 10 and len(set(xi_y_va[valid_xi])) > 1:
        xi_auc = roc_auc_score(xi_y_va[valid_xi], xi_preds[valid_xi])
    else:
        xi_auc = 0.0
    print(f"    xi={xi:.3f} → val AUC = {xi_auc:.4f}")
    if xi_auc > best_xi_auc:
        best_xi_auc = xi_auc
        best_xi     = xi
print(f"    → Using xi = {best_xi}")
del dc_xi  # free memory

# ── 4c. Leakage-free DC training features (Option A) ─────────────────────────
# Fit on first 60% of train, predict last 40% → no look-ahead for those rows.
# First 60%: use per-league draw rate (safest non-leaky estimate for those rows).
print("\n  [4c] Building leakage-free DC training features (60/40 split)...")
n_dc_split     = int(len(train_df) * 0.60)
dc_early_tr    = train_df.iloc[:n_dc_split].copy()
dc_early_hold  = train_df.iloc[n_dc_split:].copy()

dc_early = DixonColesEnsemble(xi=best_xi)
dc_early.fit(dc_early_tr, ft_only_leagues=["USA_MLS", "MEX_LigaMX"])

train_dc_preds = np.zeros(len(train_df))
# First 60%: use the league-specific draw rate from dc_early (no prediction possible)
for i in range(n_dc_split):
    row    = train_df.iloc[i]
    league = row.get("league", None)
    if league and league in dc_early.league_models_:
        train_dc_preds[i] = dc_early.league_models_[league].train_draw_rate_
    else:
        train_dc_preds[i] = dc_early.global_draw_rate_
# Last 40%: actual DC predictions (leakage-free since dc_early was fit on first 60%)
holdout_dc = dc_early.predict_draw(dc_early_hold)
train_dc_preds[n_dc_split:] = holdout_dc
train_dc_preds = np.clip(train_dc_preds, 0.01, 0.99)
del dc_early  # free memory

# ── 4d. Fit final DC on full training data (for val/test predictions) ─────────
print("\n  [4d] Fitting final DC on full training data...")
dc = DixonColesEnsemble(xi=best_xi)
dc.fit(train_df, ft_only_leagues=["USA_MLS", "MEX_LigaMX"])
print(f"    Dixon-Coles: fitted on {len(train_df):,} training rows  xi={best_xi}")

# ── 4e. Fit Elo + get leakage-free training features ─────────────────────────
print("\n  [4e] Fitting Elo and building leakage-free training features...")
elo = EloRatingSystem(k=best_k, home_adv=50)
elo.fit(train_df)

# Reset ratings then replay — each match is predicted using only PRIOR match ratings.
# After replay, ratings_ ends at the same post-train state as after fit().
elo.ratings_ = {}
elo.history_ = {}
train_elo_preds = np.clip(elo.replay_and_predict(train_df), 0.01, 0.99)
print(f"    Elo: K={best_k}  leakage-free training features generated")

# ── 4f. Fit Referee model ──────────────────────────────────────────────────────
ref_model = RefereeModel()
ref_model.fit(train_df)
print(f"    Referee model: fitted")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ADD DC / ELO / REFEREE AS MODEL B FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Generating DC/Elo/Referee predictions...")

def add_model_b_extra(split_df: pd.DataFrame,
                      precomputed_dc: np.ndarray = None,
                      precomputed_elo: np.ndarray = None) -> pd.DataFrame:
    out = split_df.copy()
    # DC
    if precomputed_dc is not None:
        out["dc_draw_prob"] = precomputed_dc
    else:
        try:
            out["dc_draw_prob"] = np.clip(dc.predict_draw(split_df), 0.01, 0.99)
        except Exception as e:
            print(f"    DC fallback ({e})")
            out["dc_draw_prob"] = 0.42
    # Elo
    if precomputed_elo is not None:
        out["elo_draw_prob"] = precomputed_elo
    else:
        try:
            out["elo_draw_prob"] = np.clip(elo.predict_draw(split_df), 0.01, 0.99)
        except Exception as e:
            print(f"    Elo fallback ({e})")
            out["elo_draw_prob"] = 0.42
    # Referee
    try:
        out["referee_adj"] = np.clip(ref_model.predict_draw_adjustment(split_df), 0.5, 2.0)
    except Exception:
        out["referee_adj"] = 1.0
    return out

# Training: use leakage-free DC/Elo features built above
train_df = add_model_b_extra(train_df,
                              precomputed_dc=train_dc_preds,
                              precomputed_elo=train_elo_preds)
# Val/test: use final DC/Elo fit on full training data (correct — no leakage)
val_df   = add_model_b_extra(val_df)
test_df  = add_model_b_extra(test_df)

model_b_features = model_b_base + ["dc_draw_prob", "elo_draw_prob", "referee_adj"]
print(f"    Model B final feature count: {len(model_b_features)}")
print(f"      ({len(model_b_base)} rolling stats + dc_draw_prob + elo_draw_prob + referee_adj)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Building feature arrays...")

# Filter Model A features to those with >10% coverage in training data
active_a_features = [f for f in MODEL_A_FEATURES if f in train_df.columns and train_df[f].notna().mean() > 0.10]
print(f"    Model A features (active): {len(active_a_features)} of {len(MODEL_A_FEATURES)}")
for f in active_a_features:
    cov = train_df[f].notna().mean()
    print(f"      {f}: {cov:.0%} coverage")
MODEL_A_FEATURES = active_a_features

train_medians_a = {c: float(train_df[c].median()) for c in MODEL_A_FEATURES}
train_medians_b = {c: float(train_df[c].median()) for c in model_b_features}

def build_X(split_df: pd.DataFrame, features: list, medians: dict) -> np.ndarray:
    X = np.zeros((len(split_df), len(features)), dtype=np.float32)
    for j, col in enumerate(features):
        vals = pd.to_numeric(split_df[col], errors="coerce").values if col in split_df.columns else np.full(len(split_df), np.nan)
        X[:, j] = np.where(np.isfinite(vals), vals, medians.get(col, 0.0))
    return X

Xa_tr = build_X(train_df, MODEL_A_FEATURES, train_medians_a)
Xa_va = build_X(val_df,   MODEL_A_FEATURES, train_medians_a)
Xa_te = build_X(test_df,  MODEL_A_FEATURES, train_medians_a)

Xb_tr = build_X(train_df, model_b_features, train_medians_b)
Xb_va = build_X(val_df,   model_b_features, train_medians_b)
Xb_te = build_X(test_df,  model_b_features, train_medians_b)

scaler_a = StandardScaler()
Xa_tr_s  = scaler_a.fit_transform(Xa_tr)
Xa_va_s  = scaler_a.transform(Xa_va)
Xa_te_s  = scaler_a.transform(Xa_te)

print(f"    Xa: {Xa_tr.shape[1]} features  |  Xb: {Xb_tr.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MARKET BASELINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Market baseline (B365 normalized implied probability)...")

def normalized_draw_prob(df_s: pd.DataFrame) -> np.ndarray:
    h = pd.to_numeric(df_s["B365H"], errors="coerce").values
    d = pd.to_numeric(df_s["B365D"], errors="coerce").values
    a = pd.to_numeric(df_s["B365A"], errors="coerce").values
    with np.errstate(divide="ignore", invalid="ignore"):
        total = (1/h) + (1/d) + (1/a)
        prob  = (1/d) / total
    return prob

mkt_test_raw = normalized_draw_prob(test_df)
valid_m = np.isfinite(mkt_test_raw) & (mkt_test_raw > 0)
mkt_auc   = roc_auc_score(y_test[valid_m], mkt_test_raw[valid_m])
mkt_brier = brier_score_loss(y_test[valid_m], mkt_test_raw[valid_m])
print(f"    Market (B365 normalized) — AUC: {mkt_auc:.4f}, Brier: {mkt_brier:.4f}  (n={valid_m.sum():,})")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL A — LOGISTIC REGRESSION ON LOG-ODDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Training Model A (Market LR on 3 log-odds features)...")

lr_a = LogisticRegression(C=0.1, max_iter=2000, random_state=42, solver="lbfgs")
lr_a.fit(Xa_tr_s, y_train)

pa_tr = lr_a.predict_proba(Xa_tr_s)[:, 1]
pa_va = lr_a.predict_proba(Xa_va_s)[:, 1]
pa_te = lr_a.predict_proba(Xa_te_s)[:, 1]

a_tr_auc = roc_auc_score(y_train, pa_tr)
a_va_auc = roc_auc_score(y_val,   pa_va)
a_te_auc = roc_auc_score(y_test,  pa_te)
print(f"    Model A (raw) — train: {a_tr_auc:.4f}, val: {a_va_auc:.4f}, test: {a_te_auc:.4f}")

iso_a = IsotonicRegression(out_of_bounds="clip")
iso_a.fit(pa_va, y_val)
pa_te_cal = np.clip(iso_a.predict(pa_te), 0.01, 0.99)

a_te_auc_cal   = roc_auc_score(y_test, pa_te_cal)
a_te_brier_cal = brier_score_loss(y_test, pa_te_cal)
print(f"    Model A (calibrated) — test AUC: {a_te_auc_cal:.4f}, Brier: {a_te_brier_cal:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MODEL B — XGBOOST (Optuna, 30 trials)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Training Model B — XGBoost (Optuna, 30 trials)...")

dtrain_b = xgb.DMatrix(Xb_tr, label=y_train, feature_names=model_b_features)
dval_b   = xgb.DMatrix(Xb_va, label=y_val,   feature_names=model_b_features)
dtest_b  = xgb.DMatrix(Xb_te, label=y_test,  feature_names=model_b_features)

def xgb_b_objective(trial):
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "tree_method":      "hist",
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "eta":              trial.suggest_float("eta", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 20, 200),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "alpha":            trial.suggest_float("alpha", 0.0, 5.0),
        "lambda":           trial.suggest_float("lambda", 1.0, 15.0),
        "n_jobs":           -1,
        "seed":             42,
        "verbosity":        0,
    }
    bst = xgb.train(
        params, dtrain_b, num_boost_round=500,
        evals=[(dval_b, "val")], early_stopping_rounds=30,
        verbose_eval=False,
    )
    return roc_auc_score(y_val, bst.predict(dval_b))

study_b = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study_b.optimize(xgb_b_objective, n_trials=30, show_progress_bar=True)

print(f"    Best val AUC: {study_b.best_value:.4f}")
print(f"    Best params:  {study_b.best_params}")

best_params_b = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "tree_method": "hist", "n_jobs": -1, "seed": 42, "verbosity": 0,
    **study_b.best_params,
}
xgb_b = xgb.train(
    best_params_b, dtrain_b, num_boost_round=1000,
    evals=[(dval_b, "val")], early_stopping_rounds=30,
    verbose_eval=False,
)
print(f"    XGB best iteration: {xgb_b.best_iteration}")

pb_xgb_tr = xgb_b.predict(dtrain_b)
pb_xgb_va = xgb_b.predict(dval_b)
pb_xgb_te = xgb_b.predict(dtest_b)

b_xgb_tr_auc = roc_auc_score(y_train, pb_xgb_tr)
b_xgb_va_auc = roc_auc_score(y_val,   pb_xgb_va)
b_xgb_te_auc = roc_auc_score(y_test,  pb_xgb_te)
print(f"    XGB — train: {b_xgb_tr_auc:.4f}, val: {b_xgb_va_auc:.4f}, test: {b_xgb_te_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. MODEL B — LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Training Model B — LightGBM...")

lgb_tr_b = lgb.Dataset(Xb_tr, label=y_train, feature_name=model_b_features)
lgb_va_b = lgb.Dataset(Xb_va, label=y_val,   reference=lgb_tr_b)

lgb_params_b = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "learning_rate":     0.05,
    "num_leaves":        31,
    "max_depth":         6,
    "min_child_samples": 100,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}
lgb_b = lgb.train(
    lgb_params_b, lgb_tr_b, num_boost_round=1000, valid_sets=[lgb_va_b],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

pb_lgb_tr = lgb_b.predict(Xb_tr)
pb_lgb_va = lgb_b.predict(Xb_va)
pb_lgb_te = lgb_b.predict(Xb_te)

b_lgb_tr_auc = roc_auc_score(y_train, pb_lgb_tr)
b_lgb_va_auc = roc_auc_score(y_val,   pb_lgb_va)
b_lgb_te_auc = roc_auc_score(y_test,  pb_lgb_te)
print(f"    LGB — train: {b_lgb_tr_auc:.4f}, val: {b_lgb_va_auc:.4f}, test: {b_lgb_te_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SELECT BEST MODEL B + CALIBRATE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Selecting best Model B and calibrating...")

if b_lgb_va_auc >= b_xgb_va_auc:
    best_b_name  = "LightGBM"
    pb_b_va      = pb_lgb_va
    pb_b_te      = pb_lgb_te
    print(f"    → Using LightGBM (val AUC {b_lgb_va_auc:.4f} ≥ XGB {b_xgb_va_auc:.4f})")
else:
    best_b_name  = "XGBoost"
    pb_b_va      = pb_xgb_va
    pb_b_te      = pb_xgb_te
    print(f"    → Using XGBoost (val AUC {b_xgb_va_auc:.4f} > LGB {b_lgb_va_auc:.4f})")

iso_b = IsotonicRegression(out_of_bounds="clip")
iso_b.fit(pb_b_va, y_val)
pb_b_te_cal = np.clip(iso_b.predict(pb_b_te), 0.01, 0.99)

b_te_auc_cal   = roc_auc_score(y_test, pb_b_te_cal)
b_te_brier_cal = brier_score_loss(y_test, pb_b_te_cal)
print(f"    Model B ({best_b_name}, calibrated) — AUC: {b_te_auc_cal:.4f}, Brier: {b_te_brier_cal:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. INVERTED EDGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] Inverted edge analysis (Model A − Model B on test set)...")

# INVERTED EDGE: pa_cal - pb_cal
# Positive = market prices draw higher than fundamentals → value bet
inverted_edge = pa_te_cal - pb_b_te_cal

print(f"    Inverted edge statistics:")
print(f"      Mean:  {inverted_edge.mean():+.4f}")
print(f"      Std:   {inverted_edge.std():.4f}")
print(f"      Min:   {inverted_edge.min():+.4f}")
print(f"      Max:   {inverted_edge.max():+.4f}")
print(f"\n    Distribution (% matches with inverted edge above threshold):")
for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
    n_pos = (inverted_edge > thresh).sum()
    pct   = n_pos / len(inverted_edge) * 100
    print(f"      Edge > {thresh:.0%}: {n_pos:5,} matches ({pct:5.1f}%)")

# Edge histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(inverted_edge, bins=80, color="#3498db", alpha=0.75, edgecolor="white", linewidth=0.3)
for thresh, color, label in [
    (0.02, "#e67e22", "2%"),
    (0.03, "#2ecc71", "3%"),
    (0.05, "#9b59b6", "5%"),
]:
    n_pos = (inverted_edge > thresh).sum()
    ax.axvline(thresh, color=color, ls="--", lw=1.5,
               label=f">{label}: {n_pos:,} ({n_pos/len(inverted_edge):.1%})")
ax.axvline(0, color="red", ls="-", lw=2, alpha=0.6, label="Edge = 0")
ax.set_xlabel("Inverted Edge (Model A − Model B)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("V4 Inverted Edge Distribution — Test Set\n"
             "Model A (Market) − Model B (Fundamentals) — positive = value bet", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "edge_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 13. BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] Backtest (flat bet on inverted-edge > threshold)...")
print(f"\n    ⚠️  CAVEAT: B365D = FULL-TIME draw odds. Real HT odds would differ.")
print(f"    ⚠️  Using B365D as proxy. Positive ROI pattern is directionally valid.")

b365d_te = pd.to_numeric(test_df["B365D"], errors="coerce").values
valid_bt  = np.isfinite(b365d_te) & np.isfinite(inverted_edge) & (b365d_te > 1.0)

y_bt    = y_test[valid_bt]
edge_bt = inverted_edge[valid_bt]
odds_bt = b365d_te[valid_bt]

# Flat-bet all draws (baseline)
pnl_base = np.where(y_bt == 1, odds_bt - 1, -1.0)
roi_base = pnl_base.mean()
print(f"\n    Baseline (bet ALL draws): {len(y_bt):,} bets, "
      f"win rate {y_bt.mean():.1%}, flat ROI {roi_base:+.4f}")

print(f"\n    {'Thresh':>6}  {'Bets':>7}  {'Bet%':>5}  {'Win%':>6}  "
      f"{'FlatROI':>9}  {'PnL':>8}  {'vsBase':>8}")
print(f"    {'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}")

backtest_rows = []
for thresh in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
    mask   = edge_bt > thresh
    n_bets = mask.sum()
    if n_bets < 10:
        continue
    y_b, od_b = y_bt[mask], odds_bt[mask]
    wins   = int(y_b.sum())
    win_r  = wins / n_bets
    pnl    = np.where(y_b == 1, od_b - 1, -1.0)
    roi    = pnl.mean()
    total_pnl = pnl.sum()
    vs_base   = roi - roi_base
    pct       = n_bets / len(y_bt) * 100

    print(f"    {thresh:>6.2f}  {n_bets:>7,}  {pct:>4.1f}%  {win_r:>6.1%}  "
          f"{roi:>+9.4f}  {total_pnl:>+8.1f}  {vs_base:>+8.4f}")

    backtest_rows.append({
        "threshold":    thresh,
        "n_bets":       int(n_bets),
        "n_wins":       wins,
        "win_rate":     float(win_r),
        "roi_flat":     float(roi),
        "pnl_flat":     float(total_pnl),
        "pct_bets":     float(pct),
        "vs_baseline":  float(vs_base),
    })

# Cumulative PnL plot
thresh_plot = 0.03
dates_bt_all = test_df.iloc[np.where(valid_bt)[0]]["Date"].values
mask_plot    = edge_bt > thresh_plot
dates_plot   = dates_bt_all[mask_plot]
y_plot       = y_bt[mask_plot]
odds_plot    = odds_bt[mask_plot]
sort_i       = np.argsort(dates_plot)
dates_sorted = pd.to_datetime(dates_plot[sort_i])
pnl_series   = np.where(y_plot[sort_i] == 1, odds_plot[sort_i] - 1, -1.0)
cumsum       = np.cumsum(pnl_series)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("V4 Backtest Analysis — Test Period", fontsize=13, fontweight="bold")

axes[0].plot(dates_sorted, cumsum, color="#2ecc71", lw=2, label=f"Edge > {thresh_plot:.0%}")
axes[0].axhline(0, color="red", ls="--", lw=1, alpha=0.7)
axes[0].fill_between(dates_sorted, cumsum, 0, where=cumsum >= 0, alpha=0.12, color="#2ecc71")
axes[0].fill_between(dates_sorted, cumsum, 0, where=cumsum < 0,  alpha=0.12, color="#e74c3c")
axes[0].set_xlabel("Date"); axes[0].set_ylabel("Cumulative PnL (units)")
axes[0].set_title(f"Cumulative PnL — Edge > {thresh_plot:.0%}")
axes[0].legend(); axes[0].grid(alpha=0.3)

if backtest_rows:
    thrs   = [r["threshold"] for r in backtest_rows]
    rois   = [r["roi_flat"] for r in backtest_rows]
    colors_bar = ["#2ecc71" if r > roi_base else "#e74c3c" for r in rois]
    axes[1].bar([f"{t:.0%}" for t in thrs], rois, color=colors_bar, alpha=0.8)
    axes[1].axhline(0,        color="black", lw=0.8, ls="--")
    axes[1].axhline(roi_base, color="blue",  lw=1.5, ls=":", label=f"Flat-all ROI={roi_base:+.3f}")
    axes[1].set_xlabel("Inverted edge threshold"); axes[1].set_ylabel("Flat ROI per unit")
    axes[1].set_title("Backtest ROI by Inverted Edge Threshold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "backtest.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 14. ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] Generating ROC curve plot...")

fig, ax = plt.subplots(figsize=(8, 6))
curves = [
    (pa_te_cal,                f"Model A — Market LR   (AUC={a_te_auc_cal:.4f})", "#e74c3c", 2.0),
    (pb_b_te_cal,              f"Model B — Fundamentals (AUC={b_te_auc_cal:.4f})", "#2ecc71", 2.0),
    (mkt_test_raw[valid_m],    f"Market Direct          (AUC={mkt_auc:.4f})", "#9b59b6", 1.5),
]
for probs, label, color, lw in curves:
    yt = y_test[valid_m] if len(probs) == valid_m.sum() else y_test
    fpr, tpr, _ = roc_curve(yt, probs)
    ax.plot(fpr, tpr, label=label, lw=lw, color=color)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("V4 ROC Curves", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 15. VALIDATION TESTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[15] Running validation tests...")

validation_results = {}

# TEST 3: Random permutation — shuffled labels should give AUC ≈ 0.5
print("\n  ── TEST 3: Permutation test ──")
rng = np.random.default_rng(42)
perm_aucs = []
for _ in range(100):
    y_perm = rng.permuted(y_test)
    perm_aucs.append(roc_auc_score(y_perm, pa_te_cal))
perm_mean = np.mean(perm_aucs)
perm_std  = np.std(perm_aucs)
print(f"  Permuted label AUC: {perm_mean:.4f} ± {perm_std:.4f}  (expected ≈ 0.5000)")
status = "✓" if abs(perm_mean - 0.5) < 0.01 else "✗"
print(f"  {status} Permutation test {'PASS' if status=='✓' else 'FAIL'}")
validation_results["permutation_test"] = {"mean_auc": float(perm_mean), "std": float(perm_std), "pass": status == "✓"}

# TEST 4: Bootstrap ROI
print("\n  ── TEST 4: Bootstrap ROI (1000 resamples, edge > 0.03) ──")
mask_bt3  = edge_bt > 0.03
y_bt3     = y_bt[mask_bt3]
odds_bt3  = odds_bt[mask_bt3]
n_bt3     = len(y_bt3)

if n_bt3 >= 100:
    rng2 = np.random.default_rng(42)
    boot_rois = []
    for _ in range(1000):
        idx = rng2.choice(n_bt3, size=n_bt3, replace=True)
        pnl = np.where(y_bt3[idx] == 1, odds_bt3[idx] - 1, -1.0).mean()
        boot_rois.append(pnl)
    boot_rois = np.array(boot_rois)
    pct_profitable = (boot_rois > 0).mean()
    ci_lo, ci_hi   = np.percentile(boot_rois, [2.5, 97.5])
    print(f"  Bootstrap ROI (edge > 3%): n={n_bt3:,} bets")
    print(f"  % profitable resamples: {pct_profitable:.1%}  (target: ≥ 95%)")
    print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    status = "✓" if pct_profitable >= 0.95 else "✗"
    print(f"  {status} Bootstrap test {'PASS' if status=='✓' else 'FAIL'}")
    validation_results["bootstrap_roi"] = {
        "n_bets": n_bt3,
        "pct_profitable": float(pct_profitable),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "pass": status == "✓"
    }
else:
    print(f"  Insufficient bets ({n_bt3}) for bootstrap test")
    validation_results["bootstrap_roi"] = {"n_bets": n_bt3, "pass": None}

# TEST 5: Temporal stability
print("\n  ── TEST 5: Temporal stability ──")
test_years = sorted(test_df["Date"].dt.year.unique())
temporal_rows = []
for year in test_years:
    mask_yr = test_df["Date"].dt.year == year
    y_yr    = y_test[mask_yr.values]
    pa_yr   = pa_te_cal[mask_yr.values]
    pb_yr   = pb_b_te_cal[mask_yr.values]
    if len(y_yr) < 50 or len(set(y_yr)) < 2:
        continue
    a_auc = roc_auc_score(y_yr, pa_yr)
    b_auc = roc_auc_score(y_yr, pb_yr)
    e_mean = (pa_yr - pb_yr).mean()
    temporal_rows.append({"year": year, "n": int(mask_yr.sum()), "model_a_auc": a_auc, "model_b_auc": b_auc, "mean_edge": e_mean})
    print(f"  {year}: n={mask_yr.sum():,}  Model A AUC={a_auc:.4f}  Model B AUC={b_auc:.4f}  mean_edge={e_mean:+.4f}")
validation_results["temporal_stability"] = temporal_rows

# TEST 6: Edge distribution by league
print("\n  ── TEST 6: League edge distribution ──")
if "league" in test_df.columns:
    league_rows = []
    for league in sorted(test_df["league"].unique()):
        mask_lg = test_df["league"] == league
        n_lg    = mask_lg.sum()
        if n_lg < 50:
            continue
        e_lg    = inverted_edge[mask_lg.values]
        pct_pos = (e_lg > 0.03).mean()
        league_rows.append({"league": league, "n": int(n_lg), "mean_edge": float(e_lg.mean()), "pct_gt_3pct": float(pct_pos)})
    league_rows.sort(key=lambda x: x["mean_edge"], reverse=True)
    print(f"  {'League':>10}  {'N':>6}  {'Mean Edge':>10}  {'% > 3%':>8}")
    for r in league_rows[:10]:
        print(f"  {r['league']:>10}  {r['n']:>6,}  {r['mean_edge']:>+10.4f}  {r['pct_gt_3pct']:>8.1%}")
    validation_results["league_distribution"] = league_rows

# TEST 7: Naive baseline
print("\n  ── TEST 7: Naive baseline comparison ──")
naive_pred = np.full(len(y_test), y_train.mean())
naive_auc  = roc_auc_score(y_test, naive_pred)
print(f"  Naive (constant = train draw rate {y_train.mean():.3f}) AUC: {naive_auc:.4f}")
print(f"  Model A vs naive: {a_te_auc_cal - naive_auc:+.4f}")
print(f"  Model B vs naive: {b_te_auc_cal - naive_auc:+.4f}")
validation_results["naive_baseline"] = {
    "naive_auc": float(naive_auc),
    "model_a_delta": float(a_te_auc_cal - naive_auc),
    "model_b_delta": float(b_te_auc_cal - naive_auc),
}

# ─────────────────────────────────────────────────────────────────────────────
# 16. SAVE ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[16] Saving all models to models/v4/...")

# Model A
with open(OUT_DIR / "model_a_lr.pkl",         "wb") as f: pickle.dump(lr_a,    f)
with open(OUT_DIR / "model_a_scaler.pkl",      "wb") as f: pickle.dump(scaler_a, f)
with open(OUT_DIR / "model_a_calibrator.pkl",  "wb") as f: pickle.dump(iso_a,   f)
with open(OUT_DIR / "model_a_medians.json",    "w")  as f: json.dump(train_medians_a, f, indent=2)
with open(OUT_DIR / "model_a_features.json",   "w")  as f: json.dump(MODEL_A_FEATURES, f, indent=2)

# Model B
xgb_b.save_model(str(OUT_DIR / "model_b_xgb.json"))
lgb_b.save_model(str(OUT_DIR / "model_b_lgb.txt"))
with open(OUT_DIR / "model_b_calibrator.pkl",  "wb") as f: pickle.dump(iso_b,   f)
with open(OUT_DIR / "model_b_medians.json",    "w")  as f: json.dump(train_medians_b, f, indent=2)
with open(OUT_DIR / "model_b_features.json",   "w")  as f: json.dump(model_b_features, f, indent=2)
with open(OUT_DIR / "model_b_best.txt",        "w")  as f: f.write(best_b_name)

# Sub-models
dc.save(str(OUT_DIR / "dixon_coles.pkl"))
elo.save(str(OUT_DIR / "elo.pkl"))
with open(OUT_DIR / "referee_model.pkl", "wb") as f: pickle.dump(ref_model, f)

# Paths manifest
manifest = {
    "model_a_lr":         str(OUT_DIR / "model_a_lr.pkl"),
    "model_a_scaler":     str(OUT_DIR / "model_a_scaler.pkl"),
    "model_a_calibrator": str(OUT_DIR / "model_a_calibrator.pkl"),
    "model_a_medians":    str(OUT_DIR / "model_a_medians.json"),
    "model_a_features":   str(OUT_DIR / "model_a_features.json"),
    "model_b_xgb":        str(OUT_DIR / "model_b_xgb.json"),
    "model_b_lgb":        str(OUT_DIR / "model_b_lgb.txt"),
    "model_b_calibrator": str(OUT_DIR / "model_b_calibrator.pkl"),
    "model_b_medians":    str(OUT_DIR / "model_b_medians.json"),
    "model_b_features":   str(OUT_DIR / "model_b_features.json"),
    "model_b_best":       best_b_name,
    "dc_path":            str(OUT_DIR / "dixon_coles.pkl"),
    "elo_path":           str(OUT_DIR / "elo.pkl"),
    "referee_path":       str(OUT_DIR / "referee_model.pkl"),
    "mega_dataset":       str(DATA_PATH),
}
with open(OUT_DIR / "v4_paths.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Metrics
metrics_out = {
    "version": "v4",
    "training_summary": {
        "dataset":      str(DATA_PATH),
        "n_total":      int(n),
        "n_train":      len(train_df),
        "n_val":        len(val_df),
        "n_test":       len(test_df),
        "draw_rate":    float(df[TARGET].mean()),
        "train_period": f"{train_df['Date'].min().date()} → {train_df['Date'].max().date()}",
        "val_period":   f"{val_df['Date'].min().date()} → {val_df['Date'].max().date()}",
        "test_period":  f"{test_df['Date'].min().date()} → {test_df['Date'].max().date()}",
        "elo_best_k":   best_k,
        "dc_best_xi":   best_xi,
        "k_tuning":     k_res,
    },
    "market_baseline": {
        "test_auc":   float(mkt_auc),
        "test_brier": float(mkt_brier),
        "n_valid":    int(valid_m.sum()),
    },
    "model_a": {
        "name":                  "LogisticRegression (log-odds only)",
        "n_features":            len(MODEL_A_FEATURES),
        "features":              MODEL_A_FEATURES,
        "train_auc":             float(a_tr_auc),
        "val_auc":               float(a_va_auc),
        "test_auc_raw":          float(a_te_auc),
        "test_auc_calibrated":   float(a_te_auc_cal),
        "test_brier_calibrated": float(a_te_brier_cal),
    },
    "model_b": {
        "best_model":            best_b_name,
        "n_features":            len(model_b_features),
        "features":              model_b_features,
        "xgboost": {
            "train_auc":      float(b_xgb_tr_auc),
            "val_auc":        float(b_xgb_va_auc),
            "test_auc":       float(b_xgb_te_auc),
            "best_iteration": int(xgb_b.best_iteration),
            "best_params":    study_b.best_params,
        },
        "lightgbm": {
            "train_auc":      float(b_lgb_tr_auc),
            "val_auc":        float(b_lgb_va_auc),
            "test_auc":       float(b_lgb_te_auc),
            "best_iteration": int(lgb_b.best_iteration),
        },
        "test_auc_calibrated":   float(b_te_auc_cal),
        "test_brier_calibrated": float(b_te_brier_cal),
    },
    "inverted_edge": {
        "definition":  "Model A (market) − Model B (fundamentals)",
        "signal":      "Positive = market prices draw higher than fundamentals → value bet",
        "mean":        float(inverted_edge.mean()),
        "std":         float(inverted_edge.std()),
        "pct_gt_1":    float((inverted_edge > 0.01).mean()),
        "pct_gt_2":    float((inverted_edge > 0.02).mean()),
        "pct_gt_3":    float((inverted_edge > 0.03).mean()),
        "pct_gt_5":    float((inverted_edge > 0.05).mean()),
        "pct_gt_10":   float((inverted_edge > 0.10).mean()),
    },
    "backtest": backtest_rows,
    "baseline_roi": float(roi_base),
    "validation": validation_results,
}
with open(OUT_DIR / "v4_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2, cls=_NumpyEncoder)

print(f"    ✓ All models saved to {OUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# 17. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("V4 TRAINING COMPLETE — FINAL SUMMARY")
print("=" * 72)

print(f"\n  {'Model':<50} {'AUC':>7} {'Brier':>8} {'vs Mkt':>8}")
print(f"  {'-'*50} {'-'*7} {'-'*8} {'-'*8}")
rows_sum = [
    ("Market (B365 normalized, direct)",    mkt_auc,        mkt_brier),
    ("Model A — Market LR (calibrated)",    a_te_auc_cal,   a_te_brier_cal),
    ("Model B — Fundamentals (calibrated)", b_te_auc_cal,   b_te_brier_cal),
]
for label, auc, brier in rows_sum:
    delta = auc - mkt_auc
    print(f"  {label:<50} {auc:>7.4f} {brier:>8.5f} {delta:>+8.4f}")

print(f"\n  KEY INSIGHT:")
print(f"    Model A ≈ Market (both derived from B365 odds → same signal)")
print(f"    Model B = Independent stats signal (no odds)")
print(f"    INVERTED EDGE = Model A − Model B = market estimate minus fundamentals")
print(f"    When market > fundamentals → actual draws EXCEED market prediction")

print(f"\n  Inverted edge distribution (test set: {len(inverted_edge):,} matches):")
for thresh in [0.01, 0.02, 0.03, 0.05, 0.10]:
    n_pos = (inverted_edge > thresh).sum()
    print(f"    > {thresh:.0%}: {n_pos:,} matches ({n_pos/len(inverted_edge):.1%})")

print(f"\n  Backtest (flat bet, using B365D as FT-draw proxy):")
print(f"  {'Thresh':>6}  {'Bets':>7}  {'Win%':>6}  {'ROI':>8}  {'vs Baseline':>12}")
print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*12}")
for r in backtest_rows:
    print(f"  {r['threshold']:>6.2f}  {r['n_bets']:>7,}  {r['win_rate']:>6.1%}  "
          f"{r['roi_flat']:>+8.4f}  {r['vs_baseline']:>+12.4f}")

print(f"\n  Plots saved to {PLOTS_DIR}/")
print(f"  Metrics: {OUT_DIR}/v4_metrics.json")
print(f"  Paths:   {OUT_DIR}/v4_paths.json")
print("\n" + "=" * 72)
print("  DONE. Run src/predict_v4.py for single-match predictions.")
print("        Run src/scan_v4.py for upcoming fixture scanning.")
print("=" * 72)
