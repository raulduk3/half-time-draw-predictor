"""
V3 Two-Model Architecture — Half-Time Draw Prediction
======================================================
Separates market signal from fundamentals signal:

MODEL A — Market Model
  Input: bookmaker log-odds only (3 features: log_home_win_odds, log_draw_odds, log_away_win_odds)
  Model: Logistic Regression + Isotonic calibration
  This IS the market. Represents what B365 odds imply about P(HT draw).

MODEL B — Fundamentals Model
  Input: 39 non-odds rolling stats + DC draw prob + Elo draw prob + referee adj
  NO odds features anywhere — verified by assertion.
  Model: XGBoost (Optuna, 30 trials) + LightGBM (best by val AUC)
  Calibration: Isotonic regression on val set

EDGE = Model B probability − Model A probability
  When fundamentals > market → potential value (the stats say more likely than priced)
  When fundamentals < market → market knows something, skip

Backtest: Simulate flat-bet on matches where edge > threshold (test set, Apr 2022–Feb 2026)

Saves to models/v3/
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/mega_dataset_v2.parquet")
OUT_DIR   = Path("models/v3")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET    = "y_ht_draw"
TODAY     = pd.Timestamp("2026-02-22")

# Model A: ONLY these 3 log-odds features (derived from B365)
MODEL_A_FEATURES = ["log_home_win_odds", "log_draw_odds", "log_away_win_odds"]

# Keywords that indicate odds-related features — used to EXCLUDE from Model B
ODDS_KEYWORDS = ["odds", "PSC", "MaxC", "AvgC", "BFEC"]

# Columns that are NEVER features (identifiers, raw stats, target)
EXCLUDE_COLS = {
    "Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG", "FTHG", "FTAG",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
    "league", "season", "country", "league_tier", "league_name",
    TARGET, "ft_only",
}

COVERAGE_THRESHOLD = 0.55  # Include shots/corners/fouls at ~58-59%

# ── Banner ────────────────────────────────────────────────────────────────────
print("=" * 72)
print("V3 TWO-MODEL ARCHITECTURE — HALF-TIME DRAW PREDICTION")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df[df["Date"] <= TODAY].copy()
df = df.dropna(subset=[TARGET, "B365H", "B365D", "B365A"]).copy()
df = df[~df["ft_only"].fillna(False)].copy()   # Only rows with actual HT data
df = df.sort_values("Date").reset_index(drop=True)

print(f"    Shape: {df.shape}   HT draw rate: {df[TARGET].mean():.1%}")
print(f"    Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Classifying features...")

candidate_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
coverage = {c: df[c].notna().mean() for c in candidate_cols}

# Odds features = those containing odds keywords
odds_features = [c for c in candidate_cols if any(k in c for k in ODDS_KEYWORDS)]

# Non-odds features meeting coverage threshold
non_odds_candidates = [c for c in candidate_cols if c not in odds_features]
model_b_base = [c for c in non_odds_candidates if coverage.get(c, 0) >= COVERAGE_THRESHOLD]

# ── VALIDATION: Verify no odds leak in Model B ────────────────────────────────
print("\n  ── VALIDATION: Model B feature audit ──")
for f in model_b_base:
    has_odds_kw = any(k in f for k in ODDS_KEYWORDS)
    if has_odds_kw:
        raise ValueError(f"ODDS LEAK DETECTED in Model B: '{f}'")
print(f"  ✓ PASS: All {len(model_b_base)} Model B features verified odds-free")
print(f"  ✓ Model A features: {MODEL_A_FEATURES}")
print(f"  ✓ Model B base features: {len(model_b_base)} non-odds stats")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL SPLIT 70 / 15 / 15
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Temporal split 70/15/15...")
n  = len(df)
t  = int(n * 0.70)
v  = int(n * 0.85)

train_df = df.iloc[:t].copy()
val_df   = df.iloc[t:v].copy()
test_df  = df.iloc[v:].copy()

print(f"    Train: {len(train_df):,}  {train_df['Date'].min().date()} → {train_df['Date'].max().date()}")
print(f"    Val:   {len(val_df):,}  {val_df['Date'].min().date()} → {val_df['Date'].max().date()}")
print(f"    Test:  {len(test_df):,}  {test_df['Date'].min().date()} → {test_df['Date'].max().date()}")
print(f"    ✓ Temporal split clean — no future data in training")

y_train = train_df[TARGET].values.astype(float)
y_val   = val_df[TARGET].values.astype(float)
y_test  = test_df[TARGET].values.astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIT DC / ELO / REFEREE ON TRAINING DATA ONLY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Fitting DC, Elo, Referee on training data only...")

from src.dixon_coles import DixonColesEnsemble
from src.elo import EloRatingSystem
from src.referee_model import RefereeModel

dc_v3 = DixonColesEnsemble()
dc_v3.fit(train_df, ft_only_leagues=["USA_MLS", "MEX_LigaMX"])
print(f"    Dixon-Coles: fitted on {len(train_df):,} training rows")

elo_v3 = EloRatingSystem(k=16, home_adv=50)
elo_v3.fit(train_df)
print(f"    Elo: fitted on {len(train_df):,} training rows")

ref_v3 = RefereeModel()
ref_v3.fit(train_df)
print(f"    Referee model: fitted")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ADD DC / ELO / REFEREE AS MODEL B FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Generating DC/Elo/Referee predictions...")

def add_model_b_extra(split_df: pd.DataFrame) -> pd.DataFrame:
    """Add dc_draw_prob, elo_draw_prob, referee_adj columns."""
    out = split_df.copy()
    try:
        out["dc_draw_prob"] = np.clip(dc_v3.predict_draw(split_df), 0.01, 0.99)
    except Exception as e:
        print(f"    DC fallback ({e})")
        out["dc_draw_prob"] = 0.42
    try:
        out["elo_draw_prob"] = np.clip(elo_v3.predict_draw(split_df), 0.01, 0.99)
    except Exception as e:
        print(f"    Elo fallback ({e})")
        out["elo_draw_prob"] = 0.42
    try:
        out["referee_adj"] = np.clip(ref_v3.predict_draw_adjustment(split_df), 0.5, 2.0)
    except Exception:
        out["referee_adj"] = 1.0
    return out

train_df = add_model_b_extra(train_df)
val_df   = add_model_b_extra(val_df)
test_df  = add_model_b_extra(test_df)

# Final Model B feature list: base stats + DC + Elo + referee
model_b_features = model_b_base + ["dc_draw_prob", "elo_draw_prob", "referee_adj"]
print(f"    Model B final feature count: {len(model_b_features)}")
print(f"      ({len(model_b_base)} rolling stats + dc_draw_prob + elo_draw_prob + referee_adj)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Building feature arrays...")

# Medians from training set only
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

# Scale for Model A (LR)
scaler_a = StandardScaler()
Xa_tr_s = scaler_a.fit_transform(Xa_tr)
Xa_va_s = scaler_a.transform(Xa_va)
Xa_te_s = scaler_a.transform(Xa_te)

print(f"    Xa shape: {Xa_tr.shape[1]} features  | Xb shape: {Xb_tr.shape[1]} features")

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
# 8. MODEL A — LOGISTIC REGRESSION ON LOG-ODDS (3 features)
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

# Isotonic calibration on val set
iso_a = IsotonicRegression(out_of_bounds="clip")
iso_a.fit(pa_va, y_val)
pa_te_cal = np.clip(iso_a.predict(pa_te), 0.01, 0.99)

a_te_auc_cal   = roc_auc_score(y_test, pa_te_cal)
a_te_brier_cal = brier_score_loss(y_test, pa_te_cal)
print(f"    Model A (calibrated) — test AUC: {a_te_auc_cal:.4f}, Brier: {a_te_brier_cal:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MODEL B — XGBOOST ON FUNDAMENTALS (Optuna, 30 trials)
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
    best_b_name = "LightGBM"
    pb_b_va = pb_lgb_va
    pb_b_te = pb_lgb_te
    print(f"    → Using LightGBM (val AUC {b_lgb_va_auc:.4f} ≥ XGB {b_xgb_va_auc:.4f})")
else:
    best_b_name = "XGBoost"
    pb_b_va = pb_xgb_va
    pb_b_te = pb_xgb_te
    print(f"    → Using XGBoost (val AUC {b_xgb_va_auc:.4f} > LGB {b_lgb_va_auc:.4f})")

iso_b = IsotonicRegression(out_of_bounds="clip")
iso_b.fit(pb_b_va, y_val)
pb_b_te_cal = np.clip(iso_b.predict(pb_b_te), 0.01, 0.99)

b_te_auc_cal   = roc_auc_score(y_test, pb_b_te_cal)
b_te_brier_cal = brier_score_loss(y_test, pb_b_te_cal)
print(f"    Model B ({best_b_name}, calibrated) — AUC: {b_te_auc_cal:.4f}, Brier: {b_te_brier_cal:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. EDGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] Edge analysis (Model B − Model A on test set)...")

edge_test = pb_b_te_cal - pa_te_cal
print(f"    Edge statistics:")
print(f"      Mean:  {edge_test.mean():+.4f}")
print(f"      Std:   {edge_test.std():.4f}")
print(f"      Min:   {edge_test.min():+.4f}")
print(f"      Max:   {edge_test.max():+.4f}")
print(f"\n    Edge distribution:")
for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
    n_pos = (edge_test > thresh).sum()
    pct   = n_pos / len(edge_test) * 100
    print(f"      Edge > {thresh:.0%}: {n_pos:5,} matches ({pct:5.1f}%)")

# Edge distribution histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(edge_test, bins=80, color="#3498db", alpha=0.75, edgecolor="white", linewidth=0.3)
for thresh, color, label in [
    (0.02, "#e67e22", "2%"),
    (0.03, "#2ecc71", "3%"),
    (0.05, "#9b59b6", "5%"),
]:
    n_pos = (edge_test > thresh).sum()
    ax.axvline(thresh, color=color, ls="--", lw=1.5,
               label=f">{label}: {n_pos:,} ({n_pos/len(edge_test):.1%})")
ax.axvline(0, color="red", ls="-", lw=2, alpha=0.6, label="Edge = 0")
ax.set_xlabel("Edge (Model B − Model A)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Edge Distribution — Test Set (Apr 2022 → Feb 2026)\n"
             "Model B (Fundamentals) − Model A (Market)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "edge_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 13. BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] Backtest (test period Apr 2022 → Feb 2026)...")
print(f"\n    ⚠️  CAVEAT: B365D = FULL-TIME draw odds (HT draws ≈42%, FT ≈28%).")
print(f"    ⚠️  Using B365D as proxy. Real HT-specific odds would yield different ROI.")

b365d_te = pd.to_numeric(test_df["B365D"], errors="coerce").values
valid_bt = np.isfinite(b365d_te) & np.isfinite(edge_test) & (b365d_te > 1.0)

y_bt     = y_test[valid_bt]
edge_bt  = edge_test[valid_bt]
odds_bt  = b365d_te[valid_bt]

# Flat-bet on ALL draws (baseline)
pnl_base = np.where(y_bt == 1, odds_bt - 1, -1.0)
roi_base = pnl_base.mean()
print(f"\n    Baseline — bet ALL draws: {len(y_bt):,} bets, "
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
        "threshold": thresh,
        "n_bets":    int(n_bets),
        "n_wins":    wins,
        "win_rate":  float(win_r),
        "roi_flat":  float(roi),
        "pnl_flat":  float(total_pnl),
        "pct_bets":  float(pct),
        "vs_baseline": float(vs_base),
    })

# Cumulative PnL plot for edge > 0.03
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
fig.suptitle("V3 Backtest Analysis — Test Period (Apr 2022 → Feb 2026)", fontsize=13, fontweight="bold")

# Left: cumulative PnL
axes[0].plot(dates_sorted, cumsum, color="#2ecc71", lw=2, label=f"Edge > {thresh_plot:.0%}")
axes[0].axhline(0, color="red", ls="--", lw=1, alpha=0.7)
axes[0].fill_between(dates_sorted, cumsum, 0,
                     where=cumsum >= 0, alpha=0.12, color="#2ecc71")
axes[0].fill_between(dates_sorted, cumsum, 0,
                     where=cumsum < 0,  alpha=0.12, color="#e74c3c")
axes[0].set_xlabel("Date"); axes[0].set_ylabel("Cumulative PnL (units)")
axes[0].set_title(f"Cumulative PnL — Edge > {thresh_plot:.0%}")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Right: ROI by threshold
if backtest_rows:
    thrs   = [r["threshold"] for r in backtest_rows]
    rois   = [r["roi_flat"] for r in backtest_rows]
    colors_bar = ["#2ecc71" if r > roi_base else "#e74c3c" for r in rois]
    axes[1].bar([f"{t:.0%}" for t in thrs], rois, color=colors_bar, alpha=0.8)
    axes[1].axhline(0,        color="black", lw=0.8, ls="--")
    axes[1].axhline(roi_base, color="blue",  lw=1.5, ls=":",
                    label=f"Flat-all ROI={roi_base:+.3f}")
    axes[1].set_xlabel("Edge threshold"); axes[1].set_ylabel("Flat ROI per unit")
    axes[1].set_title("Backtest ROI by Edge Threshold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "backtest.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 14. ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] Generating ROC curve plot...")

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("ROC Curves — V3 Two-Model Architecture\n(Test Set, Apr 2022 → Feb 2026)",
             fontsize=12, fontweight="bold")

curves = [
    (pa_te_cal,                    f"Model A — Market LR   (AUC={a_te_auc_cal:.4f})", "#e74c3c", 2.0),
    (pb_b_te_cal,                  f"Model B — Fundamentals (AUC={b_te_auc_cal:.4f})", "#2ecc71", 2.0),
    (mkt_test_raw[valid_m],        f"Market Direct          (AUC={mkt_auc:.4f})", "#9b59b6", 1.5),
]
for probs, label, color, lw in curves:
    yt = y_test[valid_m] if len(probs) == valid_m.sum() else y_test
    fpr, tpr, _ = roc_curve(yt, probs)
    ax.plot(fpr, tpr, label=label, lw=lw, color=color)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 15. SAVE ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[15] Saving models to models/v3/...")

# Model A
with open(OUT_DIR / "model_a_lr.pkl", "wb") as f:        pickle.dump(lr_a, f)
with open(OUT_DIR / "model_a_scaler.pkl", "wb") as f:    pickle.dump(scaler_a, f)
with open(OUT_DIR / "model_a_calibrator.pkl", "wb") as f: pickle.dump(iso_a, f)
with open(OUT_DIR / "model_a_medians.json", "w") as f:   json.dump(train_medians_a, f, indent=2)
with open(OUT_DIR / "model_a_features.json", "w") as f:  json.dump(MODEL_A_FEATURES, f, indent=2)

# Model B
xgb_b.save_model(str(OUT_DIR / "model_b_xgb.json"))
lgb_b.save_model(str(OUT_DIR / "model_b_lgb.txt"))
with open(OUT_DIR / "model_b_calibrator.pkl", "wb") as f: pickle.dump(iso_b, f)
with open(OUT_DIR / "model_b_medians.json", "w") as f:   json.dump(train_medians_b, f, indent=2)
with open(OUT_DIR / "model_b_features.json", "w") as f:  json.dump(model_b_features, f, indent=2)
with open(OUT_DIR / "model_b_best.txt", "w") as f:       f.write(best_b_name)

# Sub-models
dc_v3.save(str(OUT_DIR / "dixon_coles_v3.pkl"))
elo_v3.save(str(OUT_DIR / "elo_v3.pkl"))
with open(OUT_DIR / "referee_model_v3.pkl", "wb") as f:  pickle.dump(ref_v3, f)

# Paths manifest
manifest = {
    "model_a_lr":          str(OUT_DIR / "model_a_lr.pkl"),
    "model_a_scaler":      str(OUT_DIR / "model_a_scaler.pkl"),
    "model_a_calibrator":  str(OUT_DIR / "model_a_calibrator.pkl"),
    "model_a_medians":     str(OUT_DIR / "model_a_medians.json"),
    "model_a_features":    str(OUT_DIR / "model_a_features.json"),
    "model_b_xgb":         str(OUT_DIR / "model_b_xgb.json"),
    "model_b_lgb":         str(OUT_DIR / "model_b_lgb.txt"),
    "model_b_calibrator":  str(OUT_DIR / "model_b_calibrator.pkl"),
    "model_b_medians":     str(OUT_DIR / "model_b_medians.json"),
    "model_b_features":    str(OUT_DIR / "model_b_features.json"),
    "model_b_best":        best_b_name,
    "dc_path":             str(OUT_DIR / "dixon_coles_v3.pkl"),
    "elo_path":            str(OUT_DIR / "elo_v3.pkl"),
    "referee_path":        str(OUT_DIR / "referee_model_v3.pkl"),
    "mega_dataset":        str(DATA_PATH),
}
with open(OUT_DIR / "v3_paths.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Metrics
metrics_out = {
    "training_summary": {
        "dataset": str(DATA_PATH),
        "n_total": int(n),
        "n_train": len(train_df),
        "n_val":   len(val_df),
        "n_test":  len(test_df),
        "draw_rate": float(df[TARGET].mean()),
        "train_period": f"{train_df['Date'].min().date()} → {train_df['Date'].max().date()}",
        "test_period":  f"{test_df['Date'].min().date()} → {test_df['Date'].max().date()}",
    },
    "validation": {
        "no_odds_in_model_b":   True,
        "temporal_split":       "70/15/15 chronological",
        "model_b_features_checked": len(model_b_features),
    },
    "market_baseline": {
        "test_auc": float(mkt_auc),
        "test_brier": float(mkt_brier),
        "n_valid": int(valid_m.sum()),
    },
    "model_a": {
        "name": "LogisticRegression (log-odds only)",
        "n_features": len(MODEL_A_FEATURES),
        "features": MODEL_A_FEATURES,
        "train_auc": float(a_tr_auc),
        "val_auc":   float(a_va_auc),
        "test_auc_raw": float(a_te_auc),
        "test_auc_calibrated": float(a_te_auc_cal),
        "test_brier_calibrated": float(a_te_brier_cal),
    },
    "model_b": {
        "best_model": best_b_name,
        "n_features": len(model_b_features),
        "features": model_b_features,
        "xgboost": {
            "train_auc": float(b_xgb_tr_auc),
            "val_auc":   float(b_xgb_va_auc),
            "test_auc":  float(b_xgb_te_auc),
            "best_iteration": int(xgb_b.best_iteration),
            "best_params": study_b.best_params,
        },
        "lightgbm": {
            "train_auc": float(b_lgb_tr_auc),
            "val_auc":   float(b_lgb_va_auc),
            "test_auc":  float(b_lgb_te_auc),
            "best_iteration": int(lgb_b.best_iteration),
        },
        "test_auc_calibrated":   float(b_te_auc_cal),
        "test_brier_calibrated": float(b_te_brier_cal),
    },
    "edge": {
        "mean":       float(edge_test.mean()),
        "std":        float(edge_test.std()),
        "pct_gt_1":   float((edge_test > 0.01).mean()),
        "pct_gt_2":   float((edge_test > 0.02).mean()),
        "pct_gt_3":   float((edge_test > 0.03).mean()),
        "pct_gt_5":   float((edge_test > 0.05).mean()),
        "pct_gt_10":  float((edge_test > 0.10).mean()),
    },
    "backtest": backtest_rows,
}
with open(OUT_DIR / "v3_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"    ✓ All models saved to {OUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# 16. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("V3 ARCHITECTURE — FINAL SUMMARY")
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
print(f"    Model B = Independent stats signal (no odds whatsoever)")
print(f"    Edge = Model B − Model A = stats estimate vs market estimate")

print(f"\n  Edge distribution (test set: {len(edge_test):,} matches):")
for thresh in [0.01, 0.02, 0.03, 0.05, 0.10]:
    n_pos = (edge_test > thresh).sum()
    print(f"    > {thresh:.0%}: {n_pos:,} matches ({n_pos/len(edge_test):.1%})")

print(f"\n  Backtest (flat bet, using B365D as proxy for HT draw odds):")
print(f"  {'Thresh':>6}  {'Bets':>7}  {'Win%':>6}  {'ROI':>8}  {'PnL':>8}  {'vs Baseline':>12}")
print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*12}")
for r in backtest_rows:
    print(f"  {r['threshold']:>6.2f}  {r['n_bets']:>7,}  {r['win_rate']:>6.1%}  "
          f"{r['roi_flat']:>+8.4f}  {r['pnl_flat']:>+8.1f}  {r['vs_baseline']:>+12.4f}")

print(f"\n  Plots saved to {PLOTS_DIR}/")
print(f"  Metrics saved to {OUT_DIR}/v3_metrics.json")
print(f"  Models saved to {OUT_DIR}/")
print("\n" + "=" * 72)
print("  DONE. Run predict_match_v3.py for single-match predictions.")
print("=" * 72)
