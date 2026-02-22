"""
Mega Dataset Training Pipeline — Half-Time Draw Prediction
==========================================================
Trains on data/processed/mega_dataset.parquet (193k matches, 22 leagues, ~70 features).

Models trained:
  1. Logistic Regression (baseline)
  2. XGBoost (Optuna-tuned, 50 trials)
  3. LightGBM (gradient boosting with SHAP feature importance)

Key features:
  - Features with >=60% coverage: impute with median; below threshold: dropped
  - Temporal split 70/15/15 chronological
  - Calibration analysis (Platt scaling on val set)
  - Backtesting vs B365 implied probabilities (flat-bet + Kelly)
  - Comparison table: old 12-feature EFL-only vs new mega results vs market baseline
  - Saves models/mega/metrics_mega.json
"""

# Standard library
import json
import pickle
import warnings
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, roc_curve,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/mega_dataset.parquet")
OUT_DIR   = Path("models/mega")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET             = "y_ht_draw"
COVERAGE_THRESHOLD = 0.60
TODAY              = pd.Timestamp("2026-02-22")

# Non-feature columns: identifiers, raw in-game stats, target, raw odds
EXCLUDE = {
    "Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG", "FTHG", "FTAG",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
    "league", "season", "country", "league_tier", "league_name",
    TARGET,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 3 — MEGA DATASET TRAINING PIPELINE")
print("=" * 70)

print("\n[1] Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"    Raw shape: {df.shape}")

future_mask = df["Date"] > TODAY
if future_mask.sum():
    print(f"    Removing {future_mask.sum()} future-date rows")
    df = df[~future_mask].copy()

null_target = df[TARGET].isna().sum()
if null_target:
    print(f"    Removing {null_target} null-target rows")
    df = df.dropna(subset=[TARGET])

df = df.sort_values("Date").reset_index(drop=True)
print(f"    Clean shape: {df.shape}")
print(f"    Date range:  {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"    Draw rate:   {df[TARGET].mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION (coverage threshold)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Feature selection...")
all_candidate_cols = [c for c in df.columns if c not in EXCLUDE]
coverage = {c: df[c].notna().mean() for c in all_candidate_cols}
selected_features = [c for c, cov in coverage.items() if cov >= COVERAGE_THRESHOLD]
dropped_features  = [c for c, cov in coverage.items() if cov < COVERAGE_THRESHOLD]

print(f"    Candidate features:   {len(all_candidate_cols)}")
print(f"    Features kept (≥60%): {len(selected_features)}")
print(f"    Features dropped:     {len(dropped_features)}")
if dropped_features:
    print(f"    Dropped: {dropped_features}")

print(f"\n    Selected features (coverage):")
for col in selected_features:
    print(f"      {col:<45} {coverage[col]*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL SPLIT 70 / 15 / 15
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Temporal split 70/15/15...")
n         = len(df)
train_end = int(0.70 * n)
val_end   = train_end + int(0.15 * n)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

print(f"    Train: {len(train_df):,}  {train_df['Date'].min().date()} → {train_df['Date'].max().date()}")
print(f"    Val:   {len(val_df):,}  {val_df['Date'].min().date()} → {val_df['Date'].max().date()}")
print(f"    Test:  {len(test_df):,}  {test_df['Date'].min().date()} → {test_df['Date'].max().date()}")

# Medians from TRAINING SET only (no leakage)
train_medians = train_df[selected_features].median()

def impute(split_df):
    out = split_df[selected_features].copy()
    for col in selected_features:
        out[col] = out[col].fillna(train_medians[col])
    return out.values.astype(np.float32)

X_train = impute(train_df); y_train = train_df[TARGET].values.astype(float)
X_val   = impute(val_df);   y_val   = val_df[TARGET].values.astype(float)
X_test  = impute(test_df);  y_test  = test_df[TARGET].values.astype(float)

print(f"\n    Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

# Scale (fit on train only)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

with open(OUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(OUT_DIR / "medians.json", "w") as f:
    json.dump({k: float(v) for k, v in train_medians.items()}, f, indent=2)
with open(OUT_DIR / "selected_features.json", "w") as f:
    json.dump(selected_features, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MARKET BASELINE (B365 implied probability)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Market baseline (B365 normalized implied probability)...")

def safe_market_prob(df_split):
    """Return normalized implied P(draw) for rows that have B365 odds; NaN rows masked out."""
    b365h = df_split["B365H"].values.astype(float)
    b365d = df_split["B365D"].values.astype(float)
    b365a = df_split["B365A"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_draw = 1.0 / b365d
        overround    = (1.0/b365h) + (1.0/b365d) + (1.0/b365a)
        prob = implied_draw / overround
    return prob

test_market_prob_raw = safe_market_prob(test_df)
market_valid = np.isfinite(test_market_prob_raw) & (test_market_prob_raw > 0)
test_market_prob = test_market_prob_raw[market_valid]
y_test_market    = y_test[market_valid]

market_auc   = roc_auc_score(y_test_market, test_market_prob)
market_brier = brier_score_loss(y_test_market, test_market_prob)
print(f"    Market (B365) — test AUC: {market_auc:.4f}, Brier: {market_brier:.4f}  (n={market_valid.sum():,})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL 1: LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Training Logistic Regression...")
lr_model = LogisticRegression(C=0.1, max_iter=2000, random_state=42, solver="lbfgs", n_jobs=-1)
lr_model.fit(X_train_s, y_train)

lr_preds = {
    "train": lr_model.predict_proba(X_train_s)[:, 1],
    "val":   lr_model.predict_proba(X_val_s)[:, 1],
    "test":  lr_model.predict_proba(X_test_s)[:, 1],
}
lr_metrics = {
    "train_auc":     roc_auc_score(y_train, lr_preds["train"]),
    "val_auc":       roc_auc_score(y_val,   lr_preds["val"]),
    "test_auc":      roc_auc_score(y_test,  lr_preds["test"]),
    "train_brier":   brier_score_loss(y_train, lr_preds["train"]),
    "val_brier":     brier_score_loss(y_val,   lr_preds["val"]),
    "test_brier":    brier_score_loss(y_test,  lr_preds["test"]),
    "train_logloss": log_loss(y_train, lr_preds["train"]),
    "val_logloss":   log_loss(y_val,   lr_preds["val"]),
    "test_logloss":  log_loss(y_test,  lr_preds["test"]),
}
print(f"    LR — train AUC: {lr_metrics['train_auc']:.4f}, val AUC: {lr_metrics['val_auc']:.4f}, test AUC: {lr_metrics['test_auc']:.4f}")
with open(OUT_DIR / "lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL 2: XGBOOST (Optuna-tuned, 50 trials)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Training XGBoost (Optuna, 50 trials)...")

def xgb_objective(trial):
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "tree_method":      "hist",
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "eta":              trial.suggest_float("eta", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "alpha":            trial.suggest_float("alpha", 0.0, 5.0),
        "lambda":           trial.suggest_float("lambda", 0.5, 10.0),
        "n_jobs":           -1,
        "seed":             42,
        "verbosity":        0,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    bst = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    preds_val = bst.predict(dval)
    return roc_auc_score(y_val, preds_val)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

print(f"    Best val AUC: {study.best_value:.4f}  params: {study.best_params}")

best_xgb_params = {
    "objective":   "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs":      -1,
    "seed":        42,
    "verbosity":   0,
    **study.best_params,
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

xgb_model = xgb.train(
    best_xgb_params, dtrain,
    num_boost_round=1000,
    evals=[(dval, "val")],
    early_stopping_rounds=30,
    verbose_eval=False,
)
print(f"    Best iteration: {xgb_model.best_iteration}")

xgb_preds = {
    "train": xgb_model.predict(dtrain),
    "val":   xgb_model.predict(dval),
    "test":  xgb_model.predict(dtest),
}
xgb_metrics = {
    "train_auc":      roc_auc_score(y_train, xgb_preds["train"]),
    "val_auc":        roc_auc_score(y_val,   xgb_preds["val"]),
    "test_auc":       roc_auc_score(y_test,  xgb_preds["test"]),
    "train_brier":    brier_score_loss(y_train, xgb_preds["train"]),
    "val_brier":      brier_score_loss(y_val,   xgb_preds["val"]),
    "test_brier":     brier_score_loss(y_test,  xgb_preds["test"]),
    "train_logloss":  log_loss(y_train, xgb_preds["train"]),
    "val_logloss":    log_loss(y_val,   xgb_preds["val"]),
    "test_logloss":   log_loss(y_test,  xgb_preds["test"]),
    "best_iteration": xgb_model.best_iteration,
    "best_params":    study.best_params,
}
print(f"    XGB — train AUC: {xgb_metrics['train_auc']:.4f}, val AUC: {xgb_metrics['val_auc']:.4f}, test AUC: {xgb_metrics['test_auc']:.4f}")
xgb_model.save_model(str(OUT_DIR / "xgb_model.json"))

# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL 3: LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Training LightGBM...")
lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=selected_features)
lgb_val   = lgb.Dataset(X_val,   label=y_val,   reference=lgb_train)

lgb_params = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)
print(f"    Best iteration: {lgb_model.best_iteration}")

lgb_preds = {
    "train": lgb_model.predict(X_train),
    "val":   lgb_model.predict(X_val),
    "test":  lgb_model.predict(X_test),
}
lgb_metrics = {
    "train_auc":      roc_auc_score(y_train, lgb_preds["train"]),
    "val_auc":        roc_auc_score(y_val,   lgb_preds["val"]),
    "test_auc":       roc_auc_score(y_test,  lgb_preds["test"]),
    "train_brier":    brier_score_loss(y_train, lgb_preds["train"]),
    "val_brier":      brier_score_loss(y_val,   lgb_preds["val"]),
    "test_brier":     brier_score_loss(y_test,  lgb_preds["test"]),
    "train_logloss":  log_loss(y_train, lgb_preds["train"]),
    "val_logloss":    log_loss(y_val,   lgb_preds["val"]),
    "test_logloss":   log_loss(y_test,  lgb_preds["test"]),
    "best_iteration": lgb_model.best_iteration,
}
print(f"    LGB — train AUC: {lgb_metrics['train_auc']:.4f}, val AUC: {lgb_metrics['val_auc']:.4f}, test AUC: {lgb_metrics['test_auc']:.4f}")
lgb_model.save_model(str(OUT_DIR / "lgbm_model.txt"))

# ── SHAP Feature Importance (LightGBM) ────────────────────────────────────────
print("\n    Computing SHAP values (LightGBM, sample 3000 test rows)...")
sample_idx    = np.random.choice(len(X_test), min(3000, len(X_test)), replace=False)
X_test_sample = X_test[sample_idx]
explainer     = shap.TreeExplainer(lgb_model)
shap_values   = explainer.shap_values(X_test_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature":       selected_features,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)

print(f"\n    Top-15 Features by SHAP (LightGBM):")
print(f"    {'Feature':<45} {'Mean |SHAP|':>12}")
print(f"    {'-'*45} {'-'*12}")
for _, row in shap_df.head(15).iterrows():
    print(f"    {row['feature']:<45} {row['mean_abs_shap']:>12.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CALIBRATION (Platt scaling — fit on val, apply to test)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Calibration (Platt scaling on val set)...")

def platt_calibrate(raw_val_preds, y_val_true, raw_test_preds):
    cal = LogisticRegression(C=1.0, random_state=42)
    cal.fit(raw_val_preds.reshape(-1, 1), y_val_true)
    return cal.predict_proba(raw_test_preds.reshape(-1, 1))[:, 1], cal

lgb_cal_test,  lgb_cal_model  = platt_calibrate(lgb_preds["val"],  y_val, lgb_preds["test"])
xgb_cal_test,  xgb_cal_model  = platt_calibrate(xgb_preds["val"],  y_val, xgb_preds["test"])
lr_cal_test,   lr_cal_model   = platt_calibrate(lr_preds["val"],   y_val, lr_preds["test"])

cal_metrics = {
    "lr_calibrated_test_auc":    roc_auc_score(y_test, lr_cal_test),
    "lr_calibrated_test_brier":  brier_score_loss(y_test, lr_cal_test),
    "xgb_calibrated_test_auc":   roc_auc_score(y_test, xgb_cal_test),
    "xgb_calibrated_test_brier": brier_score_loss(y_test, xgb_cal_test),
    "lgb_calibrated_test_auc":   roc_auc_score(y_test, lgb_cal_test),
    "lgb_calibrated_test_brier": brier_score_loss(y_test, lgb_cal_test),
}
print(f"    LR  (calibrated) — AUC: {cal_metrics['lr_calibrated_test_auc']:.4f}, Brier: {cal_metrics['lr_calibrated_test_brier']:.4f}")
print(f"    XGB (calibrated) — AUC: {cal_metrics['xgb_calibrated_test_auc']:.4f}, Brier: {cal_metrics['xgb_calibrated_test_brier']:.4f}")
print(f"    LGB (calibrated) — AUC: {cal_metrics['lgb_calibrated_test_auc']:.4f}, Brier: {cal_metrics['lgb_calibrated_test_brier']:.4f}")

with open(OUT_DIR / "lgb_calibrator.pkl", "wb") as f:
    pickle.dump(lgb_cal_model, f)
with open(OUT_DIR / "xgb_calibrator.pkl", "wb") as f:
    pickle.dump(xgb_cal_model, f)

# ─────────────────────────────────────────────────────────────────────────────
# 9. BACKTESTING vs B365 MARKET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Backtesting (LGB calibrated vs B365, flat-bet + fractional Kelly)...")

# Use calibrated LGB as primary betting model
# Restrict to rows where we have valid B365 odds
model_prob_full  = lgb_cal_test                    # all test rows
market_prob_full = test_market_prob_raw            # raw (may contain NaN)

# Only bet where both model and market probs are finite
valid_idx = np.where(market_valid)[0]
model_prob_bt  = model_prob_full[valid_idx]
market_prob_bt = test_market_prob
y_bt           = y_test[valid_idx]
odds_bt        = test_df["B365D"].values[valid_idx].astype(float)

edge_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
backtest_results = []

print(f"\n    {'Edge':>6}  {'Bets':>7}  {'Bet%':>5}  {'Wins':>5}  {'Win%':>6}  "
      f"{'FlatROI':>8}  {'Kelly25%ROI':>12}  {'PnL(flat)':>10}")
print(f"    {'-'*6}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*6}  "
      f"{'-'*8}  {'-'*12}  {'-'*10}")

for edge_thresh in edge_thresholds:
    bet_mask = (model_prob_bt - market_prob_bt) > edge_thresh
    n_bets   = bet_mask.sum()
    if n_bets == 0:
        continue

    y_b    = y_bt[bet_mask]
    odds_b = odds_bt[bet_mask]
    m_b    = market_prob_bt[bet_mask]
    p_b    = model_prob_bt[bet_mask]
    wins   = y_b.sum()
    win_pct = wins / n_bets

    # Flat bet P&L
    pnl_flat  = np.where(y_b == 1, odds_b - 1, -1.0)
    roi_flat  = pnl_flat.sum() / n_bets
    total_pnl = pnl_flat.sum()

    # Fractional Kelly (25% of full Kelly) — stake = f * bankroll per bet
    # Kelly fraction = (p*b - (1-p)) / b  where b = odds - 1
    b_dec   = odds_b - 1.0
    kelly_f = (p_b * b_dec - (1.0 - p_b)) / b_dec
    kelly_f = np.clip(kelly_f, 0, None) * 0.25   # quarter-Kelly, no negative bets
    # P&L = kelly_f * (odds-1) for wins, -kelly_f for losses
    pnl_kelly = np.where(y_b == 1, kelly_f * b_dec, -kelly_f)
    # ROI = total P&L / total staked
    total_staked_kelly = kelly_f.sum()
    roi_kelly = pnl_kelly.sum() / total_staked_kelly if total_staked_kelly > 0 else 0.0

    bet_pct = n_bets / len(y_bt) * 100
    print(f"    {edge_thresh:>6.2f}  {n_bets:>7,}  {bet_pct:>4.1f}%  {int(wins):>5}  {win_pct:>6.1%}  "
          f"{roi_flat:>8.3f}  {roi_kelly:>12.3f}  {total_pnl:>+10.1f}")

    backtest_results.append({
        "edge_threshold":        edge_thresh,
        "n_bets":                int(n_bets),
        "n_wins":                int(wins),
        "win_rate":              float(win_pct),
        "roi_flat":              float(roi_flat),
        "roi_kelly_25pct":       float(roi_kelly),
        "total_pnl_flat_units":  float(total_pnl),
        "bet_pct":               float(bet_pct),
    })

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Generating plots...")

# A) ROC Curves (test set: LR, XGB, LGB, Market)
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("ROC Curves — Mega Dataset Test Set\n(193k matches, 22 leagues)", fontsize=12, fontweight="bold")

for probs, label, color, lw in [
    (lr_preds["test"],  f"LR   (AUC={lr_metrics['test_auc']:.4f})",  "#e74c3c", 2.0),
    (xgb_preds["test"], f"XGB  (AUC={xgb_metrics['test_auc']:.4f})", "#f39c12", 2.0),
    (lgb_preds["test"], f"LGB  (AUC={lgb_metrics['test_auc']:.4f})", "#2ecc71", 2.0),
    (test_market_prob,  f"Mkt  (AUC={market_auc:.4f})",             "#9b59b6", 1.5),
]:
    yt = y_test_market if len(probs) == len(y_test_market) else y_test
    fpr, tpr, _ = roc_curve(yt, probs)
    ax.plot(fpr, tpr, label=label, lw=lw, color=color)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# B) SHAP bar plot
fig, ax = plt.subplots(figsize=(10, 8))
top15 = shap_df.head(15).iloc[::-1]
ax.barh(range(15), top15["mean_abs_shap"].values, color="#2196F3")
ax.set_yticks(range(15))
ax.set_yticklabels(top15["feature"].values)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("LightGBM Feature Importance (SHAP)\nTop 15 Features — Mega Dataset", fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# C) Calibration plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Calibration: Before vs After Platt Scaling (Test Set)", fontsize=12, fontweight="bold")

for ax, (raw, cal, label) in zip(axes, [
    (lgb_preds["test"], lgb_cal_test, "LightGBM"),
    (xgb_preds["test"], xgb_cal_test, "XGBoost"),
    (lr_preds["test"],  lr_cal_test,  "Logistic Regression"),
]):
    for probs, name, ls in [(raw, "Raw", "-"), (cal, "Platt-calibrated", "--")]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", ls=ls, label=name, lw=2)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Perfect")
    ax.set_title(f"{label}")
    ax.set_xlabel("Mean predicted P(draw)")
    ax.set_ylabel("Fraction actual draws")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "calibration.png", dpi=150, bbox_inches="tight")
plt.close()

# D) Backtest ROI
if backtest_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Backtest Results: LGB (Platt-calibrated) vs B365 Market", fontsize=12, fontweight="bold")
    thresholds   = [r["edge_threshold"] for r in backtest_results]
    rois_flat    = [r["roi_flat"] for r in backtest_results]
    rois_kelly   = [r["roi_kelly_25pct"] for r in backtest_results]
    n_bets_list  = [r["n_bets"] for r in backtest_results]
    x = range(len(thresholds))
    ax1.bar([str(t) for t in thresholds], rois_flat,
            color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rois_flat], alpha=0.7, label="Flat ROI")
    ax1.bar([str(t) for t in thresholds], rois_kelly,
            color=["#27ae60" if r >= 0 else "#c0392b" for r in rois_kelly], alpha=0.5, label="Kelly 25% ROI")
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_xlabel("Edge threshold (model − market)")
    ax1.set_ylabel("ROI per unit staked")
    ax1.set_title("Return on Investment")
    ax1.legend()
    ax2.bar([str(t) for t in thresholds], n_bets_list, color="#3498db")
    ax2.set_xlabel("Edge threshold")
    ax2.set_ylabel("Number of bets")
    ax2.set_title("Number of Bets by Edge Threshold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

print("    Plots saved to models/mega/plots/")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE ALL METRICS
# ─────────────────────────────────────────────────────────────────────────────
all_metrics = {
    "dataset": {
        "path":                 str(DATA_PATH),
        "total_rows":           int(len(df)),
        "train_rows":           int(len(train_df)),
        "val_rows":             int(len(val_df)),
        "test_rows":            int(len(test_df)),
        "n_features_selected":  len(selected_features),
        "selected_features":    selected_features,
        "dropped_features":     dropped_features,
        "draw_rate":            float(df[TARGET].mean()),
    },
    "market_baseline": {
        "test_auc":   float(market_auc),
        "test_brier": float(market_brier),
        "n_valid":    int(market_valid.sum()),
    },
    "logistic_regression": lr_metrics,
    "xgboost": xgb_metrics,
    "lightgbm": lgb_metrics,
    "calibration": cal_metrics,
    "backtest_lgb_calibrated": backtest_results,
    "shap_top15": shap_df.head(15).to_dict(orient="records"),
}

with open(OUT_DIR / "metrics_mega.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n    Saved models/mega/metrics_mega.json")

# ─────────────────────────────────────────────────────────────────────────────
# 12. FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("FINAL COMPARISON TABLE")
print("=" * 72)

# Load old metrics (12-feature EFL-only pipeline)
try:
    with open("models/metrics.json") as f:
        old_metrics = json.load(f)
    old_lr_auc    = old_metrics["logistic_regression"]["test_auc"]
    old_lr_brier  = old_metrics["logistic_regression"]["test_brier"]
    old_lstm_auc  = old_metrics["lstm"]["test_auc"]
    old_lstm_brier= old_metrics["lstm"]["test_brier"]
    has_old = True
except Exception:
    has_old = False

HDR = f"{'Model':<40} {'Test AUC':>9} {'Test Brier':>11} {'vs Market':>10}"
print(HDR)
print("-" * len(HDR))

def mkt_delta(auc):
    d = auc - market_auc
    return f"{d:>+10.4f}"

if has_old:
    print(f"{'OLD: LR (12-feat, EFL only)':<40} {old_lr_auc:>9.4f} {old_lr_brier:>11.5f} {mkt_delta(old_lr_auc):>10}")
    print(f"{'OLD: LSTM seq1 (12-feat, EFL)':<40} {old_lstm_auc:>9.4f} {old_lstm_brier:>11.5f} {mkt_delta(old_lstm_auc):>10}")
    print("-" * len(HDR))

rows = [
    ("NEW: LR (mega, full feat)",         lr_metrics["test_auc"],  lr_metrics["test_brier"]),
    ("NEW: XGB Optuna (mega, full feat)", xgb_metrics["test_auc"], xgb_metrics["test_brier"]),
    ("NEW: LGB (mega, full feat)",        lgb_metrics["test_auc"], lgb_metrics["test_brier"]),
    ("NEW: LGB + Platt calibration",      cal_metrics["lgb_calibrated_test_auc"], cal_metrics["lgb_calibrated_test_brier"]),
    ("NEW: XGB + Platt calibration",      cal_metrics["xgb_calibrated_test_auc"], cal_metrics["xgb_calibrated_test_brier"]),
    ("MARKET: B365 implied P(draw)",      market_auc, market_brier),
]

for label, auc, brier in rows:
    print(f"{label:<40} {auc:>9.4f} {brier:>11.5f} {mkt_delta(auc):>10}")

best_auc = max(r[1] for r in rows[:-1])
print("-" * len(HDR))
print(f"\n  Market AUC:      {market_auc:.4f}")
print(f"  Best model AUC:  {best_auc:.4f}")
print(f"  AUC gap vs mkt:  {best_auc - market_auc:+.4f}")

print("\n  Backtest summary (LGB calibrated, vs B365, flat-bet ROI):")
print(f"  {'Edge':>6}  {'Bets':>7}  {'Win%':>6}  {'FlatROI':>8}  {'Kelly25%':>9}  {'PnL':>8}")
print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}")
for r in backtest_results:
    print(f"  {r['edge_threshold']:>6.2f}  {r['n_bets']:>7,}  {r['win_rate']:>6.1%}  "
          f"{r['roi_flat']:>8.3f}  {r['roi_kelly_25pct']:>9.3f}  {r['total_pnl_flat_units']:>+8.1f}")

print("\n  SHAP Top-10 features (LightGBM):")
for _, row in shap_df.head(10).iterrows():
    print(f"    {row['feature']:<45} {row['mean_abs_shap']:.5f}")

print("\n" + "=" * 72)
print("  All models trained. Outputs in models/mega/")
print("  metrics_mega.json saved.")
print("=" * 72)
