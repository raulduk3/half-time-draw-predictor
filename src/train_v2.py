"""
V2 Ensemble Training Pipeline — Half-Time Draw Prediction
==========================================================
Trains and evaluates the full v2 ensemble:
  1. Dixon-Coles bivariate Poisson (per-league, HT scores)
  2. Elo rating system (HT results)
  3. Referee impact model (EFL raw CSV data)
  4. Market model (multi-bookmaker features)
  5. Stacking meta-learner (LogisticRegression)
  6. Isotonic calibration

Architecture:
  - Temporal split 70 / 15 / 15 on mega dataset
  - Sub-models fitted on TRAIN set only
  - Meta-learner trained on VAL set predictions (no leakage)
  - Final evaluation on held-out TEST set
  - Loads pre-trained XGB + LGB from models/mega/

Outputs → models/v2/
  dixon_coles.pkl, elo.pkl, referee_model.pkl,
  market_model.pkl, ensemble_weights.pkl, ensemble_metrics.json
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MEGA_PARQUET    = Path("data/processed/mega_dataset_v2.parquet")  # expanded with MLS/MEX
MEGA_PARQUET_V1 = Path("data/processed/mega_dataset.parquet")     # fallback
RAW_DIR      = Path("data/raw")
MEGA_MODELS  = Path("models/mega")
OUT_DIR      = Path("models/v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Leagues that have only FT data (no HTHG/HTAG available)
FT_ONLY_LEAGUES = ["USA_MLS", "MEX_LigaMX"]

# ── Imports (project) ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, ".")
from src.dixon_coles    import DixonColesEnsemble
from src.elo            import EloRatingSystem, tune_k_factor
from src.referee_model  import RefereeModel, load_efl_raw
from src.market_model   import MarketModel, extract_multi_book_features
from src.ensemble_predictor import EnsemblePredictor

print("=" * 72)
print("V2 ENSEMBLE TRAINING PIPELINE — Half-Time Draw Prediction")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MEGA DATASET + TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading mega dataset...")

p = MEGA_PARQUET if MEGA_PARQUET.exists() else MEGA_PARQUET_V1
full_df = pd.read_parquet(p).sort_values("Date").reset_index(drop=True)
print(f"  Loaded {len(full_df):,} rows from {p.name}")
ft_only_mask = full_df["league"].isin(FT_ONLY_LEAGUES)
print(f"  FT-only rows (MLS/MEX): {ft_only_mask.sum():,}")

TODAY = pd.Timestamp("2026-02-22")
full_df = full_df[full_df["Date"] <= TODAY].copy()

# HT-only subset — used for temporal split, targets, XGB/LGB, evaluation
df = full_df.dropna(subset=["HTHG", "HTAG"]).sort_values("Date").reset_index(drop=True)

n         = len(df)
train_end = int(0.70 * n)
val_end   = train_end + int(0.15 * n)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

# Full training slice (includes FT-only rows) — for DC and Elo
TRAIN_CUTOFF = train_df["Date"].max()
full_train_df = full_df[full_df["Date"] <= TRAIN_CUTOFF].copy()
print(f"  Full train slice (HT+FT rows): {len(full_train_df):,} "
      f"(HT: {full_train_df['HTHG'].notna().sum():,}, "
      f"FT-only: {full_train_df['ft_only'].eq(True).sum() if 'ft_only' in full_train_df.columns else 0:,})")

y_train = (train_df["HTHG"] == train_df["HTAG"]).astype(int).values
y_val   = (val_df["HTHG"] == val_df["HTAG"]).astype(int).values
y_test  = (test_df["HTHG"] == test_df["HTAG"]).astype(int).values

GLOBAL_DRAW_RATE = float(y_train.mean())

print(f"  HT-only rows: {n:,}")
print(f"  Train:  {len(train_df):,}  {train_df['Date'].min().date()} → {train_df['Date'].max().date()}")
print(f"  Val:    {len(val_df):,}  {val_df['Date'].min().date()} → {val_df['Date'].max().date()}")
print(f"  Test:   {len(test_df):,}  {test_df['Date'].min().date()} → {test_df['Date'].max().date()}")
print(f"  Global draw rate: {GLOBAL_DRAW_RATE:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD EFL RAW DATA + ENRICH WITH REFEREE / MULTI-BOOK FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Loading EFL raw CSVs for referee & multi-book market features...")
try:
    efl_raw = load_efl_raw(str(RAW_DIR))
    efl_raw["Date"] = pd.to_datetime(efl_raw["Date"], errors="coerce")
    efl_raw = efl_raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"  EFL matches loaded: {len(efl_raw):,}")

    # Extract multi-book market features
    efl_market_feats = extract_multi_book_features(efl_raw)
    efl_enriched = pd.concat([
        efl_raw[["Date", "HomeTeam", "AwayTeam", "Referee",
                 "HTHG", "HTAG", "FTHG", "FTAG"]].copy(),
        efl_market_feats,
    ], axis=1)

    # Only keep EFL-specific enrichment columns (avoid duplicating HTHG/HTAG/etc.)
    efl_market_cols = [c for c in efl_enriched.columns
                       if c not in ("HTHG", "HTAG", "FTHG", "FTAG")]
    efl_merge = efl_enriched[efl_market_cols].drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"], keep="last"
    )

    # Normalize dates to midnight so join keys match across data sources
    efl_merge["Date"] = efl_merge["Date"].dt.normalize()
    for split_df in [train_df, val_df, test_df]:
        split_df["Date"] = pd.to_datetime(split_df["Date"]).dt.normalize()

    train_df = train_df.merge(efl_merge, on=["Date", "HomeTeam", "AwayTeam"],
                               how="left", suffixes=("", "_efl"))
    val_df   = val_df.merge(efl_merge, on=["Date", "HomeTeam", "AwayTeam"],
                             how="left", suffixes=("", "_efl"))
    test_df  = test_df.merge(efl_merge, on=["Date", "HomeTeam", "AwayTeam"],
                              how="left", suffixes=("", "_efl"))

    # Recompute y arrays after merge (row count is preserved with left join + dedup)
    y_train = (train_df["HTHG"] == train_df["HTAG"]).astype(int).values
    y_val   = (val_df["HTHG"] == val_df["HTAG"]).astype(int).values
    y_test  = (test_df["HTHG"] == test_df["HTAG"]).astype(int).values

    # Rename Referee_raw back to Referee if not already present
    for split_df in [train_df, val_df, test_df]:
        if "Referee" not in split_df.columns and "Referee_raw" in split_df.columns:
            split_df.rename(columns={"Referee_raw": "Referee"}, inplace=True)
        elif "Referee_raw" in split_df.columns:
            split_df["Referee"] = split_df["Referee"].fillna(split_df.get("Referee_raw", np.nan))

    efl_ok = True
    print(f"  Referee coverage in train: {train_df['Referee'].notna().mean():.1%}")
    print(f"  consensus_draw_prob coverage in train: {train_df['consensus_draw_prob'].notna().mean():.1%}")
except Exception as e:
    print(f"  Warning: EFL raw loading failed: {e}")
    efl_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIT DIXON-COLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Fitting Dixon-Coles per-league on training set (incl. FT-proxy for MLS/MEX)...")
dc = DixonColesEnsemble(xi=0.003)
dc.fit(full_train_df, min_matches=30, ft_only_leagues=FT_ONLY_LEAGUES)
dc.save(str(OUT_DIR / "dixon_coles.pkl"))

dc_val_preds  = dc.predict_draw(val_df)
dc_test_preds = dc.predict_draw(test_df)

valid_val_dc = np.isfinite(dc_val_preds)
if valid_val_dc.sum() > 10:
    dc_val_auc = roc_auc_score(y_val[valid_val_dc], dc_val_preds[valid_val_dc])
    print(f"  Dixon-Coles val AUC:  {dc_val_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIT ELO RATING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Fitting Elo rating system...")

# Tune K on a subset (EFL + English leagues for speed)
print("  Tuning K-factor (English leagues only for speed)...")
efl_train = train_df[train_df["league"].isin(["E1", "E0", "E2", "E3"])].copy()
if len(efl_train) > 500:
    tune_res = tune_k_factor(efl_train, k_values=[16, 24, 32, 40, 48])
    best_k = tune_res["best_k"]
else:
    best_k = 32
    print("  Using default K=32")

# Fit Elo on full training set (including FT-proxy rows for MLS/MEX teams)
elo = EloRatingSystem(k=best_k, home_adv=50)
elo.fit(full_train_df)

# Get val/test predictions (Elo has seen all train data, predict from current state)
elo_val_preds  = elo.predict_draw(val_df)
elo_test_preds = elo.predict_draw(test_df)

valid_val_elo = np.isfinite(elo_val_preds)
if valid_val_elo.sum() > 10:
    elo_val_auc = roc_auc_score(y_val[valid_val_elo], elo_val_preds[valid_val_elo])
    print(f"  Elo val AUC:          {elo_val_auc:.4f}  (K={best_k})")

elo.save(str(OUT_DIR / "elo.pkl"))

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT REFEREE MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Fitting referee model...")
rm = RefereeModel(min_matches=15)

if efl_ok and "Referee" in train_df.columns:
    rm.fit(train_df)
else:
    print("  No referee data in training set — referee model will return 1.0")
    rm.fitted_ = True
    rm.global_draw_rate_ = GLOBAL_DRAW_RATE

rm.save(str(OUT_DIR / "referee_model.pkl"))

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIT MARKET MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Fitting market model (multi-bookmaker feature extraction)...")
mm = MarketModel()
mm.fit(train_df)
mm.save(str(OUT_DIR / "market_model.pkl"))

# Market signal AUC
mf_val     = mm.transform(val_df)
mkt_probs  = mf_val["consensus_draw_prob"].values
valid_mkt  = np.isfinite(mkt_probs)
if valid_mkt.sum() > 10:
    mkt_val_auc = roc_auc_score(y_val[valid_mkt], mkt_probs[valid_mkt])
    print(f"  Market consensus val AUC: {mkt_val_auc:.4f}")

# B365-only benchmark
b365_probs = mf_val["b365_draw_prob"].values
valid_b365 = np.isfinite(b365_probs)
if valid_b365.sum() > 10:
    b365_auc = roc_auc_score(y_val[valid_b365], b365_probs[valid_b365])
    print(f"  B365-only val AUC:        {b365_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. LOAD PRE-TRAINED XGB + LGB MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Loading pre-trained XGB + LGB from models/mega/...")
import xgboost as xgb_lib
import lightgbm as lgb_lib

xgb_model, lgb_model = None, None
xgb_cal, lgb_cal     = None, None
scaler, medians, feat_names = None, None, None

try:
    xgb_model = xgb_lib.Booster()
    xgb_model.load_model(str(MEGA_MODELS / "xgb_model.json"))
    print(f"  XGBoost loaded: {MEGA_MODELS / 'xgb_model.json'}")
except Exception as e:
    print(f"  XGBoost load failed: {e}")

try:
    lgb_model = lgb_lib.Booster(model_file=str(MEGA_MODELS / "lgbm_model.txt"))
    print(f"  LightGBM loaded: {MEGA_MODELS / 'lgbm_model.txt'}")
except Exception as e:
    print(f"  LightGBM load failed: {e}")

try:
    with open(MEGA_MODELS / "xgb_calibrator.pkl", "rb") as f:
        xgb_cal = pickle.load(f)
    with open(MEGA_MODELS / "lgb_calibrator.pkl", "rb") as f:
        lgb_cal = pickle.load(f)
    print("  Calibrators loaded")
except Exception as e:
    print(f"  Calibrator load failed: {e}")

try:
    with open(MEGA_MODELS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MEGA_MODELS / "medians.json", "r") as f:
        medians = json.load(f)
    with open(MEGA_MODELS / "selected_features.json", "r") as f:
        feat_names = json.load(f)
    print(f"  Scaler + {len(feat_names)} features loaded")
except Exception as e:
    print(f"  Scaler/features load failed: {e}")

# Get XGB/LGB predictions on val+test
def get_mega_preds(df_split, model_type="xgb"):
    if feat_names is None or scaler is None:
        return np.full(len(df_split), GLOBAL_DRAW_RATE)
    X = np.zeros((len(df_split), len(feat_names)), dtype=np.float32)
    for j, col in enumerate(feat_names):
        if col in df_split.columns:
            vals = pd.to_numeric(df_split[col], errors="coerce").values
            med  = medians.get(col, 0.0) if medians else 0.0
            X[:, j] = np.where(np.isfinite(vals), vals, med)
        else:
            X[:, j] = medians.get(col, 0.0) if medians else 0.0

    if model_type == "xgb" and xgb_model is not None:
        preds = xgb_model.predict(xgb_lib.DMatrix(X))
        if xgb_cal is not None:
            preds = xgb_cal.predict_proba(preds.reshape(-1, 1))[:, 1]
    elif model_type == "lgb" and lgb_model is not None:
        preds = lgb_model.predict(X)
        if lgb_cal is not None:
            preds = lgb_cal.predict_proba(preds.reshape(-1, 1))[:, 1]
    else:
        return np.full(len(df_split), GLOBAL_DRAW_RATE)

    return np.clip(preds, 0.01, 0.99)

xgb_val_preds  = get_mega_preds(val_df, "xgb")
lgb_val_preds  = get_mega_preds(val_df, "lgb")
xgb_test_preds = get_mega_preds(test_df, "xgb")
lgb_test_preds = get_mega_preds(test_df, "lgb")

xgb_val_auc = roc_auc_score(y_val, xgb_val_preds) if xgb_model is not None else 0.5
lgb_val_auc = roc_auc_score(y_val, lgb_val_preds) if lgb_model is not None else 0.5
print(f"  XGBoost val AUC:  {xgb_val_auc:.4f}")
print(f"  LightGBM val AUC: {lgb_val_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ASSEMBLE ENSEMBLE + TRAIN META-LEARNER ON VAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Assembling and fitting ensemble meta-learner...")

ep = EnsemblePredictor()
ep.global_draw_rate_ = GLOBAL_DRAW_RATE

# Inject sub-models
ep.dixon_coles    = dc
ep.elo            = elo
ep.xgb_model      = xgb_model
ep.lgb_model      = lgb_model
ep.market_model   = mm
ep.referee_model  = rm
ep.scaler_        = scaler
ep.medians_       = medians
ep.feature_names_ = feat_names
ep.xgb_calibrator_ = xgb_cal
ep.lgb_calibrator_ = lgb_cal

ep.fit(val_df, y_val, test_df=test_df, y_test=y_test)

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL TEST EVALUATION — FULL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Final evaluation on test set...")

ensemble_test_preds = ep.predict(test_df)

# B365 market benchmark on test
test_market = mf_val  # reuse val transform; actually need test
mf_test     = mm.transform(test_df)
mkt_test    = mf_test["consensus_draw_prob"].values
b365_test   = mf_test["b365_draw_prob"].values

valid_mkt_t  = np.isfinite(mkt_test)
valid_b365_t = np.isfinite(b365_test)

metrics = {}
rows_to_print = []

def safe_auc(y_true, y_pred, name=""):
    valid = np.isfinite(y_pred)
    if valid.sum() < 10:
        return None
    try:
        return roc_auc_score(y_true[valid], y_pred[valid])
    except Exception:
        return None

def safe_brier(y_true, y_pred):
    valid = np.isfinite(y_pred)
    if valid.sum() < 10:
        return None
    try:
        return brier_score_loss(y_true[valid], y_pred[valid])
    except Exception:
        return None

model_results = [
    ("Dixon-Coles",        dc_test_preds),
    ("Elo",                elo_test_preds),
    ("XGBoost (mega)",     xgb_test_preds),
    ("LightGBM (mega)",    lgb_test_preds),
    ("Market consensus",   mkt_test),
    ("B365 baseline",      b365_test),
    ("ENSEMBLE V2",        ensemble_test_preds),
]

print("\n" + "=" * 68)
print("FINAL COMPARISON TABLE — TEST SET")
print("=" * 68)
hdr = f"{'Model':<28} {'AUC':>8} {'Brier':>8} {'vs B365':>8}"
print(hdr)
print("-" * 68)

b365_test_auc = safe_auc(y_test, b365_test) or 0.5

for label, preds in model_results:
    auc   = safe_auc(y_test, preds, label)
    brier = safe_brier(y_test, preds)
    if auc is None:
        continue
    delta = f"{auc - b365_test_auc:+.4f}"
    is_ensemble = "ENSEMBLE" in label
    mark = " ←" if is_ensemble else ""
    print(f"{'  '+label if is_ensemble else label:<28} {auc:>8.4f} {(brier or 0):>8.4f} {delta:>8}{mark}")
    metrics[label.lower().replace(" ", "_")] = {
        "test_auc": round(auc, 5) if auc else None,
        "test_brier": round(brier, 5) if brier else None,
    }

print("-" * 68)
print(f"  Test draw rate: {y_test.mean():.3f}  |  n={len(y_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE ENSEMBLE + METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Saving models...")
ep.save(str(OUT_DIR / "ensemble_weights.pkl"))

# Save a lightweight version of sub-model paths (for load_full)
submodel_paths = {
    "weights_path":   str(OUT_DIR / "ensemble_weights.pkl"),
    "dc_path":        str(OUT_DIR / "dixon_coles.pkl"),
    "elo_path":       str(OUT_DIR / "elo.pkl"),
    "xgb_path":       str(MEGA_MODELS / "xgb_model.json"),
    "lgb_path":       str(MEGA_MODELS / "lgbm_model.txt"),
    "scaler_path":    str(MEGA_MODELS / "scaler.pkl"),
    "medians_path":   str(MEGA_MODELS / "medians.json"),
    "features_path":  str(MEGA_MODELS / "selected_features.json"),
    "xgb_cal_path":   str(MEGA_MODELS / "xgb_calibrator.pkl"),
    "lgb_cal_path":   str(MEGA_MODELS / "lgb_calibrator.pkl"),
    "referee_path":   str(OUT_DIR / "referee_model.pkl"),
    "market_path":    str(OUT_DIR / "market_model.pkl"),
}
with open(OUT_DIR / "submodel_paths.json", "w") as f:
    json.dump(submodel_paths, f, indent=2)

# Ensemble metrics JSON
ensemble_metrics = {
    "dataset": {
        "n_total":    n,
        "n_train":    len(train_df),
        "n_val":      len(val_df),
        "n_test":     len(test_df),
        "draw_rate":  GLOBAL_DRAW_RATE,
        "train_end_date": str(train_df["Date"].max().date()),
        "val_end_date":   str(val_df["Date"].max().date()),
        "test_end_date":  str(test_df["Date"].max().date()),
    },
    "val_signal_aucs": ep.val_signal_aucs_,
    "test_results": metrics,
    "meta_learner_weights": {
        name: float(coef)
        for name, coef in zip(ep.SIGNAL_NAMES, ep.meta_lr_.coef_[0])
    } if ep.meta_lr_ is not None else {},
    "submodel_paths": submodel_paths,
}

with open(OUT_DIR / "ensemble_metrics.json", "w") as f:
    json.dump(ensemble_metrics, f, indent=2)
print(f"  Saved models/v2/ensemble_metrics.json")

# ─────────────────────────────────────────────────────────────────────────────
# 11. QUICK EXAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Example predictions (see predict_match.py for full output):")

examples = [
    ("Leeds United",   "Sheffield Utd",  "E1",       "M Dean"),
    ("Sunderland",     "Coventry",       "E1",       None),
    ("Man City",       "Liverpool",      "E0",       None),
    ("LA Galaxy",      "NYCFC",          "USA_MLS",  None),
    ("Club America",   "Chivas",         "MEX_LigaMX", None),
]

for home, away, league, ref in examples:
    result = ep.predict_single(
        home_team=home, away_team=away, league=league, referee=ref,
        market_odds={"B365H": 2.2, "B365D": 3.3, "B365A": 3.6}
    )
    p = result["p_ensemble"]
    ci_lo, ci_hi = result["ci_lower"], result["ci_upper"]
    ref_str = f"  Ref: {ref}" if ref else ""
    print(f"  {home:<20} vs {away:<18} [{league}]{ref_str:15}")
    print(f"    P(HT draw) = {p:.3f}  ({ci_lo:.3f}–{ci_hi:.3f} 90% CI)")
    print(f"    Breakdown: DC={result['breakdown']['dixon_coles']}  "
          f"Elo={result['breakdown']['elo']}  "
          f"XGB={result['breakdown']['xgboost']}  "
          f"Mkt={result['breakdown']['market_consensus']}")
    if result['quarter_kelly_stake']:
        print(f"    Quarter-Kelly stake: {result['quarter_kelly_stake']:.3f} units")
    print()

print("=" * 72)
print("V2 ENSEMBLE TRAINING COMPLETE")
print(f"  Models saved to: models/v2/")
print("=" * 72)
