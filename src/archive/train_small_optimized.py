"""
EFL Championship Half-Time Draw Prediction — Maximum Performance on Small Dataset
==================================================================================
Strategy:
  1. Re-build features from raw CSVs with full stat history (shots, corners, fouls, cards)
  2. Vectorised rolling stats for every team stat (windows 3 & 5)
  3. Form momentum, H2H draw rate, home-only / away-only form splits
  4. Multiple bookmaker odds: implied probability consensus + disagreement signal
  5. Models: Logistic Regression, XGBoost (Optuna), LightGBM, LSTM (real sequences), Ensemble
  6. Betting edge analysis: where does the model beat B365 implied probability?
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json, pickle, warnings, os, sys, gc
from pathlib import Path

warnings.filterwarnings("ignore")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

OUT_DIR = Path("models/small_optimized")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Load & clean raw CSVs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
    "BWH",  "BWD",  "BWA",
    "IWH",  "IWD",  "IWA",
    "LBH",  "LBD",  "LBA",
    "WHH",  "WHD",  "WHA",
    "PSH",  "PSD",  "PSA",    # Pinnacle (appears from ~2012)
    "VCH",  "VCD",  "VCA",
    "BbAvH","BbAvD","BbAvA",  # Betbrain consensus average
]


def load_raw_csvs(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load all E1*.csv files, keep relevant columns, parse dates."""
    import glob
    files = sorted(glob.glob(f"{raw_dir}/E1*.csv"))
    print(f"Loading {len(files)} raw CSV files …")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
            # keep only columns that exist in this file
            keep = [c for c in CORE_COLS if c in df.columns]
            frames.append(df[keep])
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    raw = pd.concat(frames, ignore_index=True)

    # Parse date — multiple formats appear across seasons
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG"])

    # Coerce numeric
    num_cols = [c for c in CORE_COLS if c not in ("Date", "HomeTeam", "AwayTeam", "FTR", "HTR")]
    for c in num_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.sort_values("Date").reset_index(drop=True)
    raw["y_ht_draw"] = (raw["HTHG"] == raw["HTAG"]).astype(int)
    print(f"  Combined shape: {raw.shape}, date range: {raw['Date'].min().date()} → {raw['Date'].max().date()}")
    print(f"  HT draw rate: {raw['y_ht_draw'].mean():.3f}")
    return raw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Feature engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Stats we roll over (full-time goals used as proxy when HT goals also available)
ROLL_STATS = {
    "gf":    ("FTHG", "FTAG"),   # goals for
    "ga":    ("FTAG", "FTHG"),   # goals against
    "shtf":  ("HS",   "AS"),     # shots for
    "shta":  ("AS",   "HS"),     # shots against
    "stf":   ("HST",  "AST"),    # shots-on-target for
    "sta":   ("AST",  "HST"),    # shots-on-target against
    "crnf":  ("HC",   "AC"),     # corners for
    "crna":  ("AC",   "HC"),     # corners against
    "foulf": ("HF",   "AF"),     # fouls committed (higher = more aggressive)
    "ycf":   ("HY",   "AY"),     # yellow cards for
    "rcf":   ("HR",   "AR"),     # red cards for
    # HT goals (the actual signal)
    "htgf":  ("HTHG", "HTAG"),
    "htga":  ("HTAG", "HTHG"),
}

WINDOWS = [3, 5]


def _build_team_history(df: pd.DataFrame) -> dict:
    """
    Return a dict:  team -> list of (date, stat_dict)  chronologically.
    This lets us do O(n) rolling instead of O(n²) repeated slicing.
    """
    history: dict = {}
    for row in df.itertuples():
        for team, is_home in [(row.HomeTeam, True), (row.AwayTeam, False)]:
            entry = {"date": row.Date}
            for stat, (h_col, a_col) in ROLL_STATS.items():
                hv = getattr(row, h_col, np.nan)
                av = getattr(row, a_col, np.nan)
                if is_home:
                    entry[stat] = hv
                    entry[f"{stat}_h"] = hv   # home-only track
                    entry[f"{stat}_a"] = np.nan
                else:
                    entry[stat] = av
                    entry[f"{stat}_h"] = np.nan
                    entry[f"{stat}_a"] = av   # away-only track
            history.setdefault(team, []).append(entry)
    return history


def _roll(entries: list, n: int, stat: str, up_to_idx: int) -> float:
    """Mean of last-n entries[stat] before index up_to_idx."""
    window = [e[stat] for e in entries[max(0, up_to_idx - n):up_to_idx]
              if not np.isnan(e.get(stat, np.nan))]
    return np.mean(window) if window else np.nan


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering — vectorised where possible."""
    print("Building team history index …")
    # We need per-team match index → use list position in the team's history
    # First: build team -> list[match_global_index] mapping
    team_match_idx: dict = {}   # team -> list of row index in df
    for i, row in enumerate(df.itertuples()):
        team_match_idx.setdefault(row.HomeTeam, []).append(i)
        team_match_idx.setdefault(row.AwayTeam, []).append(i)

    # Build history in one pass
    history = _build_team_history(df)
    # For each team, build a reverse map: df_index -> position in history list
    team_pos: dict = {}   # team -> {df_idx: pos_in_history}
    for team, entries in history.items():
        # entries are appended in df order (two per match — home then away)
        # We need to know: when processing match at df_idx, what is the history length?
        pass  # handled below with per-team counters

    print("Computing rolling features …")
    n_rows = len(df)
    feat_rows = []

    # Per-team counters so we know how many past matches
    team_counts: dict = {}

    for i, row in enumerate(df.itertuples()):
        if i % 2000 == 0:
            print(f"  {i}/{n_rows} …")

        home = row.HomeTeam
        away = row.AwayTeam

        # History UP TO (not including) this match
        hc = team_counts.get(home, 0)
        ac = team_counts.get(away, 0)

        h_hist = history.get(home, [])
        a_hist = history.get(away, [])

        feat = {}

        for w in WINDOWS:
            for stat in ROLL_STATS:
                feat[f"home_{stat}_r{w}"] = _roll(h_hist, w, stat, hc)
                feat[f"away_{stat}_r{w}"] = _roll(a_hist, w, stat, ac)
                # home-only and away-only splits
                feat[f"home_{stat}_h_r{w}"] = _roll(h_hist, w, f"{stat}_h", hc)
                feat[f"away_{stat}_a_r{w}"] = _roll(a_hist, w, f"{stat}_a", ac)

        # Form momentum: compare r3 vs r5 goal difference
        for side in ("home", "away"):
            gd3 = feat.get(f"{side}_gf_r3", np.nan) - feat.get(f"{side}_ga_r3", np.nan) if not np.isnan(feat.get(f"{side}_gf_r3", np.nan)) else np.nan
            gd5 = feat.get(f"{side}_gf_r5", np.nan) - feat.get(f"{side}_ga_r5", np.nan) if not np.isnan(feat.get(f"{side}_gf_r5", np.nan)) else np.nan
            feat[f"{side}_momentum"] = gd3 - gd5 if (not np.isnan(gd3) and not np.isnan(gd5)) else np.nan

        # Shot accuracy (from rolling means — only valid when shtf > 0)
        for side, shot_col, sot_col in [("home","home_shtf_r5","home_stf_r5"),
                                          ("away","away_shtf_r5","away_stf_r5")]:
            s = feat.get(shot_col, np.nan)
            st = feat.get(sot_col, np.nan)
            feat[f"{side}_shot_acc_r5"] = st / s if (s and s > 0 and not np.isnan(s) and not np.isnan(st)) else np.nan

        # Differential features (home minus away)
        for stat in ("gf_r5", "ga_r5", "shtf_r5", "stf_r5", "crnf_r5", "htgf_r5", "htga_r5"):
            h_val = feat.get(f"home_{stat}", np.nan)
            a_val = feat.get(f"away_{stat}", np.nan)
            feat[f"diff_{stat}"] = h_val - a_val if (not np.isnan(h_val) and not np.isnan(a_val)) else np.nan

        feat_rows.append(feat)

        # Update counters AFTER extracting features (use past data only)
        team_counts[home] = hc + 1
        team_counts[away] = ac + 1

    feat_df = pd.DataFrame(feat_rows)
    result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return result


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implied probabilities from multiple bookmakers.
    Compute:
      - B365 implied probs (normalised by overround)
      - Consensus implied prob (average across bookmakers)
      - Disagreement: std-dev of draw implied prob across books
      - Log odds (for LR / LSTM)
    """
    df = df.copy()
    BOOKS = {
        "b365": ("B365H", "B365D", "B365A"),
        "bw":   ("BWH",   "BWD",   "BWA"),
        "iw":   ("IWH",   "IWD",   "IWA"),
        "lb":   ("LBH",   "LBD",   "LBA"),
        "wh":   ("WHH",   "WHD",   "WHA"),
        "ps":   ("PSH",   "PSD",   "PSA"),
        "vc":   ("VCH",   "VCD",   "VCA"),
        "bb":   ("BbAvH", "BbAvD", "BbAvA"),
    }

    implied_draw = {}
    for book, (h_col, d_col, a_col) in BOOKS.items():
        if h_col not in df.columns or d_col not in df.columns or a_col not in df.columns:
            continue
        h = pd.to_numeric(df[h_col], errors="coerce")
        d = pd.to_numeric(df[d_col], errors="coerce")
        a = pd.to_numeric(df[a_col], errors="coerce")
        valid = (h > 1) & (d > 1) & (a > 1)
        overround = 1/h + 1/d + 1/a
        df[f"impl_draw_{book}"] = np.where(valid, (1/d) / overround, np.nan)
        df[f"impl_home_{book}"] = np.where(valid, (1/h) / overround, np.nan)
        df[f"impl_away_{book}"] = np.where(valid, (1/a) / overround, np.nan)
        implied_draw[book] = df[f"impl_draw_{book}"]

    if implied_draw:
        impl_matrix = pd.DataFrame(implied_draw)
        df["impl_draw_consensus"] = impl_matrix.mean(axis=1)
        df["impl_draw_std"]       = impl_matrix.std(axis=1)   # disagreement signal
        df["impl_home_consensus"] = pd.DataFrame({b: df[f"impl_home_{b}"] for b in implied_draw}).mean(axis=1)
        df["impl_away_consensus"] = pd.DataFrame({b: df[f"impl_away_{b}"] for b in implied_draw}).mean(axis=1)

    # Log B365 odds (for models that benefit from raw odds scale)
    for col, name in [("B365H","log_b365h"), ("B365D","log_b365d"), ("B365A","log_b365a")]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[name] = np.where(v > 1, np.log(v), np.nan)

    return df


def add_calendar_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, season progression, H2H draw rate."""
    df = df.copy()
    df["dow"]   = df["Date"].dt.dayofweek           # 0=Mon … 6=Sun
    df["month"] = df["Date"].dt.month
    df["year"]  = df["Date"].dt.year

    # Season: Aug-May → season year = year of August start
    df["season"] = np.where(df["month"] >= 8, df["year"], df["year"] - 1)

    # Match-day of season (proxy for early/mid/late)
    df["season_day"] = (df["Date"] - pd.to_datetime(df["season"].astype(str) + "-08-01")).dt.days.clip(0)

    # Rest days
    df = df.sort_values("Date").reset_index(drop=True)
    last_played: dict = {}
    rest_home, rest_away = [], []
    for row in df.itertuples():
        for side, team, store in [("home", row.HomeTeam, rest_home), ("away", row.AwayTeam, rest_away)]:
            last = last_played.get(team)
            store.append((row.Date - last).days if last else np.nan)
        last_played[row.HomeTeam] = row.Date
        last_played[row.AwayTeam] = row.Date
    df["home_rest"] = rest_home
    df["away_rest"] = rest_away

    # H2H draw rate (based on past meetings only — no leakage)
    df["h2h_draw_rate"] = np.nan
    df["h2h_n_games"]   = 0
    pair_draws: dict = {}
    pair_games: dict = {}
    for i, row in enumerate(df.itertuples()):
        pair = tuple(sorted([row.HomeTeam, row.AwayTeam]))
        d_rate = pair_draws.get(pair, 0) / pair_games.get(pair, 1) if pair_games.get(pair, 0) > 0 else np.nan
        df.at[i, "h2h_draw_rate"] = d_rate
        df.at[i, "h2h_n_games"]   = pair_games.get(pair, 0)
        # Update with current match outcome
        pair_games[pair] = pair_games.get(pair, 0) + 1
        if row.y_ht_draw:
            pair_draws[pair] = pair_draws.get(pair, 0) + 1

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Prepare feature matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_feature_matrix(df: pd.DataFrame):
    """Return (df_clean, feature_cols)."""
    # Candidate features
    roll_feat = [c for c in df.columns if any(c.startswith(p) for p in
                 ("home_", "away_", "diff_")) and ("_r3" in c or "_r5" in c)]
    roll_feat += [c for c in df.columns if "momentum" in c or "shot_acc" in c]

    odds_feat = [c for c in df.columns if c.startswith("impl_") or c.startswith("log_b365")]

    cal_feat  = ["dow", "month", "season_day", "home_rest", "away_rest",
                 "h2h_draw_rate", "h2h_n_games"]

    all_feat = sorted(set(roll_feat + odds_feat + cal_feat))
    all_feat = [c for c in all_feat if c in df.columns]

    # Drop columns that are >60% NaN
    missing_pct = df[all_feat].isna().mean()
    keep = missing_pct[missing_pct < 0.60].index.tolist()
    print(f"Features after NaN filter: {len(keep)} / {len(all_feat)}")

    # Drop rows missing the core odds (B365D essential)
    required = ["impl_draw_b365", "y_ht_draw"] if "impl_draw_b365" in df.columns else ["log_b365d", "y_ht_draw"]
    df_clean = df.dropna(subset=required).copy()

    # Fill remaining NaNs with column median (train-time medians applied later)
    df_clean[keep] = df_clean[keep].fillna(df_clean[keep].median())
    df_clean = df_clean.dropna(subset=keep + ["y_ht_draw"])

    print(f"Clean rows: {len(df_clean):,}  |  features: {len(keep)}")
    return df_clean, keep


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Temporal split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def temporal_split(df, feat_cols):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    i_train = int(0.70 * n)
    i_val   = i_train + int(0.15 * n)
    tr, va, te = df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]
    print(f"Train {len(tr):,}  ({tr['Date'].min().date()} – {tr['Date'].max().date()})  draw={tr['y_ht_draw'].mean():.3f}")
    print(f"Val   {len(va):,}  ({va['Date'].min().date()} – {va['Date'].max().date()})  draw={va['y_ht_draw'].mean():.3f}")
    print(f"Test  {len(te):,}  ({te['Date'].min().date()} – {te['Date'].max().date()})  draw={te['y_ht_draw'].mean():.3f}")

    Xtr = tr[feat_cols].values.astype(np.float32)
    Xva = va[feat_cols].values.astype(np.float32)
    Xte = te[feat_cols].values.astype(np.float32)
    ytr, yva, yte = tr["y_ht_draw"].values, va["y_ht_draw"].values, te["y_ht_draw"].values

    # Fit scaler on train only
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # Also fill any remaining NaNs introduced by imputation mismatch
    Xtr_s = np.nan_to_num(Xtr_s, nan=0.0)
    Xva_s = np.nan_to_num(Xva_s, nan=0.0)
    Xte_s = np.nan_to_num(Xte_s, nan=0.0)

    return (Xtr_s, ytr, tr), (Xva_s, yva, va), (Xte_s, yte, te), scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def metrics(y_true, y_prob, label):
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    print(f"  {label:12s}  AUC={auc:.4f}  Brier={brier:.4f}")
    return auc, brier


# ── 5a. Logistic Regression ─────────────────────────────────────────────────
def train_lr(Xtr, ytr, Xva, yva, Xte, yte):
    print("\n[1/5] Logistic Regression …")
    best, best_c = None, None
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                               class_weight="balanced", random_state=42)
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or auc > best:
            best, best_c, best_m = auc, C, m
    print(f"  Best C={best_c}")
    p_tr = best_m.predict_proba(Xtr)[:, 1]
    p_va = best_m.predict_proba(Xva)[:, 1]
    p_te = best_m.predict_proba(Xte)[:, 1]
    metrics(ytr, p_tr, "LR train")
    metrics(yva, p_va, "LR val")
    auc_te, brier_te = metrics(yte, p_te, "LR test")
    return best_m, p_te, p_va, {"test_auc": auc_te, "test_brier": brier_te,
                                  "val_auc": best, "name": "LogReg"}


# ── 5b. XGBoost with Optuna ─────────────────────────────────────────────────
def train_xgb(Xtr, ytr, Xva, yva, Xte, yte, feat_cols, n_trials=60):
    print("\n[2/5] XGBoost + Optuna …")
    scale_pos = (ytr == 0).sum() / (ytr == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  scale_pos,
            "eval_metric":       "auc",
            "early_stopping_rounds": 30,
            "random_state": 42,
            "n_jobs": -1,
        }
        m = xgb.XGBClassifier(**params, verbosity=0)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_p = study.best_params
    print(f"  Best params: {best_p}")

    # Retrain with best params
    best_p.update({"scale_pos_weight": scale_pos, "eval_metric": "auc",
                   "early_stopping_rounds": 30, "random_state": 42, "n_jobs": -1})
    model = xgb.XGBClassifier(**best_p, verbosity=0)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    p_tr = model.predict_proba(Xtr)[:, 1]
    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    metrics(ytr, p_tr, "XGB train")
    metrics(yva, p_va, "XGB val")
    auc_te, brier_te = metrics(yte, p_te, "XGB test")

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xte)
    shap_imp = pd.DataFrame({
        "feature": feat_cols,
        "shap_mean_abs": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_mean_abs", ascending=False)

    return model, p_te, p_va, {"test_auc": auc_te, "test_brier": brier_te,
                                 "val_auc": study.best_value, "name": "XGBoost"}, shap_imp


# ── 5c. LightGBM ─────────────────────────────────────────────────────────────
def train_lgbm(Xtr, ytr, Xva, yva, Xte, yte, feat_cols):
    print("\n[3/5] LightGBM …")
    scale_pos = (ytr == 0).sum() / (ytr == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  scale_pos,
            "n_jobs": -1, "random_state": 42, "verbose": -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr,
              eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    best_p = study.best_params
    best_p.update({"scale_pos_weight": scale_pos, "n_jobs": -1,
                   "random_state": 42, "verbose": -1})
    model = lgb.LGBMClassifier(**best_p)
    model.fit(Xtr, ytr,
              eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

    p_tr = model.predict_proba(Xtr)[:, 1]
    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    metrics(ytr, p_tr, "LGB train")
    metrics(yva, p_va, "LGB val")
    auc_te, brier_te = metrics(yte, p_te, "LGB test")

    lgb_imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, p_te, p_va, {"test_auc": auc_te, "test_brier": brier_te,
                                 "val_auc": study.best_value, "name": "LightGBM"}, lgb_imp


# ── 5d. LSTM with real 5-match sequences ────────────────────────────────────
class SequenceLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


def build_sequences(df_with_feats: pd.DataFrame, feat_cols: list, seq_len: int = 5):
    """
    For each match i, collect the previous seq_len matches of each team
    (by date) and stack them as a time-series input of shape (seq_len, n_feat).
    The current match features are the last step.
    """
    df_s = df_with_feats.sort_values("Date").reset_index(drop=True)
    n = len(df_s)
    n_f = len(feat_cols)
    X_feat = df_s[feat_cols].values.astype(np.float32)

    # Build per-team ordered match indices
    team_idxs: dict = {}
    for i, row in enumerate(df_s.itertuples()):
        team_idxs.setdefault(row.HomeTeam, []).append(i)
        team_idxs.setdefault(row.AwayTeam, []).append(i)

    sequences = []
    for i, row in enumerate(df_s.itertuples()):
        home_seq = team_idxs[row.HomeTeam]
        pos = home_seq.index(i)
        past = home_seq[max(0, pos - seq_len + 1): pos + 1]  # include current
        # Pad if needed
        pad = seq_len - len(past)
        steps = np.zeros((seq_len, n_f), dtype=np.float32)
        for j, idx in enumerate(past):
            steps[pad + j] = X_feat[idx]
        sequences.append(steps)

    return np.array(sequences, dtype=np.float32)


def train_lstm(df_tr, df_va, df_te, feat_cols, scaler, seq_len=5):
    print(f"\n[4/5] LSTM (seq_len={seq_len}) …")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    # Build sequences on unscaled df, then scale each step
    def prep(df_part):
        seqs = build_sequences(df_part, feat_cols, seq_len)  # (N, T, F)
        N, T, F = seqs.shape
        flat = seqs.reshape(-1, F)
        flat_s = scaler.transform(flat)
        flat_s = np.nan_to_num(flat_s, nan=0.0)
        return flat_s.reshape(N, T, F).astype(np.float32)

    Xtr_s = prep(df_tr)
    Xva_s = prep(df_va)
    Xte_s = prep(df_te)

    ytr = df_tr["y_ht_draw"].values.astype(np.float32)
    yva = df_va["y_ht_draw"].values.astype(np.float32)
    yte = df_te["y_ht_draw"].values.astype(np.float32)

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=128, shuffle=shuffle)

    tr_loader = make_loader(Xtr_s, ytr, shuffle=True)
    va_loader = make_loader(Xva_s, yva)
    te_loader = make_loader(Xte_s, yte)

    n_feat = Xtr_s.shape[2]
    model = SequenceLSTM(n_feat, hidden=64, n_layers=2, dropout=0.35).to(device)

    # Class weights
    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Rebuild model without sigmoid for BCEWithLogitsLoss
    class SequenceLSTMLogits(nn.Module):
        def __init__(self, n_feat, hidden=64, n_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(1)

    model = SequenceLSTMLogits(n_feat, hidden=64, n_layers=2, dropout=0.35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_va_auc, best_state, patience, wait = 0, None, 20, 0

    for epoch in range(150):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        va_probs = []
        with torch.no_grad():
            for Xb, _ in va_loader:
                logits = model(Xb.to(device))
                va_probs.extend(torch.sigmoid(logits).cpu().numpy())
        va_auc = roc_auc_score(yva, va_probs)
        if va_auc > best_va_auc:
            best_va_auc, best_state, wait = va_auc, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: val_AUC={va_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    def predict(loader):
        probs = []
        with torch.no_grad():
            for Xb, _ in loader:
                logits = model(Xb.to(device))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
        return np.array(probs)

    p_tr = predict(make_loader(Xtr_s, ytr))
    p_va = predict(va_loader)
    p_te = predict(te_loader)

    metrics(ytr, p_tr, "LSTM train")
    metrics(yva, p_va, "LSTM val")
    auc_te, brier_te = metrics(yte, p_te, "LSTM test")

    return model, p_te, p_va, {"test_auc": auc_te, "test_brier": brier_te,
                                 "val_auc": best_va_auc, "name": "LSTM"}


# ── 5e. Ensemble ──────────────────────────────────────────────────────────────
def train_ensemble(preds_val: dict, preds_test: dict, yva, yte):
    """Greedy weight search on validation, apply to test."""
    print("\n[5/5] Ensemble …")
    names = list(preds_val.keys())
    best_weights, best_auc = None, 0
    # Grid search over weights [0..1] in steps of 0.1 for up to 4 models
    from itertools import product
    steps = np.arange(0, 1.01, 0.1)
    for w in product(steps, repeat=len(names)):
        w = np.array(w)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        pred = sum(w[i] * preds_val[n] for i, n in enumerate(names))
        auc = roc_auc_score(yva, pred)
        if auc > best_auc:
            best_auc, best_weights = auc, w

    print(f"  Best val weights: {dict(zip(names, best_weights.round(2)))}")
    p_va = sum(best_weights[i] * preds_val[n] for i, n in enumerate(names))
    p_te = sum(best_weights[i] * preds_test[n] for i, n in enumerate(names))
    metrics(yva, p_va, "ENS val")
    auc_te, brier_te = metrics(yte, p_te, "ENS test")
    return p_te, p_va, best_weights, names, {"test_auc": auc_te, "test_brier": brier_te,
                                               "val_auc": best_auc, "name": "Ensemble"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — Betting edge analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def betting_analysis(df_test: pd.DataFrame, model_probs: dict, model_name="best"):
    """
    Compare model P(HT draw) vs B365 implied P(draw).
    Bet when model_prob > implied_prob (model thinks it's more likely than market).
    Simulate flat bet £1 and half-Kelly staking.
    """
    results = {}
    impl_col = "impl_draw_b365" if "impl_draw_b365" in df_test.columns else None
    if impl_col is None:
        print("  No B365 implied draw probability available — skipping betting analysis.")
        return results

    df_bet = df_test.copy()
    df_bet["b365d"] = pd.to_numeric(df_bet.get("B365D", np.nan), errors="coerce")

    for mname, probs in model_probs.items():
        df_bet[f"p_{mname}"] = probs
        # Edge = model_prob - implied_prob
        df_bet[f"edge_{mname}"] = probs - df_bet[impl_col]

    for mname in model_probs:
        p_col = f"p_{mname}"
        e_col = f"edge_{mname}"

        # Filter: only bet when edge > 0 (model > market)
        bets = df_bet[df_bet[e_col] > 0].copy()
        if len(bets) == 0:
            results[mname] = {"n_bets": 0}
            continue

        y_bet  = bets["y_ht_draw"].values
        odds   = bets["b365d"].values
        probs  = bets[p_col].values
        impl_p = bets[impl_col].values

        # Flat bet ROI
        flat_returns = np.where(y_bet == 1, odds - 1, -1)
        flat_roi     = flat_returns.mean()

        # Half-Kelly
        b = odds - 1
        p = probs
        kelly = (b * p - (1 - p)) / b  # full kelly
        kelly = np.clip(kelly * 0.5, 0, 0.25)  # half kelly, cap 25%
        kelly_returns = np.where(y_bet == 1, kelly * b, -kelly)
        kelly_roi     = kelly_returns.mean()

        # Cumulative P&L
        cum_flat   = np.cumsum(flat_returns)
        cum_kelly  = np.cumsum(kelly_returns)

        results[mname] = {
            "n_bets":      len(bets),
            "n_draws":     int(y_bet.sum()),
            "hit_rate":    float(y_bet.mean()),
            "flat_roi":    float(flat_roi),
            "kelly_roi":   float(kelly_roi),
            "cum_flat_pnl":  float(cum_flat[-1]),
            "cum_kelly_pnl": float(cum_kelly[-1]),
            "avg_edge":    float(bets[e_col].mean()),
        }
        print(f"  [{mname}] Bets={len(bets)} | Hit={y_bet.mean():.3f} | "
              f"FlatROI={flat_roi:+.3f} | KellyROI={kelly_roi:+.3f}")

        # Save cumulative plot data
        bets["cum_flat"]  = cum_flat
        bets["cum_kelly"] = cum_kelly
        results[mname]["_bet_df"] = bets

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLORS = {"LogReg": "#4c72b0", "XGBoost": "#dd8452",
          "LightGBM": "#55a868", "LSTM": "#c44e52", "Ensemble": "#9467bd"}


def plot_roc_calibration(all_probs: dict, yte, label_map: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_roc, ax_cal = axes

    for name, probs in all_probs.items():
        color = COLORS.get(name, "grey")
        fpr, tpr, _ = roc_curve(yte, probs)
        auc = roc_auc_score(yte, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, lw=2)

        frac, mean_pred = calibration_curve(yte, probs, n_bins=10)
        ax_cal.plot(mean_pred, frac, "o-", label=name, color=color, lw=2)

    ax_roc.plot([0,1],[0,1],"k--", alpha=0.4)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curves — Test Set")
    ax_roc.legend(); ax_roc.grid(alpha=0.3)

    ax_cal.plot([0,1],[0,1],"k--", alpha=0.4, label="Perfect")
    ax_cal.set_xlabel("Mean Predicted Prob"); ax_cal.set_ylabel("Fraction Positives")
    ax_cal.set_title("Calibration — Test Set")
    ax_cal.legend(); ax_cal.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved roc_calibration.png")


def plot_pred_distributions(all_probs: dict, yte):
    fig, axes = plt.subplots(1, len(all_probs), figsize=(4 * len(all_probs), 4), sharey=True)
    if len(all_probs) == 1:
        axes = [axes]
    for ax, (name, probs) in zip(axes, all_probs.items()):
        for cls, color, label in [(0, "#4c72b0", "No Draw"), (1, "#dd8452", "Draw")]:
            ax.hist(probs[yte == cls], bins=30, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(name); ax.set_xlabel("P(HT Draw)"); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pred_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved pred_distributions.png")


def plot_shap(shap_imp: pd.DataFrame, top_n=25):
    top = shap_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1))
    ax.barh(range(len(top)), top["shap_mean_abs"].values, color="#dd8452")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"XGBoost SHAP Feature Importance (Top {top_n})")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_importance.png")


def plot_betting(bet_results: dict):
    rows = {k: v for k, v in bet_results.items() if "_bet_df" in v}
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4))
    if len(rows) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, rows.items()):
        bdf = res["_bet_df"].reset_index(drop=True)
        ax.plot(bdf.index, bdf["cum_flat"], label="Flat £1", lw=2)
        ax.plot(bdf.index, bdf["cum_kelly"], label="Half-Kelly", lw=2)
        ax.axhline(0, color="k", lw=0.8, linestyle="--")
        ax.set_title(f"{name} Cumulative P&L (Test)\nFlatROI={res['flat_roi']:+.3f}")
        ax.set_xlabel("Bet #"); ax.set_ylabel("Cumulative £")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "betting_roi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved betting_roi.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 70)
    print("EFL Championship HT Draw — Maximum Performance Optimisation")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    raw = load_raw_csvs()

    # ── 2. Feature engineering ───────────────────────────────────────────
    print("\n[FE] Computing rolling stats …")
    df_fe = compute_features(raw)

    print("\n[FE] Adding odds features …")
    df_fe = add_odds_features(df_fe)

    print("\n[FE] Adding calendar / H2H features …")
    df_fe = add_calendar_h2h_features(df_fe)

    # ── 3. Build feature matrix ──────────────────────────────────────────
    print("\n[PREP] Building feature matrix …")
    df_clean, feat_cols = build_feature_matrix(df_fe)
    print(f"  Final feature count: {len(feat_cols)}")

    # ── 4. Temporal split ─────────────────────────────────────────────────
    print("\n[SPLIT] Temporal 70/15/15 …")
    (Xtr, ytr, df_tr), (Xva, yva, df_va), (Xte, yte, df_te), scaler = \
        temporal_split(df_clean, feat_cols)

    # ── 5. Train models ───────────────────────────────────────────────────
    all_metrics = {}
    preds_val   = {}
    preds_test  = {}

    lr_model, lr_pte, lr_pva, lr_m = train_lr(Xtr, ytr, Xva, yva, Xte, yte)
    all_metrics["LogReg"] = lr_m; preds_val["LogReg"] = lr_pva; preds_test["LogReg"] = lr_pte

    xgb_model, xgb_pte, xgb_pva, xgb_m, shap_imp = train_xgb(
        Xtr, ytr, Xva, yva, Xte, yte, feat_cols, n_trials=60)
    all_metrics["XGBoost"] = xgb_m; preds_val["XGBoost"] = xgb_pva; preds_test["XGBoost"] = xgb_pte

    lgb_model, lgb_pte, lgb_pva, lgb_m, lgb_imp = train_lgbm(
        Xtr, ytr, Xva, yva, Xte, yte, feat_cols)
    all_metrics["LightGBM"] = lgb_m; preds_val["LightGBM"] = lgb_pva; preds_test["LightGBM"] = lgb_pte

    lstm_model, lstm_pte, lstm_pva, lstm_m = train_lstm(
        df_tr, df_va, df_te, feat_cols, scaler, seq_len=5)
    all_metrics["LSTM"] = lstm_m; preds_val["LSTM"] = lstm_pva; preds_test["LSTM"] = lstm_pte

    ens_pte, ens_pva, ens_w, ens_names, ens_m = train_ensemble(preds_val, preds_test, yva, yte)
    all_metrics["Ensemble"] = ens_m; preds_val["Ensemble"] = ens_pva; preds_test["Ensemble"] = ens_pte

    # ── 6. Betting edge analysis ──────────────────────────────────────────
    print("\n[BET] Betting edge analysis …")
    bet_results = betting_analysis(df_te.reset_index(drop=True), preds_test)
    for mname, res in bet_results.items():
        if "_bet_df" in res:
            res_clean = {k: v for k, v in res.items() if k != "_bet_df"}
            all_metrics[mname]["betting"] = res_clean

    # ── 7. Plots ──────────────────────────────────────────────────────────
    print("\n[PLOT] Generating plots …")
    plot_roc_calibration(preds_test, yte, {})
    plot_pred_distributions(preds_test, yte)
    plot_shap(shap_imp)
    plot_betting(bet_results)

    # ── 8. Save models ────────────────────────────────────────────────────
    print("\n[SAVE] Saving models …")
    with open(OUT_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    xgb_model.save_model(str(OUT_DIR / "xgb_model.json"))
    lgb_model.booster_.save_model(str(OUT_DIR / "lgb_model.txt"))
    torch.save(lstm_model.state_dict(), OUT_DIR / "lstm_model.pt")
    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    shap_imp.to_csv(OUT_DIR / "shap_importance.csv", index=False)
    lgb_imp.to_csv(OUT_DIR / "lgb_importance.csv", index=False)

    # ── 9. Save metrics JSON ──────────────────────────────────────────────
    metrics_out = {
        "feature_count": len(feat_cols),
        "train_size": int(len(ytr)),
        "val_size":   int(len(yva)),
        "test_size":  int(len(yte)),
        "models": {}
    }
    for mname, m in all_metrics.items():
        entry = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in m.items() if not k.startswith("_")}
        metrics_out["models"][mname] = entry

    with open(OUT_DIR / "metrics_small_optimized.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("  Saved metrics_small_optimized.json")

    # ── 10. Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS — Test Set")
    print("=" * 70)
    print(f"{'Model':<12}  {'AUC':>7}  {'Brier':>7}  {'vs B365 flat ROI':>18}")
    for mname, m in all_metrics.items():
        bet = m.get("betting", {})
        roi_str = f"{bet.get('flat_roi',0):+.4f} ({bet.get('n_bets',0)} bets)" if bet else "  n/a"
        print(f"  {mname:<12}  {m['test_auc']:.4f}   {m['test_brier']:.4f}   {roi_str}")

    best_name = max(all_metrics, key=lambda x: all_metrics[x]["test_auc"])
    print(f"\nBest model: {best_name}  AUC={all_metrics[best_name]['test_auc']:.4f}")

    draw_rate = yte.mean()
    naive_brier = draw_rate * (1 - draw_rate) ** 2 + (1 - draw_rate) * draw_rate ** 2
    print(f"\nBaseline (always predict draw_rate={draw_rate:.3f}): Brier={naive_brier:.4f}")
    print(f"B365 implied draw prob AUC: ", end="")
    if "impl_draw_b365" in df_te.columns:
        b365_auc = roc_auc_score(yte, df_te["impl_draw_b365"].fillna(draw_rate))
        print(f"{b365_auc:.4f}  ← market benchmark")
    else:
        print("n/a")

    print("\nOutput directory:", OUT_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
