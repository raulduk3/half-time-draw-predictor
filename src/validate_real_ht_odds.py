"""
Real HT Odds Validation
=======================
Scores the V4 inverted edge against REAL half-time draw odds (Pinnacle,
from data/historical_ht_odds/ht_draw_odds.csv) instead of the B365 full-time
draw proxy used in train_v4.py / backtest_full.py.

Questions answered:
  1. Does each edge tier hit more often than the de-vigged HT market implies?
  2. What is the flat ROI at real HT odds (vs the FT-proxy ROI)?
  3. Does the edge carry information beyond the HT market price?
     (logit: y ~ logit(p_ht_market) + edge)
  4. Is the "inverted" story real? (logit: y ~ logit(p_ht_market) + logit(A) + logit(B);
     the claim needs B's coefficient to be negative)

Model A / Model B are scored exactly as in train_v4.py, using the saved
artifacts in models/v4/ (DC/Elo/referee sub-models are the saved ones, i.e.
the same stale-at-train-end state the test metrics were produced with).

Usage:
    python src/validate_real_ht_odds.py
    python src/validate_real_ht_odds.py --out data/processed/real_ht_validation.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, ".")

from src.merge_ht_odds import name_similarity

MODELS_DIR = Path("models/v4")
HT_ODDS_PATH = Path("data/historical_ht_odds/ht_draw_odds.csv")
_V3_PATH = Path("data/processed/mega_dataset_v3.parquet")
_V2_PATH = Path("data/processed/mega_dataset_v2.parquet")

TIERS = [
    ("STRONG VALUE (>=5%)", 0.05, 1.0),
    ("VALUE (3-5%)", 0.03, 0.05),
    ("MARGINAL (1-3%)", 0.01, 0.03),
    ("NEUTRAL (-1 to 1%)", -0.01, 0.01),
    ("ANTI-VALUE (<-1%)", -1.0, -0.01),
]


# ── Join ──────────────────────────────────────────────────────────────────────

def join_ht_odds(mega: pd.DataFrame, ht: pd.DataFrame) -> pd.DataFrame:
    """Match each HT odds row to a mega row: same league, date ±1 day, fuzzy names."""
    ht = ht.copy()
    ht["ko_date"] = pd.to_datetime(ht["commence_time"], utc=True).dt.tz_localize(None).dt.normalize()

    mega_idx, scores = [], []
    for _, r in ht.iterrows():
        cand = mega[(mega["league"] == r["league"])
                    & ((mega["Date"] - r["ko_date"]).abs() <= pd.Timedelta(days=1))]
        best_i, best_s = None, 0.0
        for i, m in cand.iterrows():
            s = (name_similarity(r["home_team"], m["HomeTeam"])
                 + name_similarity(r["away_team"], m["AwayTeam"])) / 2
            if s > best_s:
                best_i, best_s = i, s
        mega_idx.append(best_i if best_s > 0.5 else None)
        scores.append(best_s)

    ht["mega_idx"] = mega_idx
    ht["match_score"] = scores
    return ht


# ── Model scoring (mirrors train_v4.py) ──────────────────────────────────────

def engineer_model_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, src in [("log_home_win_odds", "B365H"), ("log_draw_odds", "B365D"), ("log_away_win_odds", "B365A"),
                     ("log_ps_home", "PSH"), ("log_ps_draw", "PSD"), ("log_ps_away", "PSA"),
                     ("log_avg_home", "AvgH"), ("log_avg_draw", "AvgD"), ("log_avg_away", "AvgA")]:
        if col not in df.columns and src in df.columns:
            df[col] = np.log(pd.to_numeric(df[src], errors="coerce").clip(lower=1.01))
    draw_cols = [c for c in ["B365D", "PSD", "BWD", "IWD", "WHD"] if c in df.columns]
    if "odds_spread_draw" not in df.columns and len(draw_cols) >= 2:
        m = df[draw_cols].apply(pd.to_numeric, errors="coerce")
        df["odds_spread_draw"] = m.max(axis=1) - m.min(axis=1)
    return df


def build_X(split_df: pd.DataFrame, features: list, medians: dict) -> np.ndarray:
    X = np.zeros((len(split_df), len(features)), dtype=np.float32)
    for j, col in enumerate(features):
        vals = (pd.to_numeric(split_df[col], errors="coerce").values if col in split_df.columns
                else np.full(len(split_df), np.nan))
        X[:, j] = np.where(np.isfinite(vals), vals, medians.get(col, 0.0))
    return X


def score_models(df: pd.DataFrame) -> pd.DataFrame:
    from src.dixon_coles import DixonColesEnsemble
    from src.elo import EloRatingSystem

    paths = json.load(open(MODELS_DIR / "v4_paths.json"))
    df = engineer_model_a(df).reset_index(drop=True)

    # Model A
    feat_a = json.load(open(paths["model_a_features"]))
    med_a = json.load(open(paths["model_a_medians"]))
    lr_a = pickle.load(open(paths["model_a_lr"], "rb"))
    sc_a = pickle.load(open(paths["model_a_scaler"], "rb"))
    iso_a = pickle.load(open(paths["model_a_calibrator"], "rb"))
    pa = lr_a.predict_proba(sc_a.transform(build_X(df, feat_a, med_a)))[:, 1]
    df["prob_a"] = np.clip(iso_a.predict(pa), 0.01, 0.99)

    # Model B extras
    dc = DixonColesEnsemble.load(paths["dc_path"])
    elo = EloRatingSystem.load(paths["elo_path"])
    ref = pickle.load(open(paths["referee_path"], "rb"))
    df["dc_draw_prob"] = np.clip(dc.predict_draw(df), 0.01, 0.99)
    df["elo_draw_prob"] = np.clip(elo.predict_draw(df), 0.01, 0.99)
    df["referee_adj"] = np.clip(ref.predict_draw_adjustment(df), 0.5, 2.0)

    # Model B
    feat_b = json.load(open(paths["model_b_features"]))
    med_b = json.load(open(paths["model_b_medians"]))
    iso_b = pickle.load(open(paths["model_b_calibrator"], "rb"))
    Xb = build_X(df, feat_b, med_b)
    if paths.get("model_b_best", "").lower() == "xgboost":
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(paths["model_b_xgb"])
        pb = bst.predict(xgb.DMatrix(Xb, feature_names=feat_b))
    else:
        import lightgbm as lgb
        pb = lgb.Booster(model_file=paths["model_b_lgb"]).predict(Xb)
    df["prob_b"] = np.clip(iso_b.predict(pb), 0.01, 0.99)

    df["edge"] = df["prob_a"] - df["prob_b"]
    return df


# ── Stats helpers ────────────────────────────────────────────────────────────

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logit(X: np.ndarray, y: np.ndarray, names: list) -> pd.DataFrame:
    """Unpenalised logistic regression via Newton-Raphson, with Wald SEs."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = beta / se
    return pd.DataFrame({"coef": beta, "se": se, "z": z, "p_value": 2 * norm.sf(np.abs(z))},
                        index=["intercept"] + names)


def flat_roi(y: np.ndarray, odds: np.ndarray) -> float:
    return float(np.where(y == 1, odds - 1, -1.0).mean())


def boot_roi_ci(y: np.ndarray, odds: np.ndarray, n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    rois = np.where(y[idx] == 1, odds[idx] - 1, -1.0).mean(axis=1)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)), float((rois > 0).mean())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/real_ht_validation.csv")
    args = parser.parse_args()

    data_path = _V3_PATH if _V3_PATH.exists() else _V2_PATH
    if not data_path.exists():
        sys.exit(f"Mega dataset not found ({_V3_PATH} / {_V2_PATH}). Run src/update_data.py first.")

    mega = pd.read_parquet(data_path)
    mega["Date"] = pd.to_datetime(mega["Date"])
    ht = pd.read_csv(HT_ODDS_PATH)
    mega = mega[mega["league"].isin(ht["league"].unique())
                & (mega["Date"] >= pd.to_datetime(ht["commence_time"]).min().tz_localize(None) - pd.Timedelta(days=2))]
    mega = mega[mega["HTHG"].notna() & mega["HTAG"].notna()]

    print(f"Dataset: {data_path}  ({len(mega):,} candidate rows)")
    print(f"HT odds rows: {len(ht):,}")

    # 1. Join
    joined = join_ht_odds(mega, ht)
    matched = joined[joined["mega_idx"].notna()].copy()
    print(f"Matched: {len(matched):,} / {len(ht):,}")
    unmatched = joined[joined["mega_idx"].isna()]
    if len(unmatched):
        last_date = mega["Date"].max().date()
        print(f"  Unmatched: {len(unmatched)} (dataset ends {last_date}; "
              f"later kickoffs cannot match until update_data.py is run)")
    weakest = matched.nsmallest(8, "match_score")
    if len(weakest):
        print("  Weakest name matches (eyeball these):")
        for _, r in weakest.iterrows():
            m = mega.loc[r["mega_idx"]]
            print(f"    {r['match_score']:.2f}  {r['home_team']} v {r['away_team']}  →  "
                  f"{m['HomeTeam']} v {m['AwayTeam']}")

    matched = matched.drop_duplicates("mega_idx")
    ht_cols = ["ht_home_pinnacle", "ht_draw_pinnacle", "ht_away_pinnacle", "ht_draw_best", "match_score"]
    df = mega.loc[matched["mega_idx"].astype(int)].copy()
    for c in ht_cols:
        df[c] = matched[c].values
    df = df.dropna(subset=["ht_home_pinnacle", "ht_draw_pinnacle", "ht_away_pinnacle"])
    df["y"] = (df["HTHG"] == df["HTAG"]).astype(int)

    # 2. Score models
    df = score_models(df)

    # HT market: proportional de-vig of Pinnacle 1st-half 3-way
    inv = 1 / df[["ht_home_pinnacle", "ht_draw_pinnacle", "ht_away_pinnacle"]]
    df["p_ht_mkt"] = inv["ht_draw_pinnacle"] / inv.sum(axis=1)
    df["ht_draw_best"] = df["ht_draw_best"].fillna(df["ht_draw_pinnacle"])
    b365d = pd.to_numeric(df["B365D"], errors="coerce")

    y = df["y"].values
    n = len(df)
    print(f"\nScored matches: {n:,}")
    print(f"  Actual HT draw rate:          {y.mean():.3f}")
    print(f"  De-vigged HT market (Pinn.):  {df['p_ht_mkt'].mean():.3f}")
    print(f"  Mean Pinnacle HT draw odds:   {df['ht_draw_pinnacle'].mean():.2f}")
    print(f"  Mean B365 FT draw odds:       {b365d.mean():.2f}")
    print(f"  Bet-everything ROI  @ HT Pinnacle: {flat_roi(y, df['ht_draw_pinnacle'].values):+.3f}"
          f"   @ B365 FT proxy: {flat_roi(y, b365d.values):+.3f}")

    # 3. Tiers
    print(f"\n{'Tier':>22} {'N':>5} {'Hit':>6} {'Mkt':>6} {'Hit-Mkt':>8} {'z':>6} "
          f"{'HTodds':>7} {'ROI@HT':>8} {'ROI@best':>9} {'ROI@FT':>8}")
    print("-" * 96)
    tier_rows = []
    for label, lo, hi in TIERS:
        s = df[(df["edge"] >= lo) & (df["edge"] < hi)]
        if len(s) == 0:
            continue
        ys, pm = s["y"].values, s["p_ht_mkt"].values
        diff = ys.mean() - pm.mean()
        z = (ys.sum() - pm.sum()) / np.sqrt((pm * (1 - pm)).sum())
        r_ht = flat_roi(ys, s["ht_draw_pinnacle"].values)
        r_best = flat_roi(ys, s["ht_draw_best"].values)
        r_ft = flat_roi(ys, pd.to_numeric(s["B365D"], errors="coerce").values)
        print(f"{label:>22} {len(s):>5} {ys.mean():>6.3f} {pm.mean():>6.3f} {diff:>+8.3f} {z:>+6.2f} "
              f"{s['ht_draw_pinnacle'].mean():>7.2f} {r_ht:>+8.3f} {r_best:>+9.3f} {r_ft:>+8.3f}")
        tier_rows.append({"tier": label, "n": len(s), "hit": ys.mean(), "mkt": pm.mean(), "z": z,
                          "roi_ht_pinnacle": r_ht, "roi_ht_best": r_best, "roi_ft_proxy": r_ft})

    for thr in (0.03, 0.05):
        s = df[df["edge"] >= thr]
        if len(s) >= 20:
            lo, hi, pp = boot_roi_ci(s["y"].values, s["ht_draw_pinnacle"].values)
            print(f"\n  Edge >= {thr:.0%}: n={len(s)}  ROI@HT {flat_roi(s['y'].values, s['ht_draw_pinnacle'].values):+.3f}"
                  f"  95% CI [{lo:+.3f}, {hi:+.3f}]  P(ROI>0)={pp:.1%}")

    # 4. Does the edge add anything beyond the HT market price?
    print("\nLogit 1: y ~ logit(p_ht_mkt) + edge")
    print("  (edge must be positive & significant for any real signal)")
    print(fit_logit(np.column_stack([logit(df["p_ht_mkt"]), df["edge"]]), y,
                    ["logit_p_ht_mkt", "edge"]).round(4).to_string())

    print("\nLogit 2: y ~ logit(p_ht_mkt) + logit(A) + logit(B)")
    print("  (the 'inverted' story needs B's coefficient to be NEGATIVE)")
    print(fit_logit(np.column_stack([logit(df["p_ht_mkt"]), logit(df["prob_a"]), logit(df["prob_b"])]), y,
                    ["logit_p_ht_mkt", "logit_A", "logit_B"]).round(4).to_string())

    # 5. Verdict
    strong = df[df["edge"] >= 0.03]
    edge_p = fit_logit(np.column_stack([logit(df["p_ht_mkt"]), df["edge"]]), y,
                       ["m", "edge"]).loc["edge"]
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if len(strong) and flat_roi(strong["y"].values, strong["ht_draw_pinnacle"].values) > 0 \
            and edge_p["coef"] > 0 and edge_p["p_value"] < 0.05:
        print("  Edge >= 3% is profitable at real HT odds AND the edge adds signal beyond the")
        print("  HT market (p < 0.05). Promising — confirm with more seasons before staking.")
    else:
        print("  No evidence the edge beats the real HT market on this sample.")
        print("  (Either ROI at HT odds <= 0 or the edge adds nothing beyond the HT price.)")
    print(f"  Sample is small (n={n}); treat any result as directional, not conclusive.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["Date", "league", "HomeTeam", "AwayTeam", "HTHG", "HTAG", "y", "B365D",
            "ht_draw_pinnacle", "ht_draw_best", "p_ht_mkt", "prob_a", "prob_b", "edge", "match_score"]
    df[keep].to_csv(out, index=False)
    pd.DataFrame(tier_rows).to_json(out.with_suffix(".tiers.json"), orient="records", indent=2)
    print(f"\n  Per-match rows → {out}")


if __name__ == "__main__":
    main()
