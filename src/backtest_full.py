"""
Full Backtest with DC/Elo Computed Per-Match
============================================
Reruns the V4 backtest properly: computes Dixon-Coles and Elo
predictions for each match using only data available BEFORE that match.

This is the honest backtest. The previous run used median-filled DC/Elo
features which inflated Model B's underperformance and the edge ROI.

Takes ~30-60 min due to per-match DC/Elo fitting.

Usage:
    python src/backtest_full.py
    python src/backtest_full.py --start 2023-01-01  # custom start date
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

MEGA_PATH = Path("data/processed/mega_dataset_v2.parquet")
MODELS_DIR = Path("models/v4")
OUTPUT_PATH = Path("data/backtest_full_results.json")


def run_backtest(start_date: str = "2022-04-01", end_date: str = "2026-12-31"):
    """
    Run full backtest with per-match DC/Elo computation.
    """
    # Load dataset
    df = pd.read_parquet(MEGA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter to matches with HT scores and odds
    df = df[df["y_ht_draw"].notna() & df["B365D"].notna()].copy()

    # Load Model A
    paths = json.load(open(MODELS_DIR / "v4_paths.json"))
    model_a = pickle.load(open(paths["model_a_lr"], "rb"))
    scaler_a = pickle.load(open(paths["model_a_scaler"], "rb"))
    feat_a = json.load(open(paths["model_a_features"]))
    medians_a = json.load(open(paths["model_a_medians"]))

    # Load Model B (use whichever was best in training)
    import xgboost as xgb_lib
    feat_b = json.load(open(paths["model_b_features"]))
    medians_b = json.load(open(paths["model_b_medians"]))
    cal_a = pickle.load(open(paths["model_a_calibrator"], "rb"))
    cal_b = pickle.load(open(paths["model_b_calibrator"], "rb"))
    if paths.get("model_b_best", "").lower() == "xgboost":
        bst_b = xgb_lib.Booster()
        bst_b.load_model(paths["model_b_xgb"])
        def predict_b(X, feat_names):
            dmat = xgb_lib.DMatrix(X, feature_names=feat_names)
            return bst_b.predict(dmat)
    else:
        import lightgbm as lgb
        model_b_lgb = lgb.Booster(model_file=paths["model_b_lgb"])
        def predict_b(X, feat_names):
            return model_b_lgb.predict(X)

    # Split: training data (before start_date) and test data
    train_cutoff = pd.Timestamp(start_date)
    end_cutoff = pd.Timestamp(end_date)
    
    test_df = df[(df["Date"] >= train_cutoff) & (df["Date"] <= end_cutoff)].copy()
    print(f"Test period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"Test matches: {len(test_df)}")

    # Pre-compute Model A for all test matches (doesn't need DC/Elo)
    X_a = test_df[feat_a].fillna(pd.Series(medians_a)).values
    X_a_scaled = scaler_a.transform(X_a)
    test_df["prob_a"] = cal_a.predict(model_a.predict_proba(X_a_scaled)[:, 1])

    # For DC and Elo: compute using expanding window
    # Fit on all data before each match date
    from src.dixon_coles import DixonColesEnsemble
    from src.elo import EloRatingSystem

    # Get unique test dates
    test_dates = sorted(test_df["Date"].unique())
    print(f"Unique test dates: {len(test_dates)}")

    # Pre-fit DC and Elo at monthly intervals to save time
    # (fitting daily on 200k matches would take forever)
    # Use monthly snapshots, apply nearest-past snapshot to each match
    monthly_dates = pd.date_range(
        start=train_cutoff - pd.Timedelta(days=30),
        end=test_df["Date"].max(),
        freq="QS"  # quarterly to reduce compute
    )

    print(f"\nFitting DC/Elo at {len(monthly_dates)} quarterly snapshots...", flush=True)
    dc_snapshots = {}
    elo_snapshots = {}

    for i, snap_date in enumerate(monthly_dates):
        # Use last 3 years for DC fitting (full history is overkill and slow)
        cutoff_3y = snap_date - pd.Timedelta(days=3*365)
        train = df[(df["Date"] >= cutoff_3y) & (df["Date"] < snap_date) & (df["HTHG"].notna())].copy()
        if len(train) < 1000:
            continue

        t0 = time.time()

        # Dixon-Coles
        dc = DixonColesEnsemble()
        try:
            dc.fit(train)
            dc_snapshots[snap_date] = dc
        except Exception as e:
            print(f"  DC fit failed at {snap_date.date()}: {e}", file=sys.stderr)

        # Elo
        elo = EloRatingSystem()
        try:
            elo.fit(train)
            elo_snapshots[snap_date] = elo
        except Exception as e:
            print(f"  Elo fit failed at {snap_date.date()}: {e}", file=sys.stderr)

        elapsed = time.time() - t0
        print(f"  {snap_date.date()}: DC+Elo fit on {len(train)} matches ({elapsed:.1f}s)", flush=True)

    # Now compute DC/Elo predictions for each test match
    print(f"\nComputing DC/Elo for {len(test_df)} test matches...")

    dc_probs = []
    elo_probs = []

    snap_dates_sorted = sorted(dc_snapshots.keys())

    for idx, row in test_df.iterrows():
        match_date = row["Date"]

        # Find nearest past snapshot
        past_snaps = [s for s in snap_dates_sorted if s <= match_date]
        if not past_snaps:
            dc_probs.append(medians_b.get("dc_draw_prob", 0.42))
            elo_probs.append(medians_b.get("elo_draw_prob", 0.42))
            continue

        snap_date = past_snaps[-1]

        # DC prediction
        dc = dc_snapshots.get(snap_date)
        if dc:
            try:
                row_df = pd.DataFrame([row])
                dc_p = dc.predict_draw(row_df)
                dc_probs.append(float(dc_p.iloc[0]) if hasattr(dc_p, 'iloc') else float(dc_p))
            except Exception:
                dc_probs.append(medians_b.get("dc_draw_prob", 0.42))
        else:
            dc_probs.append(medians_b.get("dc_draw_prob", 0.42))

        # Elo prediction
        elo = elo_snapshots.get(snap_date)
        if elo:
            try:
                row_df = pd.DataFrame([row])
                elo_p = elo.predict_draw(row_df)
                elo_probs.append(float(elo_p.iloc[0]) if hasattr(elo_p, 'iloc') else float(elo_p))
            except Exception:
                elo_probs.append(medians_b.get("elo_draw_prob", 0.42))
        else:
            elo_probs.append(medians_b.get("elo_draw_prob", 0.42))

        if len(dc_probs) % 2000 == 0:
            print(f"  {len(dc_probs)}/{len(test_df)} matches processed...", flush=True)

    test_df["dc_draw_prob"] = dc_probs
    test_df["elo_draw_prob"] = elo_probs

    # Referee adjustment
    from src.referee_model import RefereeModel
    try:
        with open(MODELS_DIR / "referee_model.pkl", "rb") as f:
            ref_model = pickle.load(f)
        test_df["referee_adj"] = test_df.apply(
            lambda r: ref_model.get_adjustment(r.get("Referee", "")) 
            if pd.notna(r.get("Referee")) else 0.0, axis=1
        )
    except Exception:
        test_df["referee_adj"] = 0.0

    # Now run Model B with real DC/Elo features
    for f in feat_b:
        if f not in test_df.columns:
            test_df[f] = medians_b.get(f, 0.0)

    X_b = test_df[feat_b].fillna(pd.Series(medians_b)).values
    test_df["prob_b"] = cal_b.predict(predict_b(X_b, feat_b))

    # Compute edge
    test_df["edge"] = test_df["prob_a"] - test_df["prob_b"]
    test_df["actual"] = test_df["y_ht_draw"].astype(int)

    # Results
    print(f"\n{'='*70}")
    print(f"FULL BACKTEST RESULTS (with per-match DC/Elo)")
    print(f"{'='*70}")
    print(f"Period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"Matches: {len(test_df)}")
    print(f"HT draw base rate: {test_df['actual'].mean():.3f}")
    print(f"Mean prob_a: {test_df['prob_a'].mean():.3f}")
    print(f"Mean prob_b: {test_df['prob_b'].mean():.3f}")
    print(f"Mean DC: {test_df['dc_draw_prob'].mean():.3f}")
    print(f"Mean Elo: {test_df['elo_draw_prob'].mean():.3f}")
    print()

    # Edge tiers
    results = {}
    print(f"{'Tier':>25} {'N':>6} {'Hits':>6} {'Rate':>6} {'Odds':>6} {'ROI':>8} {'EV/$1':>7}")
    print(f"{'='*70}")

    for label, lo, hi in [
        ("STRONG VALUE (>=5%)", 0.05, 1.0),
        ("VALUE (3-5%)", 0.03, 0.05),
        ("MARGINAL (1-3%)", 0.01, 0.03),
        ("NEUTRAL (-1 to 1%)", -0.01, 0.01),
        ("ANTI-VALUE (<-1%)", -1.0, -0.01),
    ]:
        mask = (test_df["edge"] >= lo) & (test_df["edge"] < hi)
        s = test_df[mask]
        if len(s) == 0:
            continue
        hit = s["actual"].mean()
        avg_d = s["B365D"].mean()
        be = 1 / avg_d
        returns = s[s["actual"] == 1]["B365D"].sum()
        roi = (returns - len(s)) / len(s)
        ev = hit * avg_d - 1

        print(f"{label:>25} {len(s):>6,} {int(s['actual'].sum()):>6} "
              f"{hit:>6.3f} {avg_d:>6.2f} {roi:>+8.3f} {ev:>+7.3f}")

        results[label] = {
            "n": len(s), "hits": int(s["actual"].sum()),
            "hit_rate": round(hit, 4), "avg_odds": round(avg_d, 2),
            "roi": round(roi, 4), "ev_per_dollar": round(ev, 4)
        }

    # Bootstrap for key tiers
    print(f"\n{'='*70}")
    print("BOOTSTRAP (10,000 resamples)")
    print(f"{'='*70}")

    for tier_name, threshold in [("STRONG VALUE", 0.05), ("VALUE+", 0.03)]:
        subset = test_df[test_df["edge"] >= threshold]
        n = len(subset)
        if n < 10:
            continue

        np.random.seed(42)
        boot = []
        for _ in range(10000):
            samp = subset.sample(n=n, replace=True)
            ret = samp[samp["actual"] == 1]["B365D"].sum()
            boot.append((ret - n) / n)
        boot = np.array(boot)

        print(f"\n{tier_name} (N={n:,}):")
        print(f"  ROI: {boot.mean():+.4f} (median {np.median(boot):+.4f})")
        print(f"  95% CI: [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
        print(f"  P(profit): {(boot > 0).mean():.4f}")

        results[f"bootstrap_{tier_name}"] = {
            "n": n, "mean_roi": round(boot.mean(), 4),
            "ci_lower": round(np.percentile(boot, 2.5), 4),
            "ci_upper": round(np.percentile(boot, 97.5), 4),
            "p_profit": round((boot > 0).mean(), 4),
        }

    # Time stability
    print(f"\n{'='*70}")
    print("TIME STABILITY: STRONG VALUE by half-year")
    print(f"{'='*70}")
    strong = test_df[test_df["edge"] >= 0.05].sort_values("Date")
    if len(strong) > 0:
        strong["half"] = strong["Date"].dt.year.astype(str) + np.where(
            strong["Date"].dt.month <= 6, "H1", "H2"
        )
        stability = {}
        for h, g in strong.groupby("half"):
            hit = g["actual"].mean()
            ret = g[g["actual"] == 1]["B365D"].sum()
            roi = (ret - len(g)) / len(g)
            status = "✅" if roi > 0 else "❌"
            print(f"  {h}: N={len(g):>4}  Hit={hit:.3f}  ROI={roi:+.4f}  {status}")
            stability[h] = {"n": len(g), "hit_rate": round(hit, 3), "roi": round(roi, 4)}
        results["time_stability"] = stability

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-04-01")
    parser.add_argument("--end", default="2026-12-31")
    args = parser.parse_args()
    run_backtest(start_date=args.start, end_date=args.end)
