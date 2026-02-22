"""
MLS Integration — Surgical Update of V2 Models
================================================
Integrates MLS data from data/raw_all/USA/USA_current.csv into the
existing trained v2 models WITHOUT re-running the full training pipeline.

Changes made:
  1. Elo   — replay all MLS matches to give every current MLS team a rating
             (7 expansion teams were missing: Austin FC, Charlotte, FC Cincinnati,
              Inter Miami, Nashville SC, San Diego FC, St. Louis City)
  2. Dixon-Coles — set ht_scale_=0.45 on the USA_MLS (and MEX_LigaMX) league model
                   so P(draw) is calibrated to half-time scoring rates, not FT rates
  3. Market model — re-fit training medians including MLS closing odds
                    (Pinnacle, MaxC, AvgC) so imputation is MLS-aware

Run:
    python src/integrate_mls.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.elo import EloRatingSystem
from src.dixon_coles import DixonColes, DixonColesEnsemble
from src.market_model import MarketModel, extract_multi_book_features

MLS_CSV   = Path("data/raw_all/USA/USA_current.csv")
V2_DIR    = Path("models/v2")
FT_ONLY_LEAGUES = {"USA_MLS", "MEX_LigaMX"}
HT_SCALE  = 0.45   # HT goals ≈ 45% of FT goals


# ── 1. Parse MLS CSV ──────────────────────────────────────────────────────────

def load_mls_df() -> pd.DataFrame:
    """Load and normalise USA_current.csv into the standard mega-dataset schema."""
    print(f"  Loading {MLS_CSV} ...")
    df = pd.read_csv(MLS_CSV, encoding="utf-8-sig")
    print(f"  Raw rows: {len(df):,}  |  columns: {list(df.columns)}")

    rename = {
        "Home":   "HomeTeam",
        "Away":   "AwayTeam",
        "HG":     "FTHG",
        "AG":     "FTAG",
    }
    # B365 closing — USA file has a typo: B36CA instead of B365CA
    if "B365CH" in df.columns:
        rename["B365CH"] = "B365H"
    if "B365CD" in df.columns:
        rename["B365CD"] = "B365D"
    if "B36CA" in df.columns and "B365CA" not in df.columns:
        rename["B36CA"] = "B365A"
    elif "B365CA" in df.columns:
        rename["B365CA"] = "B365A"

    df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["FTHG", "FTAG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    df["HTHG"]   = np.nan
    df["HTAG"]   = np.nan
    df["league"] = "USA_MLS"

    for col in ["PSCH", "PSCD", "PSCA",
                "MaxCH", "MaxCD", "MaxCA",
                "AvgCH", "AvgCD", "AvgCA",
                "BFECH", "BFECD", "BFECA",
                "B365H", "B365D", "B365A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing B365 from AvgC (closing consensus) as fallback
    for side, avg_col in [("B365H", "AvgCH"), ("B365D", "AvgCD"), ("B365A", "AvgCA")]:
        if side in df.columns and avg_col in df.columns:
            df[side] = df[side].fillna(df[avg_col])

    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Valid MLS matches: {len(df):,}  "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    print(f"  Unique teams: {len(teams)}")
    return df


# ── 2. Update Elo ─────────────────────────────────────────────────────────────

def update_elo(mls_df: pd.DataFrame, elo_path: str = str(V2_DIR / "elo.pkl")) -> None:
    print("\n[2] Updating Elo ratings with MLS matches...")
    with open(elo_path, "rb") as f:
        elo: EloRatingSystem = pickle.load(f)

    print(f"  Current Elo teams: {len(elo.ratings_):,}")

    # Find MLS teams already in ratings and teams not yet seen
    mls_teams = sorted(set(mls_df["HomeTeam"]) | set(mls_df["AwayTeam"]))
    missing   = [t for t in mls_teams if t not in elo.ratings_]
    print(f"  MLS teams missing from Elo ({len(missing)}): {missing}")

    # Replay ALL MLS matches to bring ratings up to date
    elo.extend_ratings(mls_df)

    # Verify
    still_missing = [t for t in mls_teams if t not in elo.ratings_]
    if still_missing:
        print(f"  WARNING: still missing after extend: {still_missing}")
    else:
        print(f"  All {len(mls_teams)} MLS teams now rated.")

    # Sample ratings
    sample = ["Los Angeles Galaxy", "New York City", "Austin FC",
              "Inter Miami", "Charlotte", "Nashville SC"]
    print("  Sample ratings:")
    for t in sample:
        r = elo.ratings_.get(t, "NOT FOUND")
        print(f"    {t:<30}  {r}")

    elo.save(elo_path)


# ── 3. Fix Dixon-Coles ht_scale_ ─────────────────────────────────────────────

def fix_dixon_coles(dc_path: str = str(V2_DIR / "dixon_coles.pkl")) -> None:
    print("\n[3] Fixing Dixon-Coles ht_scale_ for FT-only leagues...")
    with open(dc_path, "rb") as f:
        dc: DixonColesEnsemble = pickle.load(f)

    for league, model in dc.league_models_.items():
        old_scale = getattr(model, "ht_scale_", 1.0)
        if league in FT_ONLY_LEAGUES:
            model.ht_scale_ = HT_SCALE
            print(f"  {league}: ht_scale {old_scale:.2f} → {model.ht_scale_:.2f}")
        else:
            # Ensure attribute exists for backward compat (no-op for HT leagues)
            model.ht_scale_ = 1.0

    # Verify predictions changed
    mls_model = dc.league_models_.get("USA_MLS")
    if mls_model:
        print(f"\n  MLS model train_draw_rate_: {mls_model.train_draw_rate_:.4f}")
        test_pairs = [
            ("Los Angeles Galaxy", "New York City"),
            ("Seattle Sounders",   "Portland Timbers"),
            ("Austin FC",          "Inter Miami"),
        ]
        print("  Sample predictions after ht_scale fix:")
        for h, a in test_pairs:
            p_direct = mls_model.predict_draw_proba(h, a)
            p_ens    = dc.predict_draw_single(h, a, league="USA_MLS")
            print(f"    {h} vs {a}: direct={p_direct:.4f}  ensemble={p_ens:.4f}")

    dc.save(dc_path)


# ── 4. Update Market Model ────────────────────────────────────────────────────

def update_market_model(mls_df: pd.DataFrame,
                        mm_path: str = str(V2_DIR / "market_model.pkl")) -> None:
    print("\n[4] Updating market model medians with MLS closing odds...")
    with open(mm_path, "rb") as f:
        mm: MarketModel = pickle.load(f)

    print(f"  Existing train medians (sample):")
    for k in ["consensus_draw_prob", "pinnacle_close_prob", "n_books"]:
        print(f"    {k}: {mm.train_medians_.get(k, 'N/A'):.4f}")

    # Extract market features from MLS data
    mls_feats = extract_multi_book_features(mls_df)
    mls_coverage = mls_feats.notna().mean()
    print(f"\n  MLS market feature coverage:")
    for col in ["consensus_draw_prob", "max_draw_prob", "pinnacle_close_prob", "n_books"]:
        print(f"    {col}: {mls_coverage.get(col, 0):.1%}")

    # Update medians: take weighted average of existing and MLS medians
    # (existing medians represent European matches, MLS adds new signal)
    mls_medians = {
        col: float(mls_feats[col].median())
        for col in mm.FEATURE_COLS
        if col in mls_feats.columns and mls_feats[col].notna().any()
    }

    # Blend: 80% existing + 20% MLS (conservative — MLS is FT odds, not HT)
    for col, mls_val in mls_medians.items():
        if col in mm.train_medians_ and np.isfinite(mls_val):
            old_val = mm.train_medians_[col]
            mm.train_medians_[col] = 0.80 * old_val + 0.20 * mls_val

    print(f"\n  Updated train medians (sample):")
    for k in ["consensus_draw_prob", "pinnacle_close_prob", "n_books"]:
        print(f"    {k}: {mm.train_medians_.get(k, 'N/A'):.4f}")

    mm.save(mm_path)


# ── 5. Print comparison table ─────────────────────────────────────────────────

def print_comparison(elo_path: str = str(V2_DIR / "elo.pkl"),
                     dc_path: str  = str(V2_DIR / "dixon_coles.pkl")) -> None:
    print("\n" + "=" * 68)
    print("COMPARISON: OLD (generic) vs NEW (MLS-specific) sub-model outputs")
    print("=" * 68)

    with open(elo_path, "rb") as f:
        elo: EloRatingSystem = pickle.load(f)
    with open(dc_path, "rb") as f:
        dc: DixonColesEnsemble = pickle.load(f)

    rows = [
        ("LA Galaxy",        "NYCFC",         "USA_MLS"),
        ("Los Angeles Galaxy","New York City", "USA_MLS"),
        ("Seattle Sounders", "Portland Timbers", "USA_MLS"),
        ("Inter Miami",      "Austin FC",      "USA_MLS"),
        ("Everton",          "Man United",     "E0"),
        ("Leeds United",     "Sheffield Utd",  "E1"),
    ]

    print(f"\n{'Match':<46} {'DC':>7} {'Elo':>7}")
    print("-" * 62)
    for home, away, league in rows:
        dc_p  = dc.predict_draw_single(home, away, league=league)
        elo_p = elo.predict_draw_single(home, away)
        print(f"  {home:<20} vs {away:<20}  {dc_p:.3f}   {elo_p:.3f}")

    print()
    print("OLD global fallback: DC=0.4200  Elo=0.4270  (generic if team unknown)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("MLS INTEGRATION — Updating V2 Models")
    print("=" * 68)

    if not MLS_CSV.exists():
        print(f"ERROR: {MLS_CSV} not found.")
        sys.exit(1)

    mls_df = load_mls_df()

    update_elo(mls_df)
    fix_dixon_coles()
    update_market_model(mls_df)
    print_comparison()

    print("\n" + "=" * 68)
    print("Integration complete. Models saved to models/v2/")
    print("=" * 68)
