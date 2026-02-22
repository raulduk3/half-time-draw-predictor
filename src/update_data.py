"""
Data Update Pipeline — Half-Time Draw Predictor
================================================
INCREMENTAL update: downloads new match results, appends to existing
dataset, computes rolling features only for new rows.

Does NOT retrain ML models (Model A/B). Updates:
  - Mega dataset with new matches + rolling features
  - Dixon-Coles attack/defense parameters
  - Elo ratings
  - Referee draw-rate profiles

Usage:
    python src/update_data.py              # incremental update
    python src/update_data.py --full       # full rebuild (slow, ~10 min)
    python src/update_data.py --dry-run    # show current freshness
    python src/update_data.py --download-only  # just download CSVs
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, ".")

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw_all")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/v4")
MEGA_PATH     = PROCESSED_DIR / "mega_dataset_v2.parquet"

BASE_URL = "https://www.football-data.co.uk"

LEAGUES = {
    "E0":  ("england", "E0"),
    "E1":  ("england", "E1"),
    "E2":  ("england", "E2"),
    "E3":  ("england", "E3"),
    "SP1": ("spain",   "SP1"),
    "SP2": ("spain",   "SP2"),
    "D1":  ("germany", "D1"),
    "D2":  ("germany", "D2"),
    "I1":  ("italy",   "I1"),
    "I2":  ("italy",   "I2"),
    "F1":  ("france",  "F1"),
    "F2":  ("france",  "F2"),
    "N1":  ("netherlands", "N1"),
    "B1":  ("belgium", "B1"),
    "P1":  ("portugal", "P1"),
    "G1":  ("greece",  "G1"),
    "T1":  ("turkey",  "T1"),
    "SC0": ("scotland", "SC0"),
    "SC1": ("scotland", "SC1"),
    "SC2": ("scotland", "SC2"),
    "SC3": ("scotland", "SC3"),
    "EC":  ("england", "EC"),
}

CURRENT_SEASON = "2526"

# Stats used for rolling features (must match build_mega_dataset.py)
STAT_COLS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]


# ── Download ──────────────────────────────────────────────────────────────────

def download_current_season(force: bool = False) -> int:
    """Download current season CSVs. Returns number of files updated."""
    updated = 0
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for league_code, (country, div) in LEAGUES.items():
        league_dir = RAW_DIR / league_code
        league_dir.mkdir(parents=True, exist_ok=True)
        out_file = league_dir / f"{CURRENT_SEASON}.csv"

        url = f"{BASE_URL}/mmz4281/{CURRENT_SEASON}/{div}.csv"

        try:
            req = urllib.request.Request(url)
            if out_file.exists() and not force:
                mtime = os.path.getmtime(out_file)
                req.add_header("If-Modified-Since",
                    time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(mtime)))

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                out_file.write_bytes(data)
                updated += 1
                df = pd.read_csv(io.BytesIO(data), encoding="latin1")
                print(f"  ✅ {league_code}: {len(df)} matches", file=sys.stderr)

        except urllib.error.HTTPError as e:
            if e.code == 304:
                print(f"  — {league_code}: up to date", file=sys.stderr)
            else:
                print(f"  ❌ {league_code}: HTTP {e.code}", file=sys.stderr)
        except Exception as e:
            print(f"  ❌ {league_code}: {e}", file=sys.stderr)

    return updated


# ── Incremental Feature Computation ──────────────────────────────────────────

def _compute_rolling_for_team(team: str, history_df: pd.DataFrame,
                               match_date: pd.Timestamp, window: int = 5) -> Dict[str, float]:
    """Compute rolling features for a single team at a given date. Fast path."""
    hist = history_df[
        (history_df["Date"] < match_date) &
        ((history_df["HomeTeam"] == team) | (history_df["AwayTeam"] == team))
    ].tail(window)

    if len(hist) == 0:
        return {}

    result = {}

    # Goal stats (from FT or HT depending on availability)
    gf, ga = [], []
    for _, row in hist.iterrows():
        if row["HomeTeam"] == team:
            if pd.notna(row.get("FTHG")): gf.append(row["FTHG"])
            if pd.notna(row.get("FTAG")): ga.append(row["FTAG"])
        else:
            if pd.notna(row.get("FTAG")): gf.append(row["FTAG"])
            if pd.notna(row.get("FTHG")): ga.append(row["FTHG"])

    if gf:
        result["gf_r5"] = np.mean(gf)
        result["ga_r5"] = np.mean(ga)
        result["gd_r5"] = np.mean(gf) - np.mean(ga)

    # Match stat rolling averages
    stat_pairs = [
        ("HS", "AS"), ("HST", "AST"), ("HC", "AC"),
        ("HF", "AF"), ("HY", "AY"), ("HR", "AR"),
    ]
    for home_stat, away_stat in stat_pairs:
        if home_stat not in hist.columns:
            continue
        vals = []
        for _, row in hist.iterrows():
            if row["HomeTeam"] == team:
                if pd.notna(row.get(home_stat)): vals.append(row[home_stat])
            else:
                if pd.notna(row.get(away_stat)): vals.append(row[away_stat])
        if vals:
            col_name = home_stat.lower()
            result[f"{col_name}_r5"] = np.mean(vals)

    # Shot accuracy ratios
    for shot_stat, target_stat in [("HS", "HST"), ("AS", "AST")]:
        shots, targets = [], []
        for _, row in hist.iterrows():
            if row["HomeTeam"] == team:
                s_col, t_col = shot_stat, target_stat
            else:
                s_col = shot_stat.replace("H", "A") if "H" in shot_stat else shot_stat.replace("A", "H")
                t_col = target_stat.replace("H", "A") if "H" in target_stat else target_stat.replace("A", "H")
            if pd.notna(row.get(s_col)) and pd.notna(row.get(t_col)):
                shots.append(row[s_col])
                targets.append(row[t_col])
        if shots and sum(shots) > 0:
            base = shot_stat.lower().rstrip("t")
            result[f"{base}_ratio_r5"] = sum(targets) / sum(shots)

    # Days since last match
    if len(hist) > 0:
        last_date = hist["Date"].max()
        result["days_since_last"] = max(0, (match_date - last_date).days)

    return result


def compute_features_for_new_rows(existing_df: pd.DataFrame,
                                   new_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling features for new rows using existing dataset as history.
    Much faster than recomputing everything from scratch.
    """
    # Combine for history lookup but only compute features for new rows
    full_df = pd.concat([existing_df, new_rows], ignore_index=True)
    full_df = full_df.sort_values("Date").reset_index(drop=True)

    new_dates = set(new_rows["Date"].unique())
    new_mask = full_df["Date"].isin(new_dates)

    print(f"  Computing features for {new_mask.sum()} new matches...", file=sys.stderr)

    for idx in full_df[new_mask].index:
        row = full_df.loc[idx]
        match_date = row["Date"]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # Home team rolling stats
        home_stats = _compute_rolling_for_team(home_team, full_df, match_date)
        for k, v in home_stats.items():
            col = f"home_{k}"
            if col in full_df.columns:
                full_df.at[idx, col] = v

        # Away team rolling stats
        away_stats = _compute_rolling_for_team(away_team, full_df, match_date)
        for k, v in away_stats.items():
            col = f"away_{k}"
            if col in full_df.columns:
                full_df.at[idx, col] = v

    return full_df


# ── Incremental Update ───────────────────────────────────────────────────────

def find_new_matches() -> pd.DataFrame:
    """
    Compare current season CSVs against mega dataset to find new matches.
    Returns DataFrame of matches not yet in the dataset.
    """
    if not MEGA_PATH.exists():
        print("  No existing dataset found. Run with --full first.", file=sys.stderr)
        return pd.DataFrame()

    existing = pd.read_parquet(MEGA_PATH)
    existing["Date"] = pd.to_datetime(existing["Date"])
    latest_date = existing["Date"].max()

    print(f"  Existing dataset: {len(existing)} matches through {latest_date.date()}", file=sys.stderr)

    new_rows = []
    for league_code in LEAGUES:
        league_dir = RAW_DIR / league_code
        csv_file = league_dir / f"{CURRENT_SEASON}.csv"
        if not csv_file.exists():
            continue

        try:
            df = pd.read_csv(csv_file, encoding="latin1")
            df.columns = [c.strip() for c in df.columns]

            if "HomeTeam" not in df.columns:
                continue

            # Parse dates
            for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors="raise")
                    break
                except Exception:
                    continue
            else:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            df = df[df["Date"].notna()].copy()

            # Filter to matches after our latest date
            new = df[df["Date"] > latest_date].copy()
            if len(new) > 0:
                new["league"] = league_code
                new_rows.append(new)
                print(f"  📊 {league_code}: {len(new)} new matches", file=sys.stderr)

        except Exception as e:
            print(f"  ❌ {league_code}: {e}", file=sys.stderr)

    if not new_rows:
        print("  No new matches found.", file=sys.stderr)
        return pd.DataFrame()

    combined = pd.concat(new_rows, ignore_index=True)
    print(f"  Found {len(combined)} new matches total", file=sys.stderr)
    return combined


def incremental_update() -> Optional[pd.DataFrame]:
    """
    Incrementally update the mega dataset with new matches.
    Returns updated DataFrame or None if no updates needed.
    """
    new_matches = find_new_matches()
    if new_matches.empty:
        return None

    existing = pd.read_parquet(MEGA_PATH)
    existing["Date"] = pd.to_datetime(existing["Date"])

    # Ensure new matches have required columns
    required = ["Date", "HomeTeam", "AwayTeam"]
    for col in required:
        if col not in new_matches.columns:
            print(f"  ❌ Missing column: {col}", file=sys.stderr)
            return None

    # Add target column
    if "HTHG" in new_matches.columns and "HTAG" in new_matches.columns:
        new_matches["y_ht_draw"] = (new_matches["HTHG"] == new_matches["HTAG"]).astype(float)
        new_matches.loc[new_matches["HTHG"].isna(), "y_ht_draw"] = np.nan
    else:
        new_matches["y_ht_draw"] = np.nan

    # Add log odds
    for odds_col, log_col in [("B365H", "log_home_win_odds"),
                                ("B365D", "log_draw_odds"),
                                ("B365A", "log_away_win_odds")]:
        if odds_col in new_matches.columns:
            new_matches[log_col] = np.log(new_matches[odds_col].clip(lower=1.01))
        else:
            new_matches[log_col] = np.nan

    # Ensure all existing columns exist in new data
    for col in existing.columns:
        if col not in new_matches.columns:
            new_matches[col] = np.nan

    # Compute rolling features for new rows
    updated = compute_features_for_new_rows(existing, new_matches[existing.columns])

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(MEGA_PATH, index=False)
    print(f"  ✅ Dataset updated: {len(updated)} matches through {updated['Date'].max().date()}", file=sys.stderr)

    return updated


def refit_submodels(df: pd.DataFrame) -> None:
    """Re-fit Dixon-Coles and Elo on the updated dataset."""
    from src.dixon_coles import DixonColesEnsemble
    from src.elo import EloRatingSystem
    from src.referee_model import RefereeModel

    df_ht = df[df["HTHG"].notna() & df["HTAG"].notna()].copy()

    print(f"  Fitting Dixon-Coles on {len(df_ht)} matches...", file=sys.stderr)
    dc = DixonColesEnsemble()
    dc.fit(df_ht)
    dc.save(str(MODELS_DIR / "dixon_coles.pkl"))
    print("  ✅ Dixon-Coles updated", file=sys.stderr)

    print(f"  Fitting Elo on {len(df_ht)} matches...", file=sys.stderr)
    elo = EloRatingSystem()
    elo.fit(df_ht)
    elo.save(str(MODELS_DIR / "elo.pkl"))
    print("  ✅ Elo updated", file=sys.stderr)

    if "Referee" in df_ht.columns:
        df_ref = df_ht[df_ht["Referee"].notna()].copy()
        if len(df_ref) > 1000:
            print(f"  Fitting referee model on {len(df_ref)} matches...", file=sys.stderr)
            ref = RefereeModel()
            ref.fit(df_ref)
            with open(MODELS_DIR / "referee_model.pkl", "wb") as f:
                pickle.dump(ref, f)
            print("  ✅ Referee model updated", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Update half-time draw predictor data")
    parser.add_argument("--full", action="store_true",
                        help="Full rebuild from scratch (slow, ~10 min)")
    parser.add_argument("--skip-refit", action="store_true",
                        help="Skip refitting Dixon-Coles/Elo")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download CSVs, don't rebuild")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download even if files are current")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show current data freshness")
    args = parser.parse_args()

    if args.dry_run:
        if MEGA_PATH.exists():
            df = pd.read_parquet(MEGA_PATH)
            latest = pd.to_datetime(df["Date"]).max()
            print(f"Current dataset: {len(df)} matches, {df['league'].nunique() if 'league' in df.columns else '?'} leagues")
            print(f"Latest match: {latest.date()}")
            print(f"Age: {(pd.Timestamp.today() - latest).days} days")
        else:
            print("No mega dataset found. Run 'python src/update_data.py --full' to build.")
        return

    # Step 1: Download
    print("\n📥 Downloading current season data...", file=sys.stderr)
    n_updated = download_current_season(force=args.force_download)
    print(f"  {n_updated} league files updated\n", file=sys.stderr)

    if args.download_only:
        return

    # Step 2: Update dataset
    if args.full:
        print("🔄 Full rebuild (this will take ~10 minutes)...", file=sys.stderr)
        from src.build_mega_dataset import load_and_process_all_data, save_mega_dataset
        combined_df, stats, encoders = load_and_process_all_data()
        if combined_df is None:
            print("  ❌ Failed to load data", file=sys.stderr)
            return
        save_mega_dataset(combined_df, stats, encoders)
        df = combined_df
        print(f"  ✅ Full rebuild: {len(df)} matches through {df['Date'].max().date()}", file=sys.stderr)
    else:
        print("🔄 Incremental update...", file=sys.stderr)
        df = incremental_update()
        if df is None:
            print("  Dataset is current. Nothing to update.", file=sys.stderr)
            return

    # Step 3: Refit sub-models
    if not args.skip_refit:
        print("\n📊 Refitting sub-models...", file=sys.stderr)
        refit_submodels(df)

    print(f"\n✅ Update complete. {len(df)} matches through {df['Date'].max().date()}", file=sys.stderr)


if __name__ == "__main__":
    main()
