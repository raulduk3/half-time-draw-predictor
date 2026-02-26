#!/usr/bin/env python3
"""
Merge real HT draw odds (from ht_draw_odds.csv) with existing match data.

The HT draw odds use Odds API team names while the match data uses
football-data.co.uk team names — these need fuzzy matching.

Usage:
    python src/merge_ht_odds.py [--output data/processed/merged_ht_odds.csv]
"""

from __future__ import annotations
import csv
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent.parent / "data"
HT_ODDS_FILE = DATA_DIR / "historical_ht_odds" / "ht_draw_odds.csv"
H2H_FILE = DATA_DIR / "historical_ht_odds" / "historical_h2h_odds.json"
RAW_ALL_DIR = DATA_DIR / "raw_all"

# League mapping: our codes → raw_all subdirs
LEAGUE_DIRS = {
    "E0": "E0", "SP1": "SP1", "I1": "I1", "D1": "D1", "F1": "F1",
}

# Season → file mapping (e.g. 2023-24 → raw_all/E0/2324.csv)
def season_to_filename(season: str) -> str:
    """Convert '2023-24' → '2324'"""
    parts = season.split("-")
    if len(parts) == 2:
        y1 = parts[0][-2:]
        y2 = parts[1][-2:]
        return f"{y1}{y2}"
    return season

# Simple team name normalization
def normalize_name(name: str) -> str:
    name = name.lower().strip()
    # Remove common suffixes/words
    for suffix in [" fc", " cf", " af", " sc", " ac", " us", " afc", " united", " city",
                   " athletic", " albion", " wanderers", " rovers", " town", " county",
                   " hotspur", " palace", " villa"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    # Remove non-alpha chars
    name = re.sub(r"[^a-z\s]", "", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def name_similarity(a: str, b: str) -> float:
    """Simple character n-gram similarity."""
    a_n = normalize_name(a)
    b_n = normalize_name(b)
    if a_n == b_n:
        return 1.0
    # Check if one is substring of other
    if a_n in b_n or b_n in a_n:
        return 0.8
    # Count matching chars
    shorter = min(len(a_n), len(b_n))
    if shorter == 0:
        return 0.0
    matching = sum(c1 == c2 for c1, c2 in zip(sorted(a_n), sorted(b_n)))
    return matching / max(len(a_n), len(b_n))


def load_raw_match_data(league: str, season: str) -> List[Dict]:
    """Load a single season's match data from raw_all."""
    season_file = season_to_filename(season)
    league_dir = RAW_ALL_DIR / LEAGUE_DIRS.get(league, league)
    csv_path = league_dir / f"{season_file}.csv"

    if not csv_path.exists():
        return []

    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def match_ht_to_raw(ht_row: Dict, raw_rows: List[Dict]) -> Optional[Dict]:
    """
    Find the raw match that corresponds to a HT odds row.
    Match on: date (±1 day), home team (fuzzy), away team (fuzzy).
    """
    try:
        ht_dt = datetime.fromisoformat(ht_row["commence_time"].replace("Z", "+00:00"))
        ht_date = ht_dt.date()
    except Exception:
        return None

    ht_home = ht_row["home_team"]
    ht_away = ht_row["away_team"]

    best_match = None
    best_score = 0.0

    for raw in raw_rows:
        # Parse raw date
        raw_date_str = raw.get("Date", "")
        raw_date = None
        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
            try:
                raw_date = datetime.strptime(raw_date_str, fmt).date()
                break
            except Exception:
                continue

        if raw_date is None:
            continue

        # Date must be within 1 day
        if abs((ht_date - raw_date).days) > 1:
            continue

        raw_home = raw.get("HomeTeam", "")
        raw_away = raw.get("AwayTeam", "")

        home_sim = name_similarity(ht_home, raw_home)
        away_sim = name_similarity(ht_away, raw_away)
        score = (home_sim + away_sim) / 2

        if score > best_score and score > 0.5:
            best_score = score
            best_match = raw

    return best_match


def merge() -> List[Dict]:
    """Merge HT odds with raw match data."""
    # Load HT odds
    with open(HT_ODDS_FILE) as f:
        ht_rows = list(csv.DictReader(f))

    print(f"HT odds rows: {len(ht_rows)}")

    merged = []
    skipped = 0
    cache: Dict[str, List[Dict]] = {}

    for ht in ht_rows:
        league = ht["league"]
        season = ht.get("season", "unknown")

        cache_key = f"{league}_{season}"
        if cache_key not in cache:
            cache[cache_key] = load_raw_match_data(league, season)

        raw_rows = cache[cache_key]
        if not raw_rows:
            skipped += 1
            continue

        raw = match_ht_to_raw(ht, raw_rows)
        if raw is None:
            skipped += 1
            continue

        # Merge: HT odds + raw match outcome
        merged_row = {
            **ht,
            "raw_home": raw.get("HomeTeam", ""),
            "raw_away": raw.get("AwayTeam", ""),
            "raw_date": raw.get("Date", ""),
            "FTHG": raw.get("FTHG", ""),
            "FTAG": raw.get("FTAG", ""),
            "FTR": raw.get("FTR", ""),
            "HTHG": raw.get("HTHG", ""),
            "HTAG": raw.get("HTAG", ""),
            "HTR": raw.get("HTR", ""),
            "HT_draw": "1" if raw.get("HTR", "") == "D" else "0",
            "b365d": raw.get("B365D", raw.get("BbAvD", "")),  # FT draw odds for comparison
        }
        merged.append(merged_row)

    print(f"Merged: {len(merged)}, Skipped: {skipped}")
    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DATA_DIR / "processed" / "ht_odds_merged.csv"))
    args = parser.parse_args()

    merged = merge()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if merged:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
            writer.writeheader()
            writer.writerows(merged)
        print(f"Saved to {out_path}")

        # Quick summary
        ht_draws = sum(1 for r in merged if r["HTR"] == "D")
        print(f"\nSummary:")
        print(f"  Total matches: {len(merged)}")
        print(f"  HT draws: {ht_draws} ({100*ht_draws/len(merged):.1f}%)")

        # Calibration check
        try:
            pinnacle_odds = [float(r["ht_draw_pinnacle"]) for r in merged if r.get("ht_draw_pinnacle")]
            implied_prob_avg = sum(1/x for x in pinnacle_odds) / len(pinnacle_odds)
            actual_prob = ht_draws / len(merged)
            print(f"  Implied HT draw prob (Pinnacle): {implied_prob_avg:.3f}")
            print(f"  Actual HT draw rate:             {actual_prob:.3f}")
        except Exception as e:
            print(f"  Calibration check error: {e}")
    else:
        print("No rows to save.")


if __name__ == "__main__":
    main()
