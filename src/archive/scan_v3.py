"""
V3 Scanner — Upcoming Fixtures Ranked by Edge
==============================================
Loads upcoming fixtures and ranks them by:
  EDGE = Model B (fundamentals) − Model A (market estimate)

Filters:
  - Upcoming matches only (date >= today: 2026-02-22)
  - Matches with available odds
  - Shows only positive-edge matches (sorted descending by edge)

Usage:
    # Demo mode — uses hardcoded example matches
    python src/scan_v3.py --demo

    # From a CSV file of fixtures + odds
    python src/scan_v3.py --fixtures fixtures.csv

    # Run on specific matches
    python src/scan_v3.py --match "Everton" "Man United" 3.70 3.75 1.91 E0 \\
                          --match "Alaves" "Girona" 2.30 3.00 3.50 SP1 \\
                          --match "LA Galaxy" "NYCFC" 2.25 3.60 2.80 USA_MLS

The demo mode runs the 3 requested example matches automatically.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

TODAY = pd.Timestamp("2026-02-22")

# Rating thresholds
STRONG_EDGE  = 0.05
VALUE_EDGE   = 0.03
MARGINAL_EDGE = 0.01


def run_scan(
    fixtures: List[Dict],
    paths_file: str = "models/v3/v3_paths.json",
    min_edge: float = 0.0,
    show_all: bool = False,
) -> List[Dict]:
    """
    Run the two-model scanner on a list of fixtures.

    Each fixture dict requires:
        home, away, b365h, b365d, b365a
    Optional:
        league, referee, date

    Returns list of result dicts sorted by edge descending.
    """
    from src.predict_match_v3 import V3Predictor

    print("Loading V3 predictor...")
    predictor = V3Predictor.load(paths_file=paths_file)

    results = []
    for fix in fixtures:
        try:
            r = predictor.predict(
                home_team = fix["home"],
                away_team = fix["away"],
                b365h     = float(fix["b365h"]),
                b365d     = float(fix["b365d"]),
                b365a     = float(fix["b365a"]),
                league    = fix.get("league", ""),
                referee   = fix.get("referee"),
            )
            r["date"] = fix.get("date", str(TODAY.date()))
            results.append(r)
        except Exception as e:
            print(f"  Warning: failed to predict {fix.get('home')} vs {fix.get('away')}: {e}")

    # Sort by edge descending
    results.sort(key=lambda x: x["edge"], reverse=True)

    return results


def print_scan_results(results: List[Dict], min_edge: float = 0.0) -> None:
    """Pretty-print scanner results ranked by edge."""

    positive = [r for r in results if r["edge"] >= min_edge]
    negative = [r for r in results if r["edge"] < min_edge]

    print("\n" + "═" * 72)
    print("  V3 SCANNER — HALF-TIME DRAW VALUE FINDER")
    print("  Ranked by EDGE (Model B fundamentals − Model A market estimate)")
    print("═" * 72)
    print(f"  {len(results)} fixtures scanned  |  "
          f"{len(positive)} with edge ≥ {min_edge:.0%}  |  "
          f"Today: {TODAY.date()}")
    print()

    if not positive:
        print("  No matches with positive edge found.")
        return

    # Header
    print(f"  {'#':>2}  {'Match':<38} {'League':>8}  {'A-odds':>7}  "
          f"{'MktA':>6} {'FundB':>6} {'Edge':>7}  {'Rating':<14}")
    print(f"  {'-'*2}  {'-'*38} {'-'*8}  {'-'*7}  "
          f"{'-'*6} {'-'*6} {'-'*7}  {'-'*14}")

    for i, r in enumerate(positive, 1):
        match_str = f"{r['home_team']} vs {r['away_team']}"
        if len(match_str) > 38:
            match_str = match_str[:35] + "..."
        edge_str = f"{r['edge_pct']:+.2f}%"
        rating   = r["rating"]
        if r["edge"] >= STRONG_EDGE:
            rating_display = f"★★★ {rating}"
        elif r["edge"] >= VALUE_EDGE:
            rating_display = f"★★  {rating}"
        elif r["edge"] >= MARGINAL_EDGE:
            rating_display = f"★   {rating}"
        else:
            rating_display = f"—   {rating}"

        print(f"  {i:>2}  {match_str:<38} {r['league']:>8}  "
              f"{r['b365d']:>7.2f}  "
              f"{r['model_a_prob']:>6.1%} {r['model_b_prob']:>6.1%} {edge_str:>7}  "
              f"{rating_display:<14}")

    print()

    # Detailed view for top matches
    top_n = min(len(positive), 5)
    if top_n > 0:
        print("─" * 72)
        print(f"  TOP {top_n} MATCH DETAILS:")
        from src.predict_match_v3 import V3Predictor
        for r in positive[:top_n]:
            V3Predictor.print_result(r)

    # Summary stats
    if len(positive) >= 3:
        edges = [r["edge"] for r in positive]
        print("─" * 72)
        print(f"  EDGE SUMMARY (positive-edge matches):")
        print(f"    Count:  {len(positive)}")
        print(f"    Mean:   {np.mean(edges):+.4f}")
        print(f"    Max:    {np.max(edges):+.4f}")
        print(f"    Min:    {np.min(edges):+.4f}")
        strong = sum(1 for e in edges if e >= STRONG_EDGE)
        value  = sum(1 for e in edges if VALUE_EDGE <= e < STRONG_EDGE)
        marg   = sum(1 for e in edges if MARGINAL_EDGE <= e < VALUE_EDGE)
        print(f"    Strong Value (≥5%): {strong}")
        print(f"    Value      (3-5%): {value}")
        print(f"    Marginal   (1-3%): {marg}")

    print("═" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Demo fixtures (the 3 requested + extras)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_FIXTURES = [
    # The 3 requested matches
    {"home": "Everton",   "away": "Man United", "b365h": 3.70, "b365d": 3.75, "b365a": 1.91,
     "league": "E0",      "date": "2026-02-22"},
    {"home": "Alaves",    "away": "Girona",      "b365h": 2.30, "b365d": 3.00, "b365a": 3.50,
     "league": "SP1",     "date": "2026-02-22"},
    {"home": "LA Galaxy", "away": "NYCFC",       "b365h": 2.25, "b365d": 3.60, "b365a": 2.80,
     "league": "USA_MLS", "date": "2026-02-22"},
]


def load_fixtures_from_csv(csv_path: str) -> List[Dict]:
    """
    Load fixtures from a CSV file with columns:
    home, away, b365h, b365d, b365a, [league], [date], [referee]
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"home", "away", "b365h", "b365d", "b365a"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Filter to upcoming matches if 'date' column present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= TODAY].copy()
        if len(df) == 0:
            print("  No upcoming fixtures found in CSV (all dates are in the past).")
            return []

    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V3 Scanner — find HT draw value bets by edge"
    )
    parser.add_argument("--demo",      action="store_true",
                        help="Run demo with 3 hardcoded example matches")
    parser.add_argument("--fixtures",  type=str, default=None,
                        help="CSV file with upcoming fixtures")
    parser.add_argument("--match",     nargs="+", action="append", default=[],
                        help="Manual match: home away b365h b365d b365a [league]")
    parser.add_argument("--min-edge",  type=float, default=0.0,
                        help="Minimum edge to display (default 0.0)")
    parser.add_argument("--show-all",  action="store_true",
                        help="Show all matches regardless of edge")
    parser.add_argument("--paths",     default="models/v3/v3_paths.json")
    args = parser.parse_args()

    fixtures = []

    if args.demo:
        print("  Running demo with 3 example matches...")
        fixtures = DEMO_FIXTURES

    if args.fixtures:
        fixtures = load_fixtures_from_csv(args.fixtures)

    for m in args.match:
        if len(m) < 5:
            print(f"  Skipping malformed --match: {m}")
            continue
        fixtures.append({
            "home":   m[0], "away": m[1],
            "b365h":  float(m[2]), "b365d": float(m[3]), "b365a": float(m[4]),
            "league": m[5] if len(m) > 5 else "",
        })

    if not fixtures:
        print("No fixtures specified. Use --demo, --fixtures FILE, or --match args.")
        print("  Example: python src/scan_v3.py --demo")
        parser.print_help()
        return

    results = run_scan(
        fixtures   = fixtures,
        paths_file = args.paths,
        min_edge   = args.min_edge,
        show_all   = args.show_all,
    )
    print_scan_results(results, min_edge=args.min_edge)


if __name__ == "__main__":
    main()
