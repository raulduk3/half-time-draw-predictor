"""
V4 Scanner — Upcoming Fixture Scanner Ranked by Inverted Edge
=============================================================
Fetches upcoming fixtures, runs the V4 two-model predictor on each,
and ranks by inverted edge (Model A − Model B).

SIGNAL: Positive inverted edge = market prices HT draw HIGHER than fundamentals
        → actual HT draws historically exceed market prediction → value bet

Sources (in priority order):
  1. football-data.co.uk fixtures CSV (public: https://www.football-data.co.uk/fixtures.csv)
  2. Manual --match arguments
  3. --demo mode (hardcoded example fixtures)

Filters:
  - Future matches only (date > today)
  - Matches with odds available
  - Only shows positive-edge matches (default min_edge=0.0)

Date parsing: football-data.co.uk uses dd/mm/yyyy format.

Usage:
    python src/scan_v4.py                    # fetch live from football-data.co.uk
    python src/scan_v4.py --demo             # run example matches
    python src/scan_v4.py --fixtures file.csv
    python src/scan_v4.py --match "Everton" "Man United" 3.70 3.75 1.91 E0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

TODAY       = pd.Timestamp.today().normalize()
TOMORROW    = TODAY + pd.Timedelta(days=1)

FDCO_URL    = "https://www.football-data.co.uk/fixtures.csv"

STRONG_EDGE  = 0.05
VALUE_EDGE   = 0.03
MARGINAL_EDGE = 0.01

# ── League code mapping (football-data.co.uk → our internal codes) ─────────
FDCO_LEAGUE_MAP = {
    "E0":  "E0",    # Premier League
    "E1":  "E1",    # Championship
    "E2":  "E2",    # League One
    "E3":  "E3",    # League Two
    "SP1": "SP1",   # La Liga
    "SP2": "SP2",
    "D1":  "D1",    # Bundesliga
    "D2":  "D2",
    "I1":  "I1",    # Serie A
    "I2":  "I2",
    "F1":  "F1",    # Ligue 1
    "F2":  "F2",
    "N1":  "N1",    # Eredivisie
    "B1":  "B1",    # Belgian Pro League
    "P1":  "P1",    # Primeira Liga
    "G1":  "G1",    # Super League Greece
    "T1":  "T1",    # Süper Lig
    "SC0": "SC0",   # SPL
    "SC1": "SC1",
}


def fetch_fdco_fixtures(url: str = FDCO_URL) -> List[Dict]:
    """
    Fetch upcoming fixtures from football-data.co.uk.
    Returns list of fixture dicts with home/away/odds/league/date.
    Filters to future matches only.
    """
    try:
        import urllib.request
        import io
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode("latin1")
        df = pd.read_csv(io.StringIO(raw), on_bad_lines="skip")
    except Exception as e:
        print(f"  Warning: could not fetch fixtures from football-data.co.uk: {e}")
        return []

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Parse date — football-data.co.uk uses dd/mm/yyyy
    if "Date" not in df.columns:
        print("  Warning: no Date column in fixtures CSV")
        return []

    parsed = []
    for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
        try:
            df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors="raise")
            parsed = True
            break
        except Exception:
            continue
    if not parsed:
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")

    # Filter to future matches
    df = df[df["Date"].notna()].copy()
    df = df[df["Date"] >= TOMORROW].copy()

    if len(df) == 0:
        print("  No upcoming fixtures found after date filtering.")
        return []

    # Detect odds columns (B365H/D/A or PSH/D/A)
    fixtures = []
    for _, row in df.iterrows():
        # Try B365 first, fall back to PS (Pinnacle)
        b365h = _safe_float(row.get("B365H") or row.get("PSH"))
        b365d = _safe_float(row.get("B365D") or row.get("PSD"))
        b365a = _safe_float(row.get("B365A") or row.get("PSA"))

        if not (b365h and b365d and b365a and b365h > 1.0):
            continue

        league = str(row.get("Div", "")).strip()
        league = FDCO_LEAGUE_MAP.get(league, league)

        fixtures.append({
            "home":   str(row.get("HomeTeam", row.get("Home", ""))).strip(),
            "away":   str(row.get("AwayTeam", row.get("Away", ""))).strip(),
            "b365h":  b365h,
            "b365d":  b365d,
            "b365a":  b365a,
            "league": league,
            "date":   str(row["Date"].date()),
        })

    print(f"  Fetched {len(fixtures)} upcoming fixtures with odds from football-data.co.uk")
    return fixtures


def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def load_fixtures_from_csv(csv_path: str) -> List[Dict]:
    """Load fixtures from a local CSV file."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"home", "away", "b365h", "b365d", "b365a"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "date" in df.columns:
        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
            try:
                df["date"] = pd.to_datetime(df["date"], format=fmt, errors="raise")
                break
            except Exception:
                continue
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna() & (df["date"] >= TOMORROW)].copy()
        if len(df) == 0:
            print("  No upcoming fixtures found (all dates in the past or unparseable).")
            return []

    return df.to_dict(orient="records")


def run_scan(
    fixtures:   List[Dict],
    paths_file: str   = "models/v4/v4_paths.json",
    min_edge:   float = 0.0,
) -> List[Dict]:
    """Run V4 predictor on all fixtures, return sorted results."""
    from src.predict_v4 import V4Predictor

    print("Loading V4 predictor...")
    predictor = V4Predictor.load(paths_file=paths_file)

    results = []
    for fix in fixtures:
        try:
            r = predictor.predict(
                home_team = str(fix.get("home", "")),
                away_team = str(fix.get("away", "")),
                b365h     = float(fix["b365h"]),
                b365d     = float(fix["b365d"]),
                b365a     = float(fix["b365a"]),
                league    = str(fix.get("league", "")),
                referee   = fix.get("referee"),
            )
            r["date"] = fix.get("date", str(TODAY.date()))
            results.append(r)
        except Exception as e:
            print(f"  Warning: failed to predict {fix.get('home')} vs {fix.get('away')}: {e}")

    # Sort by inverted edge descending
    results.sort(key=lambda x: x["inverted_edge"], reverse=True)
    return results


def print_scan_results(results: List[Dict], min_edge: float = 0.0) -> None:
    """Pretty-print scanner results ranked by inverted edge."""
    positive = [r for r in results if r["inverted_edge"] >= min_edge]
    negative = [r for r in results if r["inverted_edge"] < min_edge]

    print("\n" + "═" * 76)
    print("  V4 SCANNER — HALF-TIME DRAW VALUE FINDER")
    print("  Ranked by INVERTED EDGE (Model A market − Model B fundamentals)")
    print("  Positive edge = market rates draw higher than fundamentals → value bet")
    print("═" * 76)
    print(f"  {len(results)} fixtures scanned  |  "
          f"{len(positive)} with edge ≥ {min_edge:.0%}  |  "
          f"Today: {TODAY.date()}")
    print()

    if not positive:
        print("  No matches with positive edge found.")
        if negative:
            print(f"  ({len(negative)} matches scanned, all with negative edge — market underpricing draws)")
        return

    # Table header
    print(f"  {'#':>2}  {'Date':>10}  {'Match':<35} {'Lg':>6}  {'D-odds':>7}  "
          f"{'MktA':>6} {'FndB':>6} {'Edge':>7}  {'Rating':<14}")
    print(f"  {'-'*2}  {'-'*10}  {'-'*35} {'-'*6}  {'-'*7}  "
          f"{'-'*6} {'-'*6} {'-'*7}  {'-'*14}")

    for i, r in enumerate(positive, 1):
        match_str = f"{r['home_team']} vs {r['away_team']}"
        if len(match_str) > 35:
            match_str = match_str[:32] + "..."
        edge_str = f"{r['edge_pct']:+.2f}%"
        edge_val = r["inverted_edge"]

        if edge_val >= STRONG_EDGE:     rating_display = f"★★★ {r['rating']}"
        elif edge_val >= VALUE_EDGE:    rating_display = f"★★  {r['rating']}"
        elif edge_val >= MARGINAL_EDGE: rating_display = f"★   {r['rating']}"
        else:                           rating_display = f"—   {r['rating']}"

        print(f"  {i:>2}  {r.get('date',''):>10}  {match_str:<35} {r['league']:>6}  "
              f"{r['b365d']:>7.2f}  "
              f"{r['model_a_prob']:>6.1%} {r['model_b_prob']:>6.1%} {edge_str:>7}  "
              f"{rating_display:<14}")

    print()

    # Detailed view for top matches (up to 5)
    top_n = min(len(positive), 5)
    if top_n > 0 and positive[0]["inverted_edge"] >= MARGINAL_EDGE:
        print("─" * 76)
        print(f"  TOP {top_n} MATCH DETAILS:")
        from src.predict_v4 import V4Predictor
        for r in positive[:top_n]:
            V4Predictor.print_result(r)

    # Summary stats
    if len(positive) >= 3:
        edges = [r["inverted_edge"] for r in positive]
        print("─" * 76)
        print(f"  EDGE SUMMARY:")
        print(f"    Count:  {len(positive)}")
        print(f"    Mean:   {np.mean(edges):+.4f}")
        print(f"    Max:    {np.max(edges):+.4f}")
        strong = sum(1 for e in edges if e >= STRONG_EDGE)
        value  = sum(1 for e in edges if VALUE_EDGE <= e < STRONG_EDGE)
        marg   = sum(1 for e in edges if MARGINAL_EDGE <= e < VALUE_EDGE)
        print(f"    Strong Value (≥5%): {strong}")
        print(f"    Value      (3-5%): {value}")
        print(f"    Marginal   (1-3%): {marg}")

    print("═" * 76)


# ── Demo fixtures ─────────────────────────────────────────────────────────────

DEMO_FIXTURES = [
    {"home": "Everton",   "away": "Man United", "b365h": 3.70, "b365d": 3.75, "b365a": 1.91,
     "league": "E0",      "date": "2026-02-22"},
    {"home": "Alaves",    "away": "Girona",      "b365h": 2.30, "b365d": 3.00, "b365a": 3.50,
     "league": "SP1",     "date": "2026-02-22"},
    {"home": "LA Galaxy", "away": "NYCFC",       "b365h": 2.25, "b365d": 3.60, "b365a": 2.80,
     "league": "USA_MLS", "date": "2026-02-22"},
    {"home": "Leeds United", "away": "Sheffield Utd", "b365h": 2.10, "b365d": 3.40, "b365a": 3.50,
     "league": "E1",         "date": "2026-02-22"},
    {"home": "Bayern Munich", "away": "Dortmund",     "b365h": 1.60, "b365d": 4.00, "b365a": 5.50,
     "league": "D1",          "date": "2026-02-22"},
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V4 Scanner — find HT draw value bets by inverted edge"
    )
    parser.add_argument("--demo",     action="store_true",
                        help="Run demo with example matches (no network needed)")
    parser.add_argument("--fetch",    action="store_true", default=True,
                        help="Fetch live fixtures from football-data.co.uk (default)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Disable live fixture fetching")
    parser.add_argument("--fixtures", type=str, default=None,
                        help="CSV file with upcoming fixtures")
    parser.add_argument("--match",    nargs="+", action="append", default=[],
                        help="Manual match: home away b365h b365d b365a [league] [date]")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Minimum inverted edge to display (default 0.0)")
    parser.add_argument("--show-pass", action="store_true",
                        help="Show matches with negative edge (PASS) too")
    parser.add_argument("--paths",    default="models/v4/v4_paths.json")
    args = parser.parse_args()

    fixtures = []

    if args.demo:
        print("  Running demo with example matches...")
        fixtures = DEMO_FIXTURES

    elif not args.no_fetch and not args.fixtures and not args.match:
        # Default: fetch live
        fixtures = fetch_fdco_fixtures()
        if not fixtures:
            print("  Falling back to demo mode (no live fixtures available).")
            fixtures = DEMO_FIXTURES

    if args.fixtures:
        csv_fixtures = load_fixtures_from_csv(args.fixtures)
        fixtures.extend(csv_fixtures)

    if not args.demo and args.no_fetch and args.fixtures is None:
        fixtures = DEMO_FIXTURES  # fallback

    for m in args.match:
        if len(m) < 5:
            print(f"  Skipping malformed --match: {m}")
            continue
        fixtures.append({
            "home":   m[0], "away": m[1],
            "b365h":  float(m[2]), "b365d": float(m[3]), "b365a": float(m[4]),
            "league": m[5] if len(m) > 5 else "",
            "date":   m[6] if len(m) > 6 else str(TODAY.date()),
        })

    if not fixtures:
        print("No fixtures to scan. Use --demo or --match args.")
        return

    min_edge = 0.0 if args.show_pass else args.min_edge
    results  = run_scan(fixtures=fixtures, paths_file=args.paths, min_edge=min_edge)
    print_scan_results(results, min_edge=min_edge)


if __name__ == "__main__":
    main()
