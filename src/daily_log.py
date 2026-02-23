"""
Daily Prediction Logger — Prospective Validation
=================================================
Runs the scanner on all upcoming fixtures, logs every prediction
with timestamp BEFORE matches kick off, then later scores them
against actual HT results.

This creates the paper trail needed for prospective validation:
pre-registered predictions that can't be cherry-picked.

Data stored in: data/predictions/YYYY-MM-DD.json

Usage:
    python src/daily_log.py predict     # Log today's predictions
    python src/daily_log.py score       # Score past predictions against results
    python src/daily_log.py summary     # Show cumulative validation stats
    python src/daily_log.py --json summary  # JSON output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, ".")

PREDICTIONS_DIR = Path("data/predictions")
MEGA_PATH = Path("data/processed/mega_dataset_v2.parquet")


def log_predictions() -> List[Dict]:
    """
    Run scanner on all available fixtures and log predictions.
    Timestamps everything for prospective validation.
    """
    from src.scan_v4 import fetch_fdco_fixtures, run_scan

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = fetch_fdco_fixtures()
    if not fixtures:
        print("No fixtures available.", file=sys.stderr)
        return []

    results = run_scan(fixtures)

    # Group by match date
    from collections import defaultdict
    by_date = defaultdict(list)
    for r in results:
        match_date = r.get("date", str(date.today()))
        by_date[match_date].append(r)

    logged = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    for match_date, preds in by_date.items():
        out_file = PREDICTIONS_DIR / f"{match_date}.json"

        # Load existing predictions for this date (don't overwrite)
        existing = []
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)

        # Check for duplicates
        existing_keys = set()
        for e in existing:
            existing_keys.add(f"{e['home_team']}_{e['away_team']}")

        new_preds = []
        for p in preds:
            key = f"{p['home_team']}_{p['away_team']}"
            if key in existing_keys:
                continue
            p["logged_at"] = timestamp
            p["scored"] = False
            p["ht_draw_actual"] = None
            new_preds.append(p)

        if new_preds:
            all_preds = existing + new_preds
            with open(out_file, "w") as f:
                json.dump(all_preds, f, indent=2, default=str)
            print(f"  {match_date}: logged {len(new_preds)} new predictions "
                  f"({len(all_preds)} total)", file=sys.stderr)
            logged.extend(new_preds)
        else:
            print(f"  {match_date}: all predictions already logged", file=sys.stderr)

    print(f"\n  Total: {len(logged)} new predictions logged", file=sys.stderr)
    return logged


def score_predictions() -> Dict:
    """
    Score past predictions against actual HT results.
    Downloads latest results and matches against logged predictions.
    """
    if not PREDICTIONS_DIR.exists():
        print("No predictions directory found.", file=sys.stderr)
        return {}

    # Load current dataset for results
    if not MEGA_PATH.exists():
        print("No mega dataset found. Run update_data.py first.", file=sys.stderr)
        return {}

    df = pd.read_parquet(MEGA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Build results lookup
    results_lookup = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG")):
            key = f"{row['HomeTeam']}_{row['AwayTeam']}_{row['Date'].date()}"
            results_lookup[key] = {
                "hthg": int(row["HTHG"]),
                "htag": int(row["HTAG"]),
                "ht_draw": row["HTHG"] == row["HTAG"],
                "fthg": int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None,
                "ftag": int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None,
            }

    # Also try to fuzzy match team names
    from src.utils import resolve_team_name
    all_teams = list(set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist()))

    scored_count = 0
    total_files = 0

    for pred_file in sorted(PREDICTIONS_DIR.glob("*.json")):
        total_files += 1
        with open(pred_file) as f:
            preds = json.load(f)

        match_date = pred_file.stem
        updated = False

        for p in preds:
            if p.get("scored"):
                continue

            home = p["home_team"]
            away = p["away_team"]

            # Try exact match
            key = f"{home}_{away}_{match_date}"
            result = results_lookup.get(key)

            # Try resolved names
            if not result:
                home_r = resolve_team_name(home, all_teams) or home
                away_r = resolve_team_name(away, all_teams) or away
                key = f"{home_r}_{away_r}_{match_date}"
                result = results_lookup.get(key)

            if result:
                p["scored"] = True
                p["ht_draw_actual"] = result["ht_draw"]
                p["ht_score"] = f"{result['hthg']}-{result['htag']}"
                if result["fthg"] is not None:
                    p["ft_score"] = f"{result['fthg']}-{result['ftag']}"
                p["scored_at"] = datetime.utcnow().isoformat() + "Z"
                scored_count += 1
                updated = True

        if updated:
            with open(pred_file, "w") as f:
                json.dump(preds, f, indent=2, default=str)

    print(f"  Scored {scored_count} predictions across {total_files} files", file=sys.stderr)
    return {"scored": scored_count, "files": total_files}


def get_summary() -> Dict:
    """
    Cumulative validation statistics across all scored predictions.
    """
    if not PREDICTIONS_DIR.exists():
        return {}

    all_preds = []
    for pred_file in sorted(PREDICTIONS_DIR.glob("*.json")):
        with open(pred_file) as f:
            preds = json.load(f)
        all_preds.extend(preds)

    total = len(all_preds)
    scored = [p for p in all_preds if p.get("scored")]
    unscored = [p for p in all_preds if not p.get("scored")]

    if not scored:
        return {
            "total_predictions": total,
            "scored": 0,
            "unscored": len(unscored),
            "tiers": {},
        }

    # By tier
    tiers = {}
    for tier_name, min_e, max_e in [
        ("STRONG_VALUE", 0.05, 1.0),
        ("VALUE", 0.03, 0.05),
        ("MARGINAL", 0.01, 0.03),
        ("PASS", -1.0, 0.01),
    ]:
        tier_preds = [p for p in scored if min_e <= p.get("inverted_edge", 0) < max_e]
        if not tier_preds:
            continue

        hits = sum(1 for p in tier_preds if p.get("ht_draw_actual"))
        n = len(tier_preds)
        # Prefer real HT odds when available, fall back to B365D (FT proxy)
        def _get_odds(p):
            return p.get("ht_draw_odds_real") or p.get("b365d", 3.0)
        avg_odds = np.mean([_get_odds(p) for p in tier_preds])

        # Flat bet ROI (using best available odds)
        returns = sum(_get_odds(p) for p in tier_preds if p.get("ht_draw_actual"))
        roi = (returns - n) / n if n > 0 else 0

        tiers[tier_name] = {
            "n": n,
            "hits": hits,
            "hit_rate": round(hits / n, 4),
            "avg_odds": round(avg_odds, 2),
            "roi": round(roi, 4),
            "pnl_flat_10": round((returns - n) * 10, 2),
        }

    # Overall
    all_hits = sum(1 for p in scored if p.get("ht_draw_actual"))

    return {
        "total_predictions": total,
        "scored": len(scored),
        "unscored": len(unscored),
        "overall_hit_rate": round(all_hits / len(scored), 4) if scored else 0,
        "date_range": {
            "first": min(p.get("date", "") for p in scored),
            "last": max(p.get("date", "") for p in scored),
        },
        "tiers": tiers,
    }


def print_summary(summary: Dict) -> None:
    """Pretty-print validation summary."""
    if not summary.get("scored"):
        print("No scored predictions yet. Run 'daily_log.py score' after updating data.")
        return

    print(f"\n{'═'*60}")
    print(f"  PROSPECTIVE VALIDATION — CUMULATIVE")
    print(f"{'═'*60}")
    print(f"  Predictions: {summary['total_predictions']} total, "
          f"{summary['scored']} scored, {summary['unscored']} pending")
    print(f"  Period: {summary['date_range']['first']} to {summary['date_range']['last']}")
    print(f"  Overall hit rate: {summary['overall_hit_rate']:.1%}")
    print()

    print(f"  {'Tier':<16} {'N':>5} {'Hits':>5} {'Rate':>6} {'Odds':>6} {'ROI':>8} {'PnL@$10':>8}")
    print(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")

    for tier, data in summary.get("tiers", {}).items():
        print(f"  {tier:<16} {data['n']:>5} {data['hits']:>5} "
              f"{data['hit_rate']:>6.1%} {data['avg_odds']:>6.2f} "
              f"{data['roi']:>+8.1%} {data['pnl_flat_10']:>+8.2f}")

    print(f"{'═'*60}")


def main():
    parser = argparse.ArgumentParser(description="Daily prediction logger for prospective validation")
    parser.add_argument("command", choices=["predict", "score", "summary"],
                        help="predict: log today's predictions; score: match against results; summary: show stats")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.command == "predict":
        preds = log_predictions()
        if args.json:
            print(json.dumps(preds, indent=2, default=str))

    elif args.command == "score":
        result = score_predictions()
        if args.json:
            print(json.dumps(result, indent=2))

    elif args.command == "summary":
        summary = get_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print_summary(summary)


if __name__ == "__main__":
    main()
