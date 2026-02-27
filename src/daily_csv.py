"""
Daily CSV Logger — Running record of all predictions and results.
=================================================================
Maintains a single CSV file with every prediction and its outcome.
Designed for spreadsheet analysis and sharing.

Output: data/predictions/running_log.csv

Usage:
    python src/daily_csv.py export      # Export all predictions to CSV
    python src/daily_csv.py append      # Append today's predictions (run after daily_log.py predict)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PREDICTIONS_DIR = Path("data/predictions")
CSV_PATH = PREDICTIONS_DIR / "running_log.csv"

COLUMNS = [
    "date", "home_team", "away_team", "league",
    "model_a_prob", "model_b_prob", "inverted_edge", "edge_pct", "rating",
    "b365h", "b365d", "b365a",
    "ht_draw_actual", "ht_score", "ft_score",
    "scored", "logged_at", "scored_at",
]


def _load_all_predictions() -> List[Dict]:
    """Load all prediction JSON files into a flat list."""
    all_preds = []
    if not PREDICTIONS_DIR.exists():
        return all_preds
    for f in sorted(PREDICTIONS_DIR.glob("*.json")):
        with open(f) as fh:
            preds = json.load(fh)
        for p in preds:
            p.setdefault("date", f.stem)
        all_preds.extend(preds)
    return all_preds


def export_csv() -> Path:
    """Export all predictions to a single CSV."""
    preds = _load_all_predictions()
    if not preds:
        print("No predictions found.")
        return CSV_PATH

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for p in preds:
            row = {
                "date": p.get("date", ""),
                "home_team": p.get("home_team", ""),
                "away_team": p.get("away_team", ""),
                "league": p.get("league", ""),
                "model_a_prob": _fmt(p.get("model_a_prob")),
                "model_b_prob": _fmt(p.get("model_b_prob")),
                "inverted_edge": _fmt(p.get("inverted_edge")),
                "edge_pct": _fmt(p.get("edge_pct")),
                "rating": p.get("rating", ""),
                "b365h": _fmt(p.get("b365h")),
                "b365d": _fmt(p.get("b365d")),
                "b365a": _fmt(p.get("b365a")),
                "ht_draw_actual": _fmt_bool(p.get("ht_draw_actual")),
                "ht_score": p.get("ht_score", ""),
                "ft_score": p.get("ft_score", ""),
                "scored": "yes" if p.get("scored") else "no",
                "logged_at": p.get("logged_at", ""),
                "scored_at": p.get("scored_at", ""),
            }
            writer.writerow(row)

    print(f"Exported {len(preds)} predictions to {CSV_PATH}")
    return CSV_PATH


def _fmt(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _fmt_bool(val) -> str:
    if val is None:
        return ""
    return "1" if val else "0"


def main():
    parser = argparse.ArgumentParser(description="CSV export for prediction log")
    parser.add_argument("command", choices=["export"], help="export: write running_log.csv")
    args = parser.parse_args()

    if args.command == "export":
        export_csv()


if __name__ == "__main__":
    main()
