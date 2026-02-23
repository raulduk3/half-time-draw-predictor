"""
V4 Bet Tracker — Half-Time Draw Predictions
============================================
Tracks bets placed based on V4 model predictions.

Storage: data/bets.json (human-readable JSON)

Functions:
    add_bet(...)         — Record a new pending bet
    record_result(...)   — Mark a bet as won or lost
    get_stats()          — Return full performance summary
    list_bets(...)       — Display bet list as formatted table
    backfill(...)        — Add historical bets with results

Data model per bet:
    id            — auto-incrementing integer
    date          — match date (YYYY-MM-DD)
    home          — home team name
    away          — away team name
    league        — league code
    ht_draw_odds  — decimal odds for HT draw (actual sportsbook odds)
    stake         — stake in units (default 1.0)
    model_a_prob  — Model A calibrated probability
    model_b_prob  — Model B calibrated probability
    inverted_edge — Model A − Model B (the signal)
    rating        — STRONG VALUE / VALUE / MARGINAL / PASS
    outcome       — "pending" | "win" | "loss"
    pnl           — profit/loss in units (null until resolved)
    notes         — optional string

Usage:
    python src/tracker.py list
    python src/tracker.py add --home "Everton" --away "Man United" \\
        --league E0 --odds 3.75 --stake 0.5 \\
        --model-a 0.398 --model-b 0.385 --edge 0.0139 --rating MARGINAL
    python src/tracker.py result --id 1 --outcome win
    python src/tracker.py stats
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

BETS_FILE = Path("data/bets.json")


# ─────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_bets() -> List[Dict]:
    if not BETS_FILE.exists():
        return []
    with open(BETS_FILE) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save_bets(bets: List[Dict]) -> None:
    BETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BETS_FILE, "w") as f:
        json.dump(bets, f, indent=2)


def _next_id(bets: List[Dict]) -> int:
    if not bets:
        return 1
    return max(b["id"] for b in bets) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def add_bet(
    home:          str,
    away:          str,
    ht_draw_odds:  float,
    stake:         float        = 1.0,
    model_a_prob:  float        = 0.0,
    model_b_prob:  float        = 0.0,
    inverted_edge: float        = 0.0,
    rating:        str          = "",
    league:        str          = "",
    match_date:    Optional[str] = None,
    notes:         str          = "",
) -> Dict:
    """
    Record a new pending bet.

    Returns the new bet dict (also appended to bets.json).
    """
    bets = _load_bets()
    bet = {
        "id":            _next_id(bets),
        "date":          match_date or str(date.today()),
        "home":          home,
        "away":          away,
        "league":        league,
        "ht_draw_odds":  round(float(ht_draw_odds), 3),
        "stake":         round(float(stake), 4),
        "model_a_prob":  round(float(model_a_prob), 4),
        "model_b_prob":  round(float(model_b_prob), 4),
        "inverted_edge": round(float(inverted_edge), 4),
        "edge_pct":      round(float(inverted_edge) * 100, 2),
        "rating":        rating,
        "outcome":       "pending",
        "pnl":           None,
        "added_at":      datetime.now().isoformat(timespec="seconds"),
        "notes":         notes,
    }
    bets.append(bet)
    _save_bets(bets)
    print(f"  Added bet #{bet['id']}: {home} vs {away} @ {ht_draw_odds} "
          f"(edge {inverted_edge:+.2%}, {rating})")
    return bet


def record_result(bet_id: int, outcome: str, notes: str = "") -> Dict:
    """
    Mark a bet as 'win' or 'loss'. Calculates PnL automatically.

    outcome: 'win' or 'loss'
    Returns the updated bet.
    """
    outcome = outcome.lower().strip()
    if outcome not in ("win", "loss"):
        raise ValueError(f"outcome must be 'win' or 'loss', got: {outcome!r}")

    bets = _load_bets()
    for bet in bets:
        if bet["id"] == bet_id:
            if bet["outcome"] != "pending":
                print(f"  Warning: bet #{bet_id} already resolved as '{bet['outcome']}'")
            bet["outcome"] = outcome
            if outcome == "win":
                bet["pnl"] = round(float(bet["stake"]) * (float(bet["ht_draw_odds"]) - 1), 4)
            else:
                bet["pnl"] = -round(float(bet["stake"]), 4)
            if notes:
                bet["notes"] = (bet.get("notes", "") + " | " + notes).strip(" |")
            bet["resolved_at"] = datetime.now().isoformat(timespec="seconds")
            _save_bets(bets)
            print(f"  Bet #{bet_id} ({bet['home']} vs {bet['away']}): "
                  f"{outcome.upper()}  PnL = {bet['pnl']:+.4f} units")
            return bet

    raise ValueError(f"Bet #{bet_id} not found")


def get_stats() -> Dict:
    """
    Return comprehensive performance statistics.

    Returns dict with totals, rates, ROI, and streak info.
    """
    bets = _load_bets()
    if not bets:
        return {"message": "No bets recorded yet."}

    total      = len(bets)
    pending    = [b for b in bets if b["outcome"] == "pending"]
    resolved   = [b for b in bets if b["outcome"] in ("win", "loss")]
    wins       = [b for b in resolved if b["outcome"] == "win"]
    losses     = [b for b in resolved if b["outcome"] == "loss"]

    n_pending  = len(pending)
    n_resolved = len(resolved)
    n_wins     = len(wins)
    n_losses   = len(losses)

    if n_resolved == 0:
        return {
            "total_bets": total,
            "pending":    n_pending,
            "resolved":   0,
            "message":    "No resolved bets yet."
        }

    total_staked = sum(b["stake"] for b in resolved)
    total_pnl    = sum(b["pnl"] for b in resolved if b["pnl"] is not None)
    roi          = total_pnl / total_staked if total_staked > 0 else 0.0
    hit_rate     = n_wins / n_resolved if n_resolved > 0 else 0.0

    avg_odds     = sum(b["ht_draw_odds"] for b in resolved) / n_resolved
    avg_edge     = sum(b["inverted_edge"] for b in resolved) / n_resolved

    # Current streak
    streak       = 0
    streak_type  = ""
    for b in reversed(resolved):
        if not streak_type:
            streak_type = b["outcome"]
        if b["outcome"] == streak_type:
            streak += 1
        else:
            break
    streak_str   = f"{streak} {streak_type}"

    # By rating
    by_rating    = {}
    for rating in ["STRONG VALUE", "VALUE", "MARGINAL"]:
        rbets = [b for b in resolved if b["rating"] == rating]
        if not rbets:
            continue
        r_staked = sum(b["stake"] for b in rbets)
        r_pnl    = sum(b["pnl"] for b in rbets if b["pnl"] is not None)
        by_rating[rating] = {
            "n":         len(rbets),
            "wins":      sum(1 for b in rbets if b["outcome"] == "win"),
            "hit_rate":  sum(1 for b in rbets if b["outcome"] == "win") / len(rbets),
            "pnl":       round(r_pnl, 4),
            "roi":       round(r_pnl / r_staked, 4) if r_staked > 0 else 0.0,
        }

    return {
        "total_bets":    total,
        "pending":       n_pending,
        "resolved":      n_resolved,
        "wins":          n_wins,
        "losses":        n_losses,
        "hit_rate":      round(hit_rate, 4),
        "total_staked":  round(total_staked, 4),
        "total_pnl":     round(total_pnl, 4),
        "roi":           round(roi, 4),
        "avg_odds":      round(avg_odds, 4),
        "avg_edge":      round(avg_edge, 4),
        "current_streak": streak_str,
        "by_rating":     by_rating,
    }


def list_bets(
    status: str = "all",
    n_last: Optional[int] = None,
) -> List[Dict]:
    """
    Display bets as a formatted table and return the list.

    status: 'all' | 'pending' | 'resolved'
    n_last: show only the last N bets

    Returns list of matching bet dicts.
    """
    bets = _load_bets()

    if status == "pending":
        bets = [b for b in bets if b["outcome"] == "pending"]
    elif status == "resolved":
        bets = [b for b in bets if b["outcome"] in ("win", "loss")]

    if n_last:
        bets = bets[-n_last:]

    if not bets:
        print(f"  No {status} bets found.")
        return []

    print(f"\n  {'ID':>4}  {'Date':>10}  {'Match':<35} {'Lg':>6}  "
          f"{'Odds':>6}  {'Edge':>6}  {'Rating':>12}  {'Outcome':>8}  {'PnL':>8}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*35} {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*8}")

    for b in bets:
        match_str = f"{b['home']} vs {b['away']}"
        if len(match_str) > 35:
            match_str = match_str[:32] + "..."
        pnl_str  = f"{b['pnl']:+.4f}" if b["pnl"] is not None else "—"
        edge_str = f"{b['edge_pct']:+.2f}%"

        if b["outcome"] == "win":
            outcome_str = "WIN"
        elif b["outcome"] == "loss":
            outcome_str = "LOSS"
        else:
            outcome_str = "pending"

        print(f"  {b['id']:>4}  {b['date']:>10}  {match_str:<35} {b['league']:>6}  "
              f"{b['ht_draw_odds']:>6.2f}  {edge_str:>6}  {b['rating']:>12}  "
              f"{outcome_str:>8}  {pnl_str:>8}")

    print()

    # Summary for resolved
    resolved = [b for b in bets if b["outcome"] in ("win", "loss")]
    if resolved:
        total_pnl   = sum(b["pnl"] for b in resolved if b["pnl"] is not None)
        total_stake = sum(b["stake"] for b in resolved)
        roi         = total_pnl / total_stake if total_stake > 0 else 0.0
        n_wins      = sum(1 for b in resolved if b["outcome"] == "win")
        print(f"  Resolved: {len(resolved)} bets  |  "
              f"Wins: {n_wins}/{len(resolved)} ({n_wins/len(resolved):.1%})  |  "
              f"PnL: {total_pnl:+.4f}  |  ROI: {roi:+.1%}")

    return bets


def print_stats() -> None:
    """Print formatted performance statistics."""
    stats = get_stats()

    if "message" in stats and stats.get("resolved", 0) == 0:
        print(f"\n  {stats.get('message', 'No bets yet.')}")
        if stats.get("pending"):
            print(f"  {stats['pending']} pending bet(s).")
        return

    print(f"\n{'═'*50}")
    print(f"  V4 BET TRACKER — PERFORMANCE SUMMARY")
    print(f"{'═'*50}")
    print(f"  Total bets:      {stats['total_bets']:>5}  ({stats['pending']} pending)")
    print(f"  Resolved:        {stats['resolved']:>5}  ({stats['wins']}W / {stats['losses']}L)")
    print(f"  Hit rate:        {stats['hit_rate']:>6.1%}")
    print(f"  Total staked:    {stats['total_staked']:>7.2f} units")
    print(f"  Total PnL:       {stats['total_pnl']:>+7.4f} units")
    print(f"  ROI:             {stats['roi']:>+7.1%}")
    print(f"  Avg draw odds:   {stats['avg_odds']:>6.3f}")
    print(f"  Avg edge:        {stats['avg_edge']:>+6.2%}")
    print(f"  Current streak:  {stats['current_streak']}")

    if stats.get("by_rating"):
        print(f"\n  {'Rating':<14}  {'N':>4}  {'Wins':>5}  {'Hit%':>6}  {'PnL':>8}  {'ROI':>7}")
        print(f"  {'-'*14}  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*7}")
        for rating, row in stats["by_rating"].items():
            print(f"  {rating:<14}  {row['n']:>4}  {row['wins']:>5}  "
                  f"{row['hit_rate']:>6.1%}  {row['pnl']:>+8.4f}  {row['roi']:>+7.1%}")

    print(f"{'═'*50}")


def backfill(bets_data: List[Dict]) -> None:
    """
    Backfill historical bets. Each dict should follow the bet schema.
    Existing bets with the same id are skipped.
    """
    existing = _load_bets()
    existing_ids = {b["id"] for b in existing}

    added = 0
    for b in bets_data:
        if b.get("id") in existing_ids:
            continue
        if "id" not in b:
            b["id"] = _next_id(existing + [b2 for b2 in bets_data if "id" in b2])
        if "added_at" not in b:
            b["added_at"] = datetime.now().isoformat(timespec="seconds")
        existing.append(b)
        added += 1

    _save_bets(existing)
    print(f"  Backfilled {added} bet(s).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V4 Bet Tracker — track HT draw bets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    list_p = sub.add_parser("list", help="List bets")
    list_p.add_argument("--status", choices=["all", "pending", "resolved"], default="all")
    list_p.add_argument("--last", type=int, default=None, help="Show last N bets")

    # add
    add_p = sub.add_parser("add", help="Add a new bet")
    add_p.add_argument("--home",     required=True)
    add_p.add_argument("--away",     required=True)
    add_p.add_argument("--league",   default="")
    add_p.add_argument("--odds",     type=float, required=True, help="HT draw decimal odds")
    add_p.add_argument("--stake",    type=float, default=1.0,   help="Stake in units")
    add_p.add_argument("--model-a",  type=float, default=0.0,   help="Model A probability")
    add_p.add_argument("--model-b",  type=float, default=0.0,   help="Model B probability")
    add_p.add_argument("--edge",     type=float, default=0.0,   help="Inverted edge (decimal)")
    add_p.add_argument("--rating",   default="",                help="STRONG VALUE / VALUE / MARGINAL")
    add_p.add_argument("--date",     default=None,              help="Match date YYYY-MM-DD")
    add_p.add_argument("--notes",    default="")

    # result
    res_p = sub.add_parser("result", help="Record bet outcome")
    res_p.add_argument("--id",      type=int, required=True)
    res_p.add_argument("--outcome", choices=["win", "loss"], required=True)
    res_p.add_argument("--notes",   default="")

    # stats
    sub.add_parser("stats", help="Show performance stats")

    # backfill-galaxy (one-time backfill for Galaxy/NYCFC bet)
    sub.add_parser("backfill-galaxy", help="Backfill the LA Galaxy vs NYCFC bet")

    args = parser.parse_args()

    if args.cmd == "list":
        list_bets(status=args.status, n_last=args.last)

    elif args.cmd == "add":
        add_bet(
            home          = args.home,
            away          = args.away,
            ht_draw_odds  = args.odds,
            stake         = args.stake,
            model_a_prob  = args.model_a,
            model_b_prob  = args.model_b,
            inverted_edge = args.edge,
            rating        = args.rating,
            league        = args.league,
            match_date    = args.date,
            notes         = args.notes,
        )

    elif args.cmd == "result":
        record_result(bet_id=args.id, outcome=args.outcome, notes=args.notes)

    elif args.cmd == "stats":
        print_stats()

    elif args.cmd == "backfill-galaxy":
        # Backfill the LA Galaxy vs NYCFC bet placed 2026-02-22
        # +125 American odds = 2.25 decimal | edge +1.61% MARGINAL
        backfill([{
            "id":            1,
            "date":          "2026-02-22",
            "home":          "LA Galaxy",
            "away":          "New York City",
            "league":        "USA_MLS",
            "ht_draw_odds":  2.25,
            "stake":         1.0,
            "model_a_prob":  0.4010,
            "model_b_prob":  0.3850,
            "inverted_edge": 0.0161,
            "edge_pct":      1.61,
            "rating":        "MARGINAL",
            "outcome":       "pending",
            "pnl":           None,
            "notes":         "Backfilled: placed at +125 American (2.25 decimal) HT draw odds",
            "added_at":      "2026-02-22T00:00:00",
        }])
        print("  Backfilled Galaxy/NYCFC bet. Use 'result --id 1 --outcome win/loss' when resolved.")


if __name__ == "__main__":
    main()
