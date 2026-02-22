"""
Daily Picks — Contextual HT Draw Betting Report
================================================
Generates a human-readable picks report with team context:
  - Recent form (last 5 results with HT scores)
  - Head-to-head HT draw history
  - League HT draw tendencies
  - Edge explanation in plain English

Designed to be read by a human (or sent via Discord/email).

Usage:
    python src/picks.py                  # today's picks, pretty format
    python src/picks.py --json           # JSON for automation
    python src/picks.py --min-edge 0.03  # only VALUE+ picks
    python src/picks.py --top 3          # top N picks only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

MEGA_PATH = Path("data/processed/mega_dataset_v2.parquet")


def _form_string(team: str, df: pd.DataFrame, n: int = 5) -> str:
    """Generate compact form string like 'WDLWW' from recent matches."""
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    form = []
    for _, r in matches.iterrows():
        if pd.isna(r.get("FTHG")):
            continue
        if r["HomeTeam"] == team:
            gf, ga = r["FTHG"], r["FTAG"]
        else:
            gf, ga = r["FTAG"], r["FTHG"]
        if gf > ga:
            form.append("W")
        elif gf == ga:
            form.append("D")
        else:
            form.append("L")
    return "".join(form) if form else "?"


def _recent_results(team: str, df: pd.DataFrame, n: int = 5) -> List[Dict]:
    """Get last N match results with HT scores."""
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    results = []
    for _, r in matches.iterrows():
        is_home = r["HomeTeam"] == team
        opp = r["AwayTeam"] if is_home else r["HomeTeam"]
        loc = "H" if is_home else "A"

        ft = "?"
        ht = ""
        if pd.notna(r.get("FTHG")):
            if is_home:
                ft = f"{int(r['FTHG'])}-{int(r['FTAG'])}"
            else:
                ft = f"{int(r['FTAG'])}-{int(r['FTHG'])}"
        if pd.notna(r.get("HTHG")):
            ht_draw = r["HTHG"] == r["HTAG"]
            if ht_draw:
                ht = "HT draw"
            elif is_home:
                ht = f"HT {int(r['HTHG'])}-{int(r['HTAG'])}"
            else:
                ht = f"HT {int(r['HTAG'])}-{int(r['HTHG'])}"

        results.append({
            "date": str(r["Date"].date()) if pd.notna(r.get("Date")) else "?",
            "opponent": opp,
            "location": loc,
            "score": ft,
            "ht_note": ht,
        })
    return results


def _h2h_ht_draws(home: str, away: str, df: pd.DataFrame) -> Dict:
    """Head-to-head HT draw stats."""
    h2h = df[
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
        ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    ]
    h2h = h2h[h2h["HTHG"].notna()].copy()
    if len(h2h) == 0:
        return {"meetings": 0, "ht_draws": 0, "ht_draw_rate": None}

    ht_draws = (h2h["HTHG"] == h2h["HTAG"]).sum()
    return {
        "meetings": len(h2h),
        "ht_draws": int(ht_draws),
        "ht_draw_rate": round(ht_draws / len(h2h), 3),
        "last_meeting": str(h2h["Date"].max().date()),
    }


def _league_context(league: str, df: pd.DataFrame) -> Dict:
    """League-level HT draw context."""
    lg = df[df["league"] == league]
    lg_ht = lg[lg["y_ht_draw"].notna()]
    if len(lg_ht) == 0:
        return {}

    name = lg["league_name"].iloc[0] if "league_name" in lg.columns else league
    country = lg["country"].iloc[0] if "country" in lg.columns else "?"
    ht_rate = lg_ht["y_ht_draw"].mean()

    # This season's rate
    this_season = lg_ht[lg_ht["Date"] >= "2025-07-01"]
    season_rate = this_season["y_ht_draw"].mean() if len(this_season) > 20 else None

    return {
        "name": name,
        "country": country,
        "historical_ht_draw_rate": round(ht_rate, 3),
        "season_ht_draw_rate": round(season_rate, 3) if season_rate else None,
        "total_matches": len(lg_ht),
    }


def _edge_explanation(result: Dict) -> str:
    """Plain English explanation of the edge."""
    edge = result["inverted_edge"]
    ma = result["model_a_prob"]
    mb = result["model_b_prob"]

    if edge >= 0.05:
        strength = "Strong divergence"
    elif edge >= 0.03:
        strength = "Clear divergence"
    elif edge >= 0.01:
        strength = "Slight divergence"
    else:
        return "No meaningful edge. Market and fundamentals roughly agree."

    return (f"{strength} between market and fundamentals. "
            f"Market prices HT draw at {ma:.0%}, but team stats say {mb:.0%}. "
            f"Historically, when fundamentals underrate the draw by this much, "
            f"the actual draw rate beats the market.")


def _team_tendency(team: str, df: pd.DataFrame, as_home: bool) -> str:
    """Describe team's recent HT draw tendency."""
    if as_home:
        matches = df[df["HomeTeam"] == team].tail(10)
    else:
        matches = df[df["AwayTeam"] == team].tail(10)

    matches = matches[matches["HTHG"].notna()]
    if len(matches) < 3:
        return ""

    ht_draws = (matches["HTHG"] == matches["HTAG"]).sum()
    rate = ht_draws / len(matches)
    loc = "home" if as_home else "away"

    if rate >= 0.5:
        return f"{team} drawn at HT in {ht_draws}/{len(matches)} recent {loc} matches ({rate:.0%}). High draw tendency."
    elif rate >= 0.35:
        return f"{team}: {ht_draws}/{len(matches)} HT draws in recent {loc} matches ({rate:.0%})."
    elif rate <= 0.15:
        return f"{team} rarely level at HT in {loc} matches ({ht_draws}/{len(matches)}). Goals tend to come early."
    return ""


def build_picks_report(min_edge: float = 0.01,
                        top_n: Optional[int] = None) -> List[Dict]:
    """Build full contextual picks report."""
    from src.scan_v4 import fetch_fdco_fixtures, run_scan

    fixtures = fetch_fdco_fixtures()
    if not fixtures:
        print("No fixtures available.", file=sys.stderr)
        return []

    results = run_scan(fixtures)

    # Filter
    picks = [r for r in results if r["inverted_edge"] >= min_edge]
    if top_n:
        picks = picks[:top_n]

    if not picks:
        return []

    # Load dataset for context
    if not MEGA_PATH.exists():
        return picks

    df = pd.read_parquet(MEGA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    from src.utils import resolve_team_name
    all_teams = list(set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist()))

    # Enrich each pick with context
    for pick in picks:
        home = resolve_team_name(pick["home_team"], all_teams) or pick["home_team"]
        away = resolve_team_name(pick["away_team"], all_teams) or pick["away_team"]

        pick["context"] = {
            "home_form": _form_string(home, df),
            "away_form": _form_string(away, df),
            "home_recent": _recent_results(home, df, n=5),
            "away_recent": _recent_results(away, df, n=5),
            "h2h": _h2h_ht_draws(home, away, df),
            "league": _league_context(pick.get("league", ""), df),
            "home_tendency": _team_tendency(home, df, as_home=True),
            "away_tendency": _team_tendency(away, df, as_home=False),
            "edge_explanation": _edge_explanation(pick),
        }

    return picks


def print_picks_report(picks: List[Dict]) -> None:
    """Print human-readable picks report."""
    if not picks:
        print("No picks today.")
        return

    today = datetime.now().strftime("%A, %B %d")
    print(f"\n{'═'*60}")
    print(f"  🎯 HT DRAW PICKS — {today}")
    print(f"{'═'*60}")

    for i, p in enumerate(picks, 1):
        ctx = p.get("context", {})
        league = ctx.get("league", {})
        h2h = ctx.get("h2h", {})

        print(f"\n{'─'*60}")
        print(f"  #{i}  {p['home_team']} vs {p['away_team']}")
        if league:
            print(f"       {league.get('name', p.get('league',''))} ({league.get('country', '')})")
        print(f"       {p['date']}")
        print()

        # Rating + Edge
        print(f"  {p['rating']} {p['rating_icon']}  |  Edge: {p['edge_pct']:+.2f}%  |  Draw odds: {p['b365d']}")
        print()

        # Edge explanation
        if ctx.get("edge_explanation"):
            print(f"  Why: {ctx['edge_explanation']}")
            print()

        # Form
        home_form = ctx.get("home_form", "?")
        away_form = ctx.get("away_form", "?")
        print(f"  Form:  {p['home_team']}: {home_form}  |  {p['away_team']}: {away_form}")

        # Tendencies
        for tend in ["home_tendency", "away_tendency"]:
            if ctx.get(tend):
                print(f"  {ctx[tend]}")

        # H2H
        if h2h.get("meetings", 0) > 0:
            print(f"  H2H: {h2h['ht_draws']}/{h2h['meetings']} HT draws "
                  f"({h2h['ht_draw_rate']:.0%}), last met {h2h.get('last_meeting', '?')}")

        # League context
        if league.get("historical_ht_draw_rate"):
            season_note = ""
            if league.get("season_ht_draw_rate"):
                diff = league["season_ht_draw_rate"] - league["historical_ht_draw_rate"]
                if abs(diff) > 0.02:
                    direction = "up" if diff > 0 else "down"
                    season_note = f" (this season: {league['season_ht_draw_rate']:.0%}, {direction})"
            print(f"  League HT draw rate: {league['historical_ht_draw_rate']:.0%}{season_note}")

        # Model breakdown
        print(f"\n  Model A (market):       {p['model_a_prob']:.1%}")
        print(f"  Model B (fundamentals): {p['model_b_prob']:.1%}")
        print(f"    Dixon-Coles: {p['dc_draw_prob']:.1%}  |  Elo: {p['elo_draw_prob']:.1%}")

        # Kelly
        if p.get("kelly_25pct", 0) > 0:
            print(f"\n  Suggested stake (quarter Kelly): {p['kelly_25pct']:.2%} of bankroll")

        # Recent results
        for side, label in [("home_recent", p["home_team"]), ("away_recent", p["away_team"])]:
            recent = ctx.get(side, [])
            if recent:
                print(f"\n  {label} recent:")
                for m in recent[-3:]:
                    ht = f"  ({m['ht_note']})" if m.get("ht_note") else ""
                    print(f"    {m['date']}  {m['location']} vs {m['opponent']:<18} {m['score']}{ht}")

    print(f"\n{'═'*60}")
    print(f"  {len(picks)} picks  |  Data through: ", end="")
    if MEGA_PATH.exists():
        df = pd.read_parquet(MEGA_PATH, columns=["Date"])
        print(f"{pd.to_datetime(df['Date']).max().date()}")
    else:
        print("?")
    print(f"{'═'*60}")


def main():
    parser = argparse.ArgumentParser(description="Daily HT draw picks with context")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--min-edge", type=float, default=0.01,
                        help="Minimum edge threshold (default 0.01 = 1%%)")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top N picks")
    args = parser.parse_args()

    picks = build_picks_report(min_edge=args.min_edge, top_n=args.top)

    if args.json:
        # Clean context for JSON (remove DataFrame-heavy bits)
        print(json.dumps(picks, indent=2, default=str))
    else:
        print_picks_report(picks)


if __name__ == "__main__":
    main()
