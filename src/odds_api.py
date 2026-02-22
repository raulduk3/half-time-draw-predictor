"""
The Odds API Integration — Real HT Draw Odds
=============================================
Fetches actual half-time draw odds from The Odds API (the-odds-api.com).

This replaces the FT draw odds proxy with REAL HT draw lines from
DraftKings, FanDuel, Betfair, and other books.

Market: h2h_3_way_h1 (1st Half 3-Way Result: Home/Draw/Away at HT)

API key: set ODDS_API_KEY environment variable
Free tier: 500 requests/month (enough for daily scanning)

Usage:
    # As module
    from src.odds_api import fetch_ht_odds, get_sport_keys
    odds = fetch_ht_odds("soccer_epl")

    # CLI
    python src/odds_api.py sports                    # list available soccer sports
    python src/odds_api.py odds soccer_epl           # EPL HT draw odds
    python src/odds_api.py odds soccer_spain_la_liga  # La Liga HT draw odds
    python src/odds_api.py all                       # all soccer leagues
    python src/odds_api.py --json all                # JSON output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Dict, List, Optional

API_HOST = "https://api.the-odds-api.com"
HT_MARKET = "h2h_3_way_h1"   # 1st Half 3-Way Result

# Map the-odds-api sport keys to our internal league codes
SPORT_TO_LEAGUE = {
    "soccer_epl":             "E0",
    "soccer_efl_champ":       "E1",
    "soccer_england_league1": "E2",
    "soccer_england_league2": "E3",
    "soccer_spain_la_liga":   "SP1",
    "soccer_spain_segunda_division": "SP2",
    "soccer_germany_bundesliga":  "D1",
    "soccer_germany_bundesliga2": "D2",
    "soccer_italy_serie_a":       "I1",
    "soccer_italy_serie_b":       "I2",
    "soccer_france_ligue_one":    "F1",
    "soccer_france_ligue_two":    "F2",
    "soccer_netherlands_eredivisie": "N1",
    "soccer_belgium_first_div":   "B1",
    "soccer_portugal_primeira_liga": "P1",
    "soccer_greece_super_league":  "G1",
    "soccer_turkey_super_league":  "T1",
    "soccer_spl":             "SC0",
    "soccer_usa_mls":         "USA_MLS",
    "soccer_mexico_ligamx":   "MEX_LigaMX",
}

LEAGUE_TO_SPORT = {v: k for k, v in SPORT_TO_LEAGUE.items()}

# Preferred bookmakers (US-accessible, in priority order)
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "pointsbetus",
                   "williamhill_us", "bovada", "betrivers", "unibet_us"]


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        print("⚠️  ODDS_API_KEY not set. Get a free key at https://the-odds-api.com", file=sys.stderr)
    return key


def _get(endpoint: str, params: Dict[str, str] = {}) -> dict:
    """Make a GET request to The Odds API."""
    key = _api_key()
    if not key:
        return {"error": "No API key"}

    params["apiKey"] = key
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API_HOST}{endpoint}?{qs}"

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        # Log quota usage
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"  API quota: {used} used, {remaining} remaining", file=sys.stderr)
        return data


def get_sport_keys() -> List[Dict]:
    """Get list of available sports (free, doesn't count against quota)."""
    data = _get("/v4/sports", {"all": "false"})
    if isinstance(data, dict) and "error" in data:
        return []
    # Filter to soccer only
    return [s for s in data if s.get("group", "").lower() == "soccer"]


def fetch_ht_odds(sport_key: str,
                  regions: str = "us,uk,eu",
                  odds_format: str = "decimal") -> List[Dict]:
    """
    Fetch real HT draw odds for a given sport.

    Returns list of dicts with:
        home, away, commence_time, league,
        ht_home_odds, ht_draw_odds, ht_away_odds,
        bookmaker, best_ht_draw_odds, best_bookmaker
    """
    try:
        data = _get(f"/v4/sports/{sport_key}/odds", {
            "regions": regions,
            "markets": HT_MARKET,
            "oddsFormat": odds_format,
        })
    except Exception as e:
        print(f"  ❌ Failed to fetch {sport_key}: {e}", file=sys.stderr)
        return []

    if isinstance(data, dict) and "error" in data:
        print(f"  ❌ API error: {data.get('message', data['error'])}", file=sys.stderr)
        return []

    league_code = SPORT_TO_LEAGUE.get(sport_key, sport_key)
    results = []

    for event in data:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")

        # Extract HT odds from all bookmakers
        book_odds = []
        for bm in event.get("bookmakers", []):
            bm_key = bm.get("key", "")
            for market in bm.get("markets", []):
                if market.get("key") != HT_MARKET:
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                ht_home = outcomes.get(home, outcomes.get("Home", None))
                ht_draw = outcomes.get("Draw", None)
                ht_away = outcomes.get(away, outcomes.get("Away", None))

                if ht_draw is not None:
                    book_odds.append({
                        "bookmaker": bm_key,
                        "ht_home": ht_home,
                        "ht_draw": ht_draw,
                        "ht_away": ht_away,
                    })

        if not book_odds:
            continue

        # Find best HT draw odds
        best = max(book_odds, key=lambda x: x["ht_draw"])

        # Also get a preferred US book if available
        preferred = None
        for book in PREFERRED_BOOKS:
            match = [b for b in book_odds if b["bookmaker"] == book]
            if match:
                preferred = match[0]
                break

        use = preferred or best

        results.append({
            "home": home,
            "away": away,
            "commence_time": commence,
            "league": league_code,
            "sport_key": sport_key,
            "ht_draw_odds": use["ht_draw"],
            "ht_home_odds": use.get("ht_home"),
            "ht_away_odds": use.get("ht_away"),
            "bookmaker": use["bookmaker"],
            "best_ht_draw_odds": best["ht_draw"],
            "best_bookmaker": best["bookmaker"],
            "n_books": len(book_odds),
            "all_books": book_odds,
        })

    return results


def fetch_all_soccer_ht_odds() -> List[Dict]:
    """Fetch HT odds for all available soccer leagues."""
    sports = get_sport_keys()
    if not sports:
        return []

    all_odds = []
    for sport in sports:
        key = sport["key"]
        if key not in SPORT_TO_LEAGUE:
            continue
        print(f"  Fetching {sport.get('title', key)}...", file=sys.stderr)
        odds = fetch_ht_odds(key)
        all_odds.extend(odds)
        if not odds:
            print(f"    (no HT odds available)", file=sys.stderr)

    return all_odds


def print_odds(odds_list: List[Dict]) -> None:
    """Pretty-print HT draw odds."""
    if not odds_list:
        print("  No HT draw odds found.")
        return

    print(f"\n{'═'*76}")
    print("  REAL HT DRAW ODDS (from The Odds API)")
    print(f"  Market: {HT_MARKET} (1st Half 3-Way Result)")
    print(f"{'═'*76}")
    print(f"  {'Match':<35} {'Lg':>6}  {'HT Draw':>8}  {'Book':>12}  {'Best':>8}  {'#':>3}")
    print(f"  {'-'*35} {'-'*6}  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*3}")

    for o in sorted(odds_list, key=lambda x: x["ht_draw_odds"]):
        match_str = f"{o['home']} vs {o['away']}"
        if len(match_str) > 35:
            match_str = match_str[:32] + "..."
        print(f"  {match_str:<35} {o['league']:>6}  "
              f"{o['ht_draw_odds']:>8.2f}  {o['bookmaker']:>12}  "
              f"{o['best_ht_draw_odds']:>8.2f}  {o['n_books']:>3}")

    print(f"{'═'*76}")
    print(f"  {len(odds_list)} matches with HT draw odds")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch real HT draw odds from The Odds API")
    parser.add_argument("command", choices=["sports", "odds", "all"],
                        help="sports: list soccer sports; odds: fetch for one sport; all: fetch all")
    parser.add_argument("sport_key", nargs="?", default=None,
                        help="Sport key (e.g. soccer_epl) — required for 'odds' command")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--regions", default="us,uk,eu", help="Comma-separated regions")
    args = parser.parse_args()

    if args.command == "sports":
        sports = get_sport_keys()
        if args.json:
            print(json.dumps(sports, indent=2))
        else:
            print(f"\n  Available soccer sports ({len(sports)}):")
            for s in sports:
                mapped = "→ " + SPORT_TO_LEAGUE[s["key"]] if s["key"] in SPORT_TO_LEAGUE else "(unmapped)"
                print(f"    {s['key']:<40} {s.get('title',''):<30} {mapped}")

    elif args.command == "odds":
        if not args.sport_key:
            parser.error("'odds' command requires a sport_key argument")
        odds = fetch_ht_odds(args.sport_key, regions=args.regions)
        if args.json:
            print(json.dumps(odds, indent=2))
        else:
            print_odds(odds)

    elif args.command == "all":
        odds = fetch_all_soccer_ht_odds()
        if args.json:
            print(json.dumps(odds, indent=2))
        else:
            print_odds(odds)


if __name__ == "__main__":
    main()
