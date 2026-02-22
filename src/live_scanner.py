"""
Live Match Scanner — Half-Time Draw Prediction
===============================================
Fetches today's football fixtures, runs the v2 ensemble predictor,
compares model P(draw) vs market implied P(draw), and flags value bets.

Data sources:
  - Fixtures:  The Odds API (https://the-odds-api.com)
               Set ODDS_API_KEY env var. If absent → DEMO MODE (historical data).
  - Weather:   wttr.in (no API key, free)
  - Odds:      Pulled directly from The Odds API (multiple books)
  - Model:     models/v2/ ensemble predictor

Output: Ranked list of today's matches by predicted edge (model − market).
        Highlights where model disagrees with market by >3%.

Usage:
    python src/live_scanner.py
    python src/live_scanner.py --demo         # force demo mode
    python src/live_scanner.py --date 2025-03-15  # scan specific date
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# ── Weather helper ────────────────────────────────────────────────────────────

def get_weather(city: str) -> Optional[Dict]:
    """Fetch current weather from wttr.in for a city. Returns dict or None."""
    try:
        import urllib.request
        url  = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
        req  = urllib.request.Request(url, headers={"User-Agent": "half-time-predictor/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())

        current = data["current_condition"][0]
        return {
            "city":        city,
            "temp_c":      int(current.get("temp_C", 10)),
            "feels_like":  int(current.get("FeelsLikeC", 10)),
            "humidity":    int(current.get("humidity", 60)),
            "wind_kmph":   int(current.get("windspeedKmph", 10)),
            "description": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
            "is_raining":  any(
                kw in current.get("weatherDesc", [{}])[0].get("value", "").lower()
                for kw in ["rain", "drizzle", "shower", "snow", "sleet"]
            ),
        }
    except Exception:
        return None


def weather_notes(w: Optional[Dict]) -> str:
    """Human-readable weather note."""
    if w is None:
        return ""
    notes = []
    if w["is_raining"]:
        notes.append(f"Rain/wet ({w['description']})")
    if w["wind_kmph"] > 40:
        notes.append(f"Strong wind {w['wind_kmph']}km/h")
    if w["temp_c"] < 2:
        notes.append(f"Cold {w['temp_c']}°C")
    return " | ".join(notes) if notes else f"{w['description']} {w['temp_c']}°C"


# ── Odds API helpers ──────────────────────────────────────────────────────────

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# League codes → odds API sport keys
LEAGUE_MAP = {
    "soccer_efl_champ":  {"league": "E1", "name": "EFL Championship"},
    "soccer_england_league1": {"league": "E2", "name": "League One"},
    "soccer_england_league2": {"league": "E3", "name": "League Two"},
    "soccer_epl":        {"league": "E0", "name": "Premier League"},
    "soccer_germany_bundesliga": {"league": "D1", "name": "Bundesliga"},
    "soccer_spain_la_liga": {"league": "SP1", "name": "La Liga"},
    "soccer_italy_serie_a": {"league": "I1", "name": "Serie A"},
    "soccer_france_ligue_one": {"league": "F1", "name": "Ligue 1"},
}

FOOTBALL_SPORTS = list(LEAGUE_MAP.keys())


def fetch_live_fixtures(api_key: str, target_date: date) -> List[Dict]:
    """
    Fetch today's fixtures from The Odds API.
    Returns list of match dicts with odds from multiple books.
    """
    import urllib.request
    import urllib.parse

    results = []
    date_str = target_date.strftime("%Y-%m-%dT00:00:00Z")
    date_end = (target_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    for sport_key in FOOTBALL_SPORTS:
        url = (f"{ODDS_API_BASE}/sports/{sport_key}/odds"
               f"?apiKey={api_key}"
               f"&regions=uk,eu"
               f"&markets=h2h"
               f"&oddsFormat=decimal"
               f"&commenceTimeFrom={date_str}"
               f"&commenceTimeTo={date_end}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "half-time-predictor/1.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                events = json.loads(r.read().decode())

            league_info = LEAGUE_MAP.get(sport_key, {})
            for event in events:
                home = event.get("home_team", "")
                away = event.get("away_team", "")

                # Build odds dict from bookmakers
                odds = {}
                for bm in event.get("bookmakers", []):
                    bm_key = bm.get("key", "").upper()[:4]
                    for market in bm.get("markets", []):
                        if market.get("key") == "h2h":
                            for outcome in market.get("outcomes", []):
                                name = outcome.get("name", "")
                                price = float(outcome.get("price", 3.0))
                                if name == home:
                                    odds[f"{bm_key}H"] = price
                                elif name == away:
                                    odds[f"{bm_key}A"] = price
                                elif name == "Draw":
                                    odds[f"{bm_key}D"] = price

                results.append({
                    "home_team":    home,
                    "away_team":    away,
                    "league":       league_info.get("league", ""),
                    "league_name":  league_info.get("name", sport_key),
                    "sport_key":    sport_key,
                    "commence_time": event.get("commence_time", ""),
                    "odds":         odds,
                })
        except Exception as e:
            pass  # Skip failed sport

    return results


# ── Demo fixtures (historical matches for testing) ────────────────────────────

DEMO_FIXTURES = [
    # ── MLS — February 22 2026 (season opener weekend) ────────────────────────
    {
        "home_team": "LA Galaxy",
        "away_team": "NYCFC",
        "league":    "MLS",
        "league_name": "MLS",
        "commence_time": "22:30 ET",
        # DraftKings: Home +125, Draw +260, Away +180
        # American → decimal: +125=2.25, +260=3.60, +180=2.80
        "odds": {"B365H": 2.25, "B365D": 3.60, "B365A": 2.80,
                 "BWH": 2.20, "BWD": 3.55, "BWA": 2.85},
    },
    {
        "home_team": "Seattle Sounders",
        "away_team": "Colorado Rapids",
        "league":    "MLS",
        "league_name": "MLS",
        "commence_time": "01:00 ET",
        # DraftKings: Home -275, Draw +390, Away +600
        # -275→1.364, +390→4.90, +600→7.00
        "odds": {"B365H": 1.36, "B365D": 4.90, "B365A": 7.00,
                 "BWH": 1.38, "BWD": 4.80, "BWA": 6.80},
    },
    {
        "home_team": "LAFC",
        "away_team": "Inter Miami",
        "league":    "MLS",
        "league_name": "MLS",
        "commence_time": "22:00 ET",
        # LAFC strong home side; Inter Miami Messi-era roster
        # Estimated: LAFC -150 / Draw +310 / Miami +350
        "odds": {"B365H": 1.67, "B365D": 4.10, "B365A": 4.50,
                 "BWH": 1.65, "BWD": 4.05, "BWA": 4.55},
    },
    # ── EFL Championship (for model comparison with in-sample leagues) ─────────
    {
        "home_team": "Leeds United",
        "away_team": "Sheffield Utd",
        "league":    "E1",
        "league_name": "EFL Championship",
        "commence_time": "15:00 GMT",
        "odds": {"B365H": 2.30, "B365D": 3.30, "B365A": 3.20,
                 "BWH": 2.25, "BWD": 3.20, "BWA": 3.30,
                 "PSH": 2.35, "PSD": 3.35, "PSA": 3.15,
                 "PSCH": 2.32, "PSCD": 3.38, "PSCA": 3.18},
    },
    {
        "home_team": "Sunderland",
        "away_team": "Coventry",
        "league":    "E1",
        "league_name": "EFL Championship",
        "commence_time": "15:00 GMT",
        "odds": {"B365H": 2.00, "B365D": 3.40, "B365A": 3.80,
                 "BWH": 1.95, "BWD": 3.30, "BWA": 3.90,
                 "PSH": 2.02, "PSD": 3.45, "PSA": 3.75},
    },
    {
        "home_team": "Man City",
        "away_team": "Liverpool",
        "league":    "E0",
        "league_name": "Premier League",
        "commence_time": "16:30 GMT",
        "odds": {"B365H": 2.10, "B365D": 3.60, "B365A": 3.50,
                 "PSH": 2.12, "PSD": 3.65, "PSA": 3.45,
                 "PSCH": 2.08, "PSCD": 3.70, "PSCA": 3.50},
    },
    {
        "home_team": "West Brom",
        "away_team": "QPR",
        "league":    "E1",
        "league_name": "EFL Championship",
        "commence_time": "15:00 GMT",
        "odds": {"B365H": 1.73, "B365D": 3.60, "B365A": 5.00,
                 "BWH": 1.70, "BWD": 3.50, "BWA": 5.20},
    },
]

# City lookup for venue weather
CITY_MAP = {
    # MLS
    "LA Galaxy":        "Carson",
    "NYCFC":            "New York",
    "Seattle Sounders": "Seattle",
    "Colorado Rapids":  "Commerce City",
    "LAFC":             "Los Angeles",
    "Inter Miami":      "Fort Lauderdale",
    "LA FC":            "Los Angeles",
    "New York City FC": "New York",
    # EFL / EPL
    "Leeds United":     "Leeds",
    "Sheffield Utd":    "Sheffield",
    "Sunderland":       "Sunderland",
    "Coventry":         "Coventry",
    "Watford":          "Watford",
    "Middlesbrough":    "Middlesbrough",
    "Man City":         "Manchester",
    "Liverpool":        "Liverpool",
    "West Brom":        "West Bromwich",
    "QPR":              "London",
    "Bristol City":     "Bristol",
    "Hull City":        "Hull",
    "Arsenal":          "London",
    "Chelsea":          "London",
    "Man United":       "Manchester",
    "Tottenham":        "London",
    "Leicester":        "Leicester",
    "Nottm Forest":     "Nottingham",
}


# ── Scanner ───────────────────────────────────────────────────────────────────

class LiveScanner:
    """
    Scans today's fixtures and ranks by model edge over market.

    Parameters
    ----------
    min_edge : float
        Minimum (model − market) probability difference to flag as value.
    fetch_weather : bool
        Whether to fetch weather data for each fixture.
    """

    def __init__(
        self,
        min_edge: float = 0.03,
        fetch_weather: bool = True,
    ):
        self.min_edge       = min_edge
        self.fetch_weather  = fetch_weather
        self.predictor_: Optional[EnsemblePredictor] = None

    def load_predictor(self, paths: Dict[str, str]) -> None:
        """Load the ensemble predictor from saved model files."""
        from src.ensemble_predictor import EnsemblePredictor
        self.predictor_ = EnsemblePredictor.load_full(**paths)
        print(f"  Ensemble predictor loaded.")

    def _market_implied_draw(self, odds: Dict[str, float]) -> Optional[float]:
        """Compute normalized B365 implied P(draw) from odds dict."""
        h_key = next((k for k in odds if k.endswith("H") and "C" not in k), None)
        d_key = next((k for k in odds if k.endswith("D") and "C" not in k), None)
        a_key = next((k for k in odds if k.endswith("A") and "C" not in k), None)

        # Prefer B365
        for prefix in ["B365", "BW", "PS", "WH"]:
            hk, dk, ak = f"{prefix}H", f"{prefix}D", f"{prefix}A"
            if all(k in odds and odds[k] > 1 for k in [hk, dk, ak]):
                h, d, a = odds[hk], odds[dk], odds[ak]
                total = (1/h) + (1/d) + (1/a)
                return (1/d) / total

        return None

    def _multi_book_consensus(self, odds: Dict[str, float]) -> Optional[float]:
        """Compute average normalized P(draw) across all available books."""
        probs = []
        prefixes = ["B365", "BW", "IW", "LB", "WH", "VC", "PS", "SJ"]
        for prefix in prefixes:
            hk, dk, ak = f"{prefix}H", f"{prefix}D", f"{prefix}A"
            if all(k in odds and float(odds[k]) > 1 for k in [hk, dk, ak]):
                h, d, a = float(odds[hk]), float(odds[dk]), float(odds[ak])
                total = (1/h) + (1/d) + (1/a)
                probs.append((1/d) / total)
        return float(np.mean(probs)) if probs else None

    def scan(
        self,
        fixtures: List[Dict],
        target_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Run the ensemble predictor on all fixtures.
        Returns DataFrame ranked by (model_p_draw − market_p_draw).
        """
        if target_date is None:
            target_date = date.today()

        # Historical base rates from training (fallback)
        league_base_rates = {
            "E1": 0.420, "E0": 0.380, "E2": 0.415,
            "D1": 0.390, "SP1": 0.410, "I1": 0.405, "F1": 0.415,
        }

        results = []
        for fix in fixtures:
            home  = fix["home_team"]
            away  = fix["away_team"]
            league = fix.get("league", "")
            odds  = fix.get("odds", {})

            # Weather
            weather_str = ""
            weather_data = None
            if self.fetch_weather:
                city = CITY_MAP.get(home, home.split()[0])
                weather_data = get_weather(city)
                weather_str  = weather_notes(weather_data)

            # Model prediction
            if self.predictor_ is not None:
                result = self.predictor_.predict_single(
                    home_team=home, away_team=away, league=league,
                    market_odds=odds if odds else None,
                )
                model_p   = result["p_ensemble"]
                breakdown = result["breakdown"]
                ci_lo     = result["ci_lower"]
                ci_hi     = result["ci_upper"]
                qk        = result.get("quarter_kelly_stake")
                drivers   = result.get("key_drivers", [])
            else:
                model_p   = league_base_rates.get(league, 0.42)
                breakdown = {}
                ci_lo, ci_hi = model_p * 0.8, model_p * 1.2
                qk        = None
                drivers   = []

            # Market
            mkt_p    = self._market_implied_draw(odds) or league_base_rates.get(league, 0.42)
            cons_p   = self._multi_book_consensus(odds) or mkt_p
            base_p   = league_base_rates.get(league, 0.42)

            edge = model_p - mkt_p

            # Best draw odds available
            best_d_odds = None
            for prefix in ["PS", "B365", "Max", "BW", "WH"]:
                dk = f"{prefix}D"
                dk_close = f"{prefix}CD"
                for k in [dk_close, dk]:
                    if k in odds and float(odds[k]) > 1:
                        best_d_odds = float(odds[k])
                        break
                if best_d_odds:
                    break

            ev = None
            if best_d_odds and model_p > 0:
                ev = model_p * (best_d_odds - 1) - (1 - model_p)

            results.append({
                "home_team":    home,
                "away_team":    away,
                "league":       fix.get("league_name", league),
                "kickoff":      fix.get("commence_time", ""),
                "model_p":      round(model_p, 4),
                "market_p":     round(mkt_p, 4),
                "consensus_p":  round(cons_p, 4),
                "base_rate":    round(base_p, 4),
                "edge":         round(edge, 4),
                "ci_lower":     round(ci_lo, 4),
                "ci_upper":     round(ci_hi, 4),
                "best_draw_odds": best_d_odds,
                "ev":           round(ev, 4) if ev is not None else None,
                "quarter_kelly": qk,
                "weather":      weather_str,
                "weather_data": weather_data,
                "breakdown":    breakdown,
                "key_drivers":  drivers,
                "flagged":      abs(edge) >= self.min_edge,
                "value_bet":    edge >= self.min_edge,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("edge", ascending=False).reset_index(drop=True)
        return df

    def print_report(
        self,
        scan_df: pd.DataFrame,
        target_date: Optional[date] = None,
        show_all: bool = True,
    ) -> None:
        """Pretty-print the scan results to the terminal."""
        from colorama import Fore, Style, init
        try:
            init(autoreset=True)
            USE_COLOR = True
        except ImportError:
            USE_COLOR = False

        if target_date is None:
            target_date = date.today()

        width = 72
        print("\n" + "=" * width)
        print(f"  HALF-TIME DRAW SCANNER — {target_date.strftime('%A, %d %B %Y')}")
        print(f"  Edge threshold: ≥{self.min_edge*100:.0f}%  |  "
              f"Matches scanned: {len(scan_df)}")
        print("=" * width)

        if scan_df.empty:
            print("  No fixtures found for this date.")
            return

        value_bets = scan_df[scan_df["value_bet"]]
        if not value_bets.empty:
            tag = "VALUE BETS (model > market by ≥" + f"{self.min_edge*100:.0f}%)"
            print(f"\n  ★ {tag}")
            print("  " + "-" * (width - 2))
            for _, row in value_bets.iterrows():
                self._print_match_row(row, highlight=True, use_color=USE_COLOR)
        else:
            print(f"\n  No value bets found (edge < {self.min_edge*100:.0f}%)")

        if show_all:
            non_value = scan_df[~scan_df["value_bet"]]
            if not non_value.empty:
                print(f"\n  ALL OTHER FIXTURES")
                print("  " + "-" * (width - 2))
                for _, row in non_value.iterrows():
                    self._print_match_row(row, highlight=False, use_color=USE_COLOR)

        # Summary stats
        print("\n  " + "-" * (width - 2))
        print(f"  Avg model P(draw):  {scan_df['model_p'].mean():.3f}")
        print(f"  Avg market P(draw): {scan_df['market_p'].mean():.3f}")
        print(f"  Value bets found:   {len(value_bets)}")
        print("=" * width + "\n")

    def _print_match_row(
        self,
        row: pd.Series,
        highlight: bool = False,
        use_color: bool = False,
    ) -> None:
        star   = "★ " if row["value_bet"] else "  "
        league = row.get("league", "")[:18]
        ko     = row.get("kickoff", "")[:8]

        edge_pct = row["edge"] * 100
        edge_str = f"{edge_pct:+.1f}%"

        draw_odds = row.get("best_draw_odds")
        odds_str  = f"@{draw_odds:.2f}" if draw_odds else ""

        ev = row.get("ev")
        ev_str = f"EV={ev:+.3f}" if ev is not None else ""

        qk = row.get("quarter_kelly")
        qk_str = f"QK={qk:.3f}u" if qk else ""

        weather = row.get("weather", "")
        weather_str = f"  [{weather}]" if weather else ""

        print(f"\n  {star}{row['home_team']:<18} vs {row['away_team']:<18} [{league}]  {ko}")
        print(f"    Model: {row['model_p']:.3f}  Market: {row['market_p']:.3f}  "
              f"Consensus: {row['consensus_p']:.3f}  Base: {row['base_rate']:.3f}")
        print(f"    Edge: {edge_str}  CI: [{row['ci_lower']:.3f}–{row['ci_upper']:.3f}]  "
              f"{odds_str}  {ev_str}  {qk_str}{weather_str}")

        # Breakdown
        bd = row.get("breakdown", {})
        if bd:
            bd_parts = [f"{k[:4]}={v:.3f}" for k, v in bd.items() if v is not None]
            print(f"    Breakdown: {' | '.join(bd_parts)}")

        # Key drivers
        drivers = row.get("key_drivers", [])
        if drivers:
            top = drivers[:2]
            drv_str = "  ".join(
                f"{d['factor']}: {'+' if d['delta']>0 else ''}{d['delta']:.3f}"
                for d in top
            )
            print(f"    Drivers: {drv_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Half-Time Draw Live Scanner")
    parser.add_argument("--demo",  action="store_true", help="Force demo mode")
    parser.add_argument("--date",  type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--edge",  type=float, default=0.03,
                        help="Minimum edge to flag (default 0.03)")
    parser.add_argument("--no-weather", action="store_true",
                        help="Skip weather lookup")
    parser.add_argument("--all", action="store_true",
                        help="Show all fixtures (not just value bets)")
    args = parser.parse_args()

    # Parse target date
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = date.today()

    # Check for API key
    api_key  = os.environ.get("ODDS_API_KEY", "")
    use_demo = args.demo or not api_key

    if use_demo:
        print("\n  [DEMO MODE] No ODDS_API_KEY found — using historical demo fixtures.")
        print("  Set ODDS_API_KEY=<your_key> for live data from the-odds-api.com")
        print("  Get a free key at: https://the-odds-api.com\n")
        fixtures = DEMO_FIXTURES
    else:
        print(f"\n  Fetching live fixtures for {target_date} from The Odds API...")
        fixtures = fetch_live_fixtures(api_key, target_date)
        print(f"  Found {len(fixtures)} fixtures")

    if not fixtures:
        print("  No fixtures found. Try --demo for a demo.")
        return

    # Load ensemble predictor
    scanner = LiveScanner(min_edge=args.edge, fetch_weather=not args.no_weather)

    paths_file = Path("models/v2/submodel_paths.json")
    if paths_file.exists():
        with open(paths_file) as f:
            paths = json.load(f)
        try:
            scanner.load_predictor(paths)
        except Exception as e:
            print(f"  Warning: could not load ensemble ({e}). Using base rates.")
    else:
        print("  Warning: models/v2/ not found. Run src/train_v2.py first.")
        print("  Using league base rates as predictions.")

    # Run scan
    print(f"\n  Scanning {len(fixtures)} fixtures...")
    scan_df = scanner.scan(fixtures, target_date=target_date)

    # Print report
    scanner.print_report(scan_df, target_date=target_date, show_all=True)

    # Save to file
    out_path = Path(f"models/v2/scan_{target_date.strftime('%Y%m%d')}.json")
    scan_df.drop(columns=["weather_data", "breakdown", "key_drivers"],
                 errors="ignore").to_json(str(out_path), orient="records", indent=2)
    print(f"  Scan results saved → {out_path}")


if __name__ == "__main__":
    main()
