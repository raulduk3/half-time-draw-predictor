"""
Single Match Prediction Interface — Half-Time Draw
===================================================
Interactive / command-line tool for predicting P(HT draw) for any match.

Usage (CLI):
    python src/predict_match.py "Leeds United" "Sheffield Utd" --league E1
    python src/predict_match.py "Man City" "Liverpool" --league E0 --referee "M Dean"
    python src/predict_match.py "Sunderland" "Coventry" \\
        --league E1 --b365h 2.2 --b365d 3.3 --b365a 3.6

Usage (Python):
    from src.predict_match import MatchPredictor
    mp = MatchPredictor.load()
    result = mp.predict("Leeds United", "Sheffield Utd", league="E1")
    mp.print_result(result)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


# ── MatchPredictor ────────────────────────────────────────────────────────────

class MatchPredictor:
    """
    Wraps EnsemblePredictor with additional context:
      - Looks up team recent form from the mega dataset
      - Fetches weather from wttr.in
      - Formats result for terminal display
    """

    def __init__(self):
        self.predictor_ = None
        self.mega_df_   = None

    @classmethod
    def load(
        cls,
        paths_file: str = "models/v2/submodel_paths.json",
        mega_path: str  = "data/processed/mega_dataset.parquet",
    ) -> "MatchPredictor":
        """Load the full ensemble from saved models."""
        mp = cls()

        # Load ensemble
        p = Path(paths_file)
        if p.exists():
            with open(p) as f:
                paths = json.load(f)
            try:
                from src.ensemble_predictor import EnsemblePredictor
                mp.predictor_ = EnsemblePredictor.load_full(**paths)
                print("  Ensemble loaded successfully.")
            except Exception as e:
                print(f"  Warning: ensemble load failed ({e}). Using base rates.")
        else:
            print(f"  Warning: {paths_file} not found. Run src/train_v2.py first.")

        # Load mega dataset for form lookup
        if Path(mega_path).exists():
            mp.mega_df_ = pd.read_parquet(mega_path)
            mp.mega_df_["Date"] = pd.to_datetime(mp.mega_df_["Date"])
            mp.mega_df_ = mp.mega_df_.sort_values("Date").reset_index(drop=True)

        return mp

    def get_recent_form(
        self,
        team: str,
        n_matches: int = 5,
        before_date: Optional[pd.Timestamp] = None,
    ) -> Dict:
        """
        Look up a team's recent form from the mega dataset.
        Returns dict with: last_n_ht_draws, last_n_ht_goals_scored, etc.
        """
        if self.mega_df_ is None:
            return {}

        df = self.mega_df_
        if before_date:
            df = df[df["Date"] < before_date]

        home_mask = df["HomeTeam"].str.lower() == team.lower()
        away_mask = df["AwayTeam"].str.lower() == team.lower()
        team_matches = df[home_mask | away_mask].tail(n_matches)

        if team_matches.empty:
            return {"found": False, "team": team}

        ht_draws = (team_matches["HTHG"] == team_matches["HTAG"]).sum()
        ht_goals = []
        for _, row in team_matches.iterrows():
            if row["HomeTeam"].lower() == team.lower():
                ht_goals.append(row.get("HTHG", 0))
            else:
                ht_goals.append(row.get("HTAG", 0))

        return {
            "found":          True,
            "team":           team,
            "n_matches":      len(team_matches),
            "ht_draw_rate":   round(ht_draws / len(team_matches), 3),
            "ht_goals_avg":   round(np.mean(ht_goals), 2) if ht_goals else 0,
            "last_dates":     [str(d.date()) for d in team_matches["Date"].tail(3)],
        }

    def get_league_base_rate(self, league: str) -> float:
        """Get historical HT draw rate for a league."""
        if self.mega_df_ is None:
            return 0.42

        mask = self.mega_df_["league"] == league
        sub  = self.mega_df_[mask]
        if sub.empty:
            return 0.42

        return float((sub["HTHG"] == sub["HTAG"]).mean())

    def predict(
        self,
        home_team: str,
        away_team: str,
        league: str = "",
        referee: Optional[str] = None,
        market_odds: Optional[Dict[str, float]] = None,
        fetch_weather: bool = True,
    ) -> Dict:
        """
        Run the full prediction pipeline for one match.

        Returns a rich dict with:
          - p_ensemble, ci_lower, ci_upper
          - breakdown, key_drivers
          - recent_form (home + away)
          - league_base_rate
          - market_implied_p
          - edge (model − market)
          - quarter_kelly_stake
          - expected_value
          - weather (if fetch_weather=True)
        """
        # Get model prediction
        if self.predictor_ is not None:
            result = self.predictor_.predict_single(
                home_team=home_team,
                away_team=away_team,
                league=league,
                referee=referee,
                market_odds=market_odds,
            )
        else:
            base_rate = self.get_league_base_rate(league) if league else 0.42
            result = {
                "p_ensemble": base_rate,
                "ci_lower":   base_rate * 0.8,
                "ci_upper":   base_rate * 1.2,
                "breakdown":  {},
                "key_drivers": [],
                "quarter_kelly_stake": None,
            }

        # Recent form
        home_form = self.get_recent_form(home_team)
        away_form = self.get_recent_form(away_team)

        # League base rate
        league_base_rate = self.get_league_base_rate(league) if league else 0.42

        # Market implied P(draw)
        market_implied_p = None
        if market_odds:
            for prefix in ["B365", "BW", "PS", "WH"]:
                hk, dk, ak = f"{prefix}H", f"{prefix}D", f"{prefix}A"
                if all(k in market_odds and float(market_odds[k]) > 1
                       for k in [hk, dk, ak]):
                    h, d, a = float(market_odds[hk]), float(market_odds[dk]), float(market_odds[ak])
                    total = (1/h) + (1/d) + (1/a)
                    market_implied_p = (1/d) / total
                    break

        edge = None
        if market_implied_p is not None:
            edge = result["p_ensemble"] - market_implied_p

        # EV and Kelly
        ev, qk_stake = None, result.get("quarter_kelly_stake")
        if market_odds:
            best_d_odds = None
            for prefix in ["PSCH", "PSCD", "B365CD", "B365D", "PSD", "MaxD", "AvgD"]:
                if prefix in market_odds and float(market_odds.get(prefix, 0)) > 1:
                    best_d_odds = float(market_odds[prefix])
                    break
            if best_d_odds:
                p = result["p_ensemble"]
                ev = round(p * (best_d_odds - 1) - (1 - p), 4)
                if ev > 0:
                    b = best_d_odds - 1
                    kelly = max(0, (p * b - (1 - p)) / b)
                    qk_stake = round(kelly * 0.25, 4)
                else:
                    qk_stake = 0.0

        # Weather
        weather_data = None
        weather_str  = ""
        if fetch_weather:
            from src.live_scanner import get_weather, weather_notes, CITY_MAP
            city = CITY_MAP.get(home_team, home_team.split()[0])
            weather_data = get_weather(city)
            weather_str  = weather_notes(weather_data)

        result.update({
            "home_form":        home_form,
            "away_form":        away_form,
            "league_base_rate": round(league_base_rate, 4),
            "market_implied_p": round(market_implied_p, 4) if market_implied_p else None,
            "edge":             round(edge, 4) if edge is not None else None,
            "expected_value":   ev,
            "quarter_kelly_stake": qk_stake,
            "weather":          weather_str,
            "weather_data":     weather_data,
        })

        return result

    def print_result(self, result: Dict) -> None:
        """Pretty-print prediction to terminal."""
        width = 68
        print("\n" + "═" * width)
        print(f"  HALF-TIME DRAW PREDICTION")
        print("═" * width)

        p    = result["p_ensemble"]
        ci_l = result["ci_lower"]
        ci_h = result["ci_upper"]

        home = result.get("home_team", "Home")
        away = result.get("away_team", "Away")

        if "home_team" not in result:
            # Extract from first key driver or use generic
            home, away = "Home", "Away"

        print(f"\n  {home} vs {away}")
        league = result.get("league", "")
        if league:
            base = result.get("league_base_rate", 0.42)
            print(f"  League: {league}  |  Historical base rate: {base:.1%}")

        referee = result.get("referee")
        if referee:
            print(f"  Referee: {referee}")

        weather = result.get("weather", "")
        if weather:
            print(f"  Weather: {weather}")

        print()
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  P(HT Draw)  = {p*100:>5.1f}%                  │")
        print(f"  │  90% CI      = {ci_l*100:.1f}% – {ci_h*100:.1f}%            │")
        base = result.get("league_base_rate", 0.42)
        diff = p - base
        diff_str = f"{'+' if diff>0 else ''}{diff*100:.1f}% vs base"
        print(f"  │  vs Base     = {diff_str:<27}│")
        print(f"  └─────────────────────────────────────────┘")

        # Market comparison
        mkt_p = result.get("market_implied_p")
        edge  = result.get("edge")
        if mkt_p is not None:
            print(f"\n  Market implied P(draw): {mkt_p:.1%}")
            if edge is not None:
                edge_str = f"{'+' if edge>0 else ''}{edge*100:.1f}%"
                flag = " ← VALUE BET" if edge >= 0.03 else (" ← FADE" if edge <= -0.03 else "")
                print(f"  Model edge:             {edge_str}{flag}")

        # EV and Kelly
        ev = result.get("expected_value")
        qk = result.get("quarter_kelly_stake")
        if ev is not None:
            print(f"\n  Expected Value:   {ev:+.4f} per unit")
        if qk is not None and qk > 0:
            print(f"  Quarter-Kelly:    {qk:.4f} units staked")
        elif qk == 0:
            print(f"  Quarter-Kelly:    0 (no edge)")

        # Signal breakdown
        bd = result.get("breakdown", {})
        if bd:
            print(f"\n  ── Signal Breakdown ──────────────────────")
            signal_labels = {
                "dixon_coles":       "Dixon-Coles (bivariate Poisson)",
                "elo":               "Elo rating system",
                "xgboost":           "XGBoost (mega model)",
                "lightgbm":          "LightGBM (mega model)",
                "market_consensus":  "Market consensus (multi-book)",
                "referee_adj":       "Referee adjustment factor",
            }
            for key, label in signal_labels.items():
                val = bd.get(key)
                if val is not None:
                    if key == "referee_adj":
                        print(f"  {label:<35} ×{val:.3f}")
                    else:
                        bar_len = int(val * 30)
                        bar = "█" * bar_len + "░" * (30 - bar_len)
                        print(f"  {label:<35} {val:.3f}  [{bar}]")

        # Key drivers
        drivers = result.get("key_drivers", [])
        if drivers:
            print(f"\n  ── Key Drivers ───────────────────────────")
            for d in drivers[:4]:
                sign   = "↑" if d["direction"] == "toward draw" else "↓"
                print(f"  {sign} {d['factor']:<30} Δ{d['delta']:+.3f}")

        # Recent form
        hf = result.get("home_form", {})
        af = result.get("away_form", {})
        if hf.get("found") or af.get("found"):
            print(f"\n  ── Recent Form (last 5 matches) ──────────")
            for form, label in [(hf, "Home"), (af, "Away")]:
                if form.get("found"):
                    print(f"  {label} ({form['team']}):  "
                          f"HT draw rate={form['ht_draw_rate']:.0%}  "
                          f"Avg HT goals={form['ht_goals_avg']:.1f}")

        print("\n" + "═" * width + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict P(HT draw) for a single match",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict_match.py "Leeds United" "Sheffield Utd" --league E1
  python src/predict_match.py "Man City" "Liverpool" --league E0 --referee "M Dean"
  python src/predict_match.py "Sunderland" "Coventry" --league E1 \\
      --b365h 2.2 --b365d 3.3 --b365a 3.6 --psh 2.18 --psd 3.38 --psa 3.22
        """
    )
    parser.add_argument("home_team", type=str, help="Home team name")
    parser.add_argument("away_team", type=str, help="Away team name")
    parser.add_argument("--league",   type=str, default="", help="League code (e.g. E1, E0)")
    parser.add_argument("--referee",  type=str, default=None, help="Referee name")
    parser.add_argument("--no-weather", action="store_true", help="Skip weather fetch")

    # Odds
    parser.add_argument("--b365h", type=float, default=None, help="B365 home odds")
    parser.add_argument("--b365d", type=float, default=None, help="B365 draw odds")
    parser.add_argument("--b365a", type=float, default=None, help="B365 away odds")
    parser.add_argument("--psh",   type=float, default=None, help="Pinnacle home opening")
    parser.add_argument("--psd",   type=float, default=None, help="Pinnacle draw opening")
    parser.add_argument("--psa",   type=float, default=None, help="Pinnacle away opening")
    parser.add_argument("--psch",  type=float, default=None, help="Pinnacle home closing")
    parser.add_argument("--pscd",  type=float, default=None, help="Pinnacle draw closing")
    parser.add_argument("--psca",  type=float, default=None, help="Pinnacle away closing")

    args = parser.parse_args()

    # Build odds dict
    market_odds = {}
    if args.b365h: market_odds["B365H"] = args.b365h
    if args.b365d: market_odds["B365D"] = args.b365d
    if args.b365a: market_odds["B365A"] = args.b365a
    if args.psh:   market_odds["PSH"]   = args.psh
    if args.psd:   market_odds["PSD"]   = args.psd
    if args.psa:   market_odds["PSA"]   = args.psa
    if args.psch:  market_odds["PSCH"]  = args.psch
    if args.pscd:  market_odds["PSCD"]  = args.pscd
    if args.psca:  market_odds["PSCA"]  = args.psca

    # Load predictor
    print(f"\nLoading models from models/v2/ ...")
    mp = MatchPredictor.load()

    # Predict
    result = mp.predict(
        home_team=args.home_team,
        away_team=args.away_team,
        league=args.league,
        referee=args.referee,
        market_odds=market_odds if market_odds else None,
        fetch_weather=not args.no_weather,
    )

    # Add team names for print_result
    result["home_team"] = args.home_team
    result["away_team"] = args.away_team
    result["league"]    = args.league
    result["referee"]   = args.referee

    # Print
    mp.print_result(result)


if __name__ == "__main__":
    main()
