"""
V4 Match Predictor — Half-Time Draw
=====================================
Unified, canonical prediction interface for the V4 two-model system.

Architecture:
  MODEL A — Market estimate: LogisticRegression on 3 B365 log-odds features
  MODEL B — Fundamentals:    XGBoost on 42 non-odds stats (rolling + DC + Elo)
  SIGNAL  — Inverted Edge = Model A − Model B
             Positive = market prices draw HIGHER than fundamentals
             → actual draw rate historically EXCEEDS market estimate

Kelly sizing (fixed — uses backtest-calibrated hit rates per edge bucket):
  Edge ≥ 5%:  p_bet = 0.470 (backtest hit rate at this threshold)
  Edge 3–5%:  p_bet = 0.460
  Edge 1–3%:  p_bet = 0.440
  Edge < 1%:  no bet

Usage:
    python src/predict_v4.py 'Home' 'Away' --odds H/D/A [--league CODE] [--referee NAME]

    python src/predict_v4.py 'Everton' 'Man United' --odds 3.70/3.75/1.91
    python src/predict_v4.py 'Alaves' 'Girona' --odds 2.30/3.00/3.50 --league SP1
    python src/predict_v4.py 'LA Galaxy' 'NYCFC' --odds 2.25/3.60/2.80 --league USA_MLS

    # Or legacy --b365h/--b365d/--b365a flags:
    python src/predict_v4.py 'Leeds United' 'Sheffield Utd' --b365h 2.2 --b365d 3.3 --b365a 3.6

Loads from: models/v4/v4_paths.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# Backtest-calibrated hit rates per edge bucket
# These are empirically derived from V3/V4 test-set backtesting
_KELLY_HIT_RATES = {
    "strong":   0.470,   # edge >= 5%
    "value":    0.460,   # edge 3–5%
    "marginal": 0.440,   # edge 1–3%
}


def _kelly_fraction(p_hit: float, decimal_odds: float) -> float:
    """Standard Kelly fraction: f = (p * b - q) / b."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p_hit * b - (1.0 - p_hit)) / b)


class V4Predictor:
    """
    V4 two-model predictor for P(HT draw).

    Model A — Market: log-odds LR (calibrated)
    Model B — Fundamentals: XGBoost/LightGBM on non-odds stats + DC + Elo (calibrated)
    Inverted Edge = Model A − Model B  (positive = value bet)
    """

    STRONG_VALUE_EDGE = 0.05
    VALUE_EDGE        = 0.03
    MARGINAL_EDGE     = 0.01

    def __init__(self):
        self.lr_a      = None
        self.scaler_a  = None
        self.iso_a     = None
        self.feat_a    = None
        self.medians_a = {}

        self.xgb_b     = None
        self.lgb_b     = None
        self.iso_b     = None
        self.feat_b    = None
        self.medians_b = {}
        self.best_b    = "XGBoost"

        self.dc        = None
        self.elo       = None
        self.ref_model = None

        self.mega_df          = None
        self.global_draw_rate = 0.42

    @classmethod
    def load(
        cls,
        paths_file: str = "models/v4/v4_paths.json",
        mega_path:  str = "data/processed/mega_dataset_v2.parquet",
    ) -> "V4Predictor":
        p = cls()

        if not Path(paths_file).exists():
            raise FileNotFoundError(
                f"{paths_file} not found. Run 'python src/train_v4.py' first."
            )

        with open(paths_file) as f:
            paths = json.load(f)

        # Model A
        with open(paths["model_a_lr"],         "rb") as f: p.lr_a      = pickle.load(f)
        with open(paths["model_a_scaler"],      "rb") as f: p.scaler_a  = pickle.load(f)
        with open(paths["model_a_calibrator"],  "rb") as f: p.iso_a     = pickle.load(f)
        with open(paths["model_a_features"])         as f: p.feat_a    = json.load(f)
        with open(paths["model_a_medians"])          as f: p.medians_a = json.load(f)

        # Model B
        import xgboost as xgb_lib
        import lightgbm as lgb_lib

        xgb_path = paths.get("model_b_xgb", "")
        lgb_path = paths.get("model_b_lgb", "")
        if xgb_path and Path(xgb_path).exists():
            p.xgb_b = xgb_lib.Booster()
            p.xgb_b.load_model(xgb_path)
        if lgb_path and Path(lgb_path).exists():
            p.lgb_b = lgb_lib.Booster(model_file=lgb_path)

        with open(paths["model_b_calibrator"],  "rb") as f: p.iso_b     = pickle.load(f)
        with open(paths["model_b_features"])         as f: p.feat_b    = json.load(f)
        with open(paths["model_b_medians"])          as f: p.medians_b = json.load(f)
        p.best_b = paths.get("model_b_best", "XGBoost")

        # Sub-models
        from src.dixon_coles import DixonColesEnsemble
        from src.elo import EloRatingSystem

        dc_path  = paths.get("dc_path",      "models/v4/dixon_coles.pkl")
        elo_path = paths.get("elo_path",     "models/v4/elo.pkl")
        ref_path = paths.get("referee_path", "models/v4/referee_model.pkl")

        if Path(dc_path).exists():
            p.dc = DixonColesEnsemble.load(dc_path)
        if Path(elo_path).exists():
            p.elo = EloRatingSystem.load(elo_path)
        if Path(ref_path).exists():
            with open(ref_path, "rb") as f:
                p.ref_model = pickle.load(f)

        # Mega dataset for form lookup
        mdp = paths.get("mega_dataset", mega_path)
        if Path(mdp).exists():
            p.mega_df = pd.read_parquet(mdp)
            p.mega_df["Date"] = pd.to_datetime(p.mega_df["Date"])
            p.mega_df = p.mega_df.sort_values("Date").reset_index(drop=True)

        return p

    # ── Form lookup ───────────────────────────────────────────────────────────

    def _get_team_form(self, team_name: str, as_home: bool) -> Dict[str, float]:
        if self.mega_df is None:
            return {}

        from src.utils import resolve_team_name
        all_teams = list(set(self.mega_df["HomeTeam"].tolist() + self.mega_df["AwayTeam"].tolist()))
        resolved  = resolve_team_name(team_name, all_teams)
        if not resolved:
            return {}

        if as_home:
            rows = self.mega_df[self.mega_df["HomeTeam"] == resolved].tail(5)
        else:
            rows = self.mega_df[self.mega_df["AwayTeam"] == resolved].tail(5)

        if len(rows) == 0:
            return {}

        latest = rows.iloc[-1]
        result = {}
        prefix = "home_" if as_home else "away_"
        other  = "away_" if as_home else "home_"

        for col in self.feat_b:
            if col.startswith(prefix) or col.startswith(other) or col in [
                "league_encoded", "country_encoded", "league_ht_draw_rate_historical",
            ]:
                if col in latest.index and pd.notna(latest[col]):
                    result[col] = float(latest[col])

        if "Date" in latest.index:
            try:
                days = (pd.Timestamp.today().normalize() - pd.to_datetime(latest["Date"])).days
                result[f"{prefix}days_since_last"] = float(max(0, days))
            except Exception:
                pass

        return result

    # ── Core predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        b365h:     float,
        b365d:     float,
        b365a:     float,
        league:    str           = "",
        referee:   Optional[str] = None,
    ) -> Dict:
        import xgboost as xgb_lib

        # ── Model A: log-odds → LR ────────────────────────────────────────────
        log_h = np.log(b365h) if b365h > 1 else 0.0
        log_d = np.log(b365d) if b365d > 1 else 0.0
        log_a = np.log(b365a) if b365a > 1 else 0.0

        row_a = {
            "log_home_win_odds": log_h,
            "log_draw_odds":     log_d,
            "log_away_win_odds": log_a,
        }
        xa     = np.array([[row_a.get(f, self.medians_a.get(f, 0.0)) for f in self.feat_a]], dtype=np.float32)
        xa_s   = self.scaler_a.transform(xa)
        pa_raw = self.lr_a.predict_proba(xa_s)[0, 1]
        pa_cal = float(np.clip(self.iso_a.predict([pa_raw])[0], 0.01, 0.99))

        # Market implied (direct, for reference)
        total          = (1/b365h) + (1/b365d) + (1/b365a)
        market_implied = float((1/b365d) / total) if total > 0 else self.global_draw_rate

        # ── Model B: fundamentals ─────────────────────────────────────────────
        row_b: Dict[str, float] = {}
        home_form = self._get_team_form(home_team, as_home=True)
        away_form = self._get_team_form(away_team, as_home=False)
        row_b.update(home_form)
        row_b.update(away_form)

        match_row = pd.DataFrame([{
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "league":   league,
            "Date":     pd.Timestamp.today(),
            "B365H": b365h, "B365D": b365d, "B365A": b365a,
        }])
        if referee:
            match_row["Referee"] = referee

        dc_prob  = self.global_draw_rate
        elo_prob = self.global_draw_rate
        ref_adj  = 1.0

        if self.dc is not None:
            try:
                dc_prob = float(np.clip(self.dc.predict_draw(match_row)[0], 0.01, 0.99))
            except Exception:
                pass

        if self.elo is not None:
            try:
                elo_prob = float(np.clip(self.elo.predict_draw(match_row)[0], 0.01, 0.99))
            except Exception:
                pass

        if self.ref_model is not None and referee:
            try:
                ref_adj = float(np.clip(self.ref_model.predict_draw_adjustment(match_row)[0], 0.5, 2.0))
            except Exception:
                pass

        row_b["dc_draw_prob"]  = dc_prob
        row_b["elo_draw_prob"] = elo_prob
        row_b["referee_adj"]   = ref_adj

        xb = np.array([[row_b.get(f, self.medians_b.get(f, 0.0)) for f in self.feat_b]], dtype=np.float32)

        if self.best_b == "LightGBM" and self.lgb_b is not None:
            pb_raw = float(self.lgb_b.predict(xb)[0])
        elif self.xgb_b is not None:
            dmat   = xgb_lib.DMatrix(xb, feature_names=self.feat_b)
            pb_raw = float(self.xgb_b.predict(dmat)[0])
        else:
            pb_raw = self.global_draw_rate

        pb_cal = float(np.clip(self.iso_b.predict([pb_raw])[0], 0.01, 0.99))

        # ── Inverted Edge ─────────────────────────────────────────────────────
        inverted_edge = pa_cal - pb_cal   # positive = value bet

        # ── Bet rating ────────────────────────────────────────────────────────
        if inverted_edge >= self.STRONG_VALUE_EDGE:
            rating      = "STRONG VALUE"
            rating_icon = "★★★"
            p_hit       = _KELLY_HIT_RATES["strong"]
        elif inverted_edge >= self.VALUE_EDGE:
            rating      = "VALUE"
            rating_icon = "★★"
            p_hit       = _KELLY_HIT_RATES["value"]
        elif inverted_edge >= self.MARGINAL_EDGE:
            rating      = "MARGINAL"
            rating_icon = "★"
            p_hit       = _KELLY_HIT_RATES["marginal"]
        else:
            rating      = "PASS"
            rating_icon = "—"
            p_hit       = 0.0

        # ── Kelly sizing (calibrated) ─────────────────────────────────────────
        # Uses backtest-derived hit rate per edge bucket (not Model B's raw prob).
        # This corrects the V3 bug where Model B probability was used directly,
        # underestimating the true edge since actual hit rates exceed Model B.
        if p_hit > 0 and inverted_edge > 0:
            kelly_f  = _kelly_fraction(p_hit, b365d)
            kelly_25 = round(kelly_f * 0.25, 4)
            kelly_10 = round(kelly_f * 0.10, 4)
        else:
            kelly_f  = 0.0
            kelly_25 = 0.0
            kelly_10 = 0.0

        return {
            "home_team":      home_team,
            "away_team":      away_team,
            "league":         league,
            "model_a_prob":   round(pa_cal, 4),
            "market_implied": round(market_implied, 4),
            "model_b_prob":   round(pb_cal, 4),
            "dc_draw_prob":   round(dc_prob, 4),
            "elo_draw_prob":  round(elo_prob, 4),
            "referee_adj":    round(ref_adj, 4),
            "inverted_edge":  round(inverted_edge, 4),
            "edge_pct":       round(inverted_edge * 100, 2),
            "rating":         rating,
            "rating_icon":    rating_icon,
            "p_hit":          round(p_hit, 4),
            "kelly_full":     round(kelly_f, 4),
            "kelly_25pct":    kelly_25,
            "kelly_10pct":    kelly_10,
            "b365h":          b365h,
            "b365d":          b365d,
            "b365a":          b365a,
        }

    @staticmethod
    def print_result(r: Dict, wide: bool = True) -> None:
        w   = 60 if wide else 50
        bar = "═" * w

        if r["inverted_edge"] >= 0.05:   edge_str = f"  ← STRONG VALUE {r['rating_icon']}"
        elif r["inverted_edge"] >= 0.03: edge_str = f"  ← VALUE {r['rating_icon']}"
        elif r["inverted_edge"] >= 0.01: edge_str = f"  ← MARGINAL {r['rating_icon']}"
        else:                            edge_str = "  ← PASS"

        print(f"\n{bar}")
        print(f"  {r['home_team']} vs {r['away_team']}"
              + (f"  [{r['league']}]" if r["league"] else ""))
        print(bar)
        print(f"  ODDS: Home {r['b365h']}  |  Draw {r['b365d']}  |  Away {r['b365a']}")
        print(f"  B365 implied FT draw prob: {r['market_implied']:.1%}  (FT odds proxy)")
        print()
        print(f"  ┌─ MODEL A — Market Estimate ───────────────────────┐")
        print(f"  │  P(HT draw) = {r['model_a_prob']:.1%}  (logistic on B365 log-odds)    │")
        print(f"  └────────────────────────────────────────────────────┘")
        print()
        print(f"  ┌─ MODEL B — Fundamentals Estimate ─────────────────┐")
        print(f"  │  P(HT draw) = {r['model_b_prob']:.1%}  (team stats only — NO odds)    │")
        print(f"  │                                                    │")
        print(f"  │  Components:                                       │")
        print(f"  │    Dixon-Coles:  {r['dc_draw_prob']:.1%}  (Poisson attack/defence)    │")
        print(f"  │    Elo ratings:  {r['elo_draw_prob']:.1%}  (rolling team strength)    │")
        print(f"  │    Referee adj: ×{r['referee_adj']:.3f}  (draw-rate multiplier)       │")
        print(f"  └────────────────────────────────────────────────────┘")
        print()

        edge_bar = ("█" * min(int(abs(r["inverted_edge"]) * 200), 30)).ljust(30)
        sign     = "+" if r["inverted_edge"] > 0 else ""
        print(f"  INVERTED EDGE (A − B): {sign}{r['edge_pct']:.2f}%{edge_str}")
        print(f"  [{edge_bar}]  Market {r['model_a_prob']:.1%} vs Fundamentals {r['model_b_prob']:.1%}")
        print(f"  (When market > fundamentals → actual HT draws historically exceed market)")
        print()
        print(f"  BET RATING: {r['rating']} {r['rating_icon']}")

        if r["inverted_edge"] > 0 and r["kelly_full"] > 0:
            p_hit_pct = r.get("p_hit", 0.0)
            print(f"\n  Kelly staking (draw @ {r['b365d']}, calibrated hit rate {p_hit_pct:.1%}):")
            print(f"    Full Kelly:     {r['kelly_full']:.2%} of bankroll")
            print(f"    Quarter Kelly:  {r['kelly_25pct']:.2%} of bankroll  ← recommended")
            print(f"    Tenth Kelly:    {r['kelly_10pct']:.2%} of bankroll  ← conservative")
        elif r["inverted_edge"] > 0:
            print(f"\n  Marginal positive edge — consider skipping or very small stake.")
        else:
            print(f"\n  No positive edge — skip.")

        print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V4 Half-Time Draw Predictor — Two-Model Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict_v4.py 'Everton' 'Man United' --odds 3.70/3.75/1.91
  python src/predict_v4.py 'Alaves' 'Girona' --odds 2.30/3.00/3.50 --league SP1
  python src/predict_v4.py 'LA Galaxy' 'NYCFC' --odds 2.25/3.60/2.80 --league USA_MLS
  python src/predict_v4.py 'Leeds' 'Sheffield Utd' --b365h 2.2 --b365d 3.3 --b365a 3.6
        """
    )
    parser.add_argument("home",     help="Home team name")
    parser.add_argument("away",     help="Away team name")

    # Convenience: single slash-separated odds string
    parser.add_argument("--odds",   default=None,
                        help="Odds as Home/Draw/Away (e.g. 3.70/3.75/1.91)")

    # Legacy individual flags
    parser.add_argument("--b365h",  type=float, default=None, help="B365 home win odds")
    parser.add_argument("--b365d",  type=float, default=None, help="B365 draw odds")
    parser.add_argument("--b365a",  type=float, default=None, help="B365 away win odds")

    parser.add_argument("--league",  default="",  help="League code (e.g. E0, SP1, USA_MLS)")
    parser.add_argument("--referee", default=None, help="Referee name (optional)")
    parser.add_argument("--paths",   default="models/v4/v4_paths.json",
                        help="Paths manifest file (default: models/v4/v4_paths.json)")
    args = parser.parse_args()

    # Parse odds
    if args.odds:
        parts = args.odds.split("/")
        if len(parts) != 3:
            parser.error("--odds must be in format H/D/A (e.g. 3.70/3.75/1.91)")
        try:
            b365h, b365d, b365a = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            parser.error("--odds values must be numeric")
    elif args.b365h and args.b365d and args.b365a:
        b365h, b365d, b365a = args.b365h, args.b365d, args.b365a
    else:
        parser.error("Provide odds via --odds H/D/A  OR  --b365h X --b365d X --b365a X")

    print("Loading V4 predictor...")
    predictor = V4Predictor.load(paths_file=args.paths)

    result = predictor.predict(
        home_team = args.home,
        away_team = args.away,
        b365h     = b365h,
        b365d     = b365d,
        b365a     = b365a,
        league    = args.league,
        referee   = args.referee,
    )
    V4Predictor.print_result(result)


if __name__ == "__main__":
    main()
