"""
V3 Two-Model Match Predictor — Half-Time Draw
==============================================
Shows BOTH market estimate and fundamentals estimate, plus the EDGE between them.

Usage:
    python src/predict_match_v3.py "Everton" "Man United" \\
        --b365h 3.70 --b365d 3.75 --b365a 1.91 --league E0

    python src/predict_match_v3.py "Alaves" "Girona" \\
        --b365h 2.30 --b365d 3.00 --b365a 3.50 --league SP1

    python src/predict_match_v3.py "LA Galaxy" "NYCFC" \\
        --b365h 2.25 --b365d 3.60 --b365a 2.80 --league USA_MLS

Output shows:
  - Model A: what the market implies about P(HT draw)
  - Model B: what match stats / team form / DC / Elo imply
  - EDGE = B - A: when positive, stats see more value than the market prices
  - Bet rating: STRONG VALUE / VALUE / MARGINAL / PASS
  - Kelly sizing based on edge
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

# ─────────────────────────────────────────────────────────────────────────────
# V3Predictor
# ─────────────────────────────────────────────────────────────────────────────

class V3Predictor:
    """
    Two-model predictor for P(HT draw).

    Model A — Market: log-odds LR (calibrated)
    Model B — Fundamentals: XGBoost or LightGBM on non-odds stats + DC + Elo (calibrated)
    Edge = Model B − Model A
    """

    STRONG_VALUE_EDGE = 0.05
    VALUE_EDGE        = 0.03
    MARGINAL_EDGE     = 0.01

    def __init__(self):
        # Model A
        self.lr_a       = None
        self.scaler_a   = None
        self.iso_a      = None
        self.feat_a     = None
        self.medians_a  = {}

        # Model B
        self.xgb_b      = None
        self.lgb_b      = None
        self.iso_b      = None
        self.feat_b     = None
        self.medians_b  = {}
        self.best_b     = "XGBoost"

        # Sub-models
        self.dc         = None
        self.elo        = None
        self.ref_model  = None

        # Mega dataset for form lookup
        self.mega_df    = None
        self.global_draw_rate = 0.42

    @classmethod
    def load(
        cls,
        paths_file: str = "models/v3/v3_paths.json",
        mega_path:  str = "data/processed/mega_dataset_v2.parquet",
    ) -> "V3Predictor":
        p = cls()

        if not Path(paths_file).exists():
            raise FileNotFoundError(
                f"{paths_file} not found. Run src/train_v3.py first."
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
        if Path(xgb_path).exists():
            p.xgb_b = xgb_lib.Booster()
            p.xgb_b.load_model(xgb_path)
        if Path(lgb_path).exists():
            p.lgb_b = lgb_lib.Booster(model_file=lgb_path)

        with open(paths["model_b_calibrator"],  "rb") as f: p.iso_b     = pickle.load(f)
        with open(paths["model_b_features"])         as f: p.feat_b    = json.load(f)
        with open(paths["model_b_medians"])          as f: p.medians_b = json.load(f)
        p.best_b = paths.get("model_b_best", "XGBoost")

        # Sub-models
        from src.dixon_coles import DixonColesEnsemble
        from src.elo import EloRatingSystem
        from src.referee_model import RefereeModel

        dc_path  = paths.get("dc_path",      "models/v3/dixon_coles_v3.pkl")
        elo_path = paths.get("elo_path",     "models/v3/elo_v3.pkl")
        ref_path = paths.get("referee_path", "models/v3/referee_model_v3.pkl")

        if Path(dc_path).exists():
            p.dc = DixonColesEnsemble.load(dc_path)
        if Path(elo_path).exists():
            p.elo = EloRatingSystem.load(elo_path)
        if Path(ref_path).exists():
            with open(ref_path, "rb") as f:
                p.ref_model = pickle.load(f)

        # Mega dataset
        mdp = paths.get("mega_dataset", mega_path)
        if Path(mdp).exists():
            p.mega_df = pd.read_parquet(mdp)
            p.mega_df["Date"] = pd.to_datetime(p.mega_df["Date"])
            p.mega_df = p.mega_df.sort_values("Date").reset_index(drop=True)

        return p

    # ── Form lookup ───────────────────────────────────────────────────────────

    def _get_team_form(self, team_name: str, as_home: bool) -> Dict[str, float]:
        """
        Look up the most recent rolling-stat row for a team in the mega dataset.
        Returns a dict of feature_name → value for Model B stats.
        """
        if self.mega_df is None:
            return {}

        from src.utils import resolve_team_name
        resolved = resolve_team_name(team_name, list(
            set(self.mega_df["HomeTeam"].tolist() + self.mega_df["AwayTeam"].tolist())
        ))

        if as_home:
            rows = self.mega_df[self.mega_df["HomeTeam"] == resolved].tail(5)
        else:
            rows = self.mega_df[self.mega_df["AwayTeam"] == resolved].tail(5)

        if len(rows) == 0:
            return {}

        latest = rows.iloc[-1]
        result = {}

        # Pull relevant rolling stats
        prefix = "home_" if as_home else "away_"
        other  = "away_" if as_home else "home_"

        for col in self.feat_b:
            if col.startswith(prefix) or col.startswith(other) or col in [
                "league_encoded", "country_encoded", "league_ht_draw_rate_historical",
            ]:
                if col in latest.index and pd.notna(latest[col]):
                    result[col] = float(latest[col])

        # Also days_since_last
        if "Date" in latest.index:
            try:
                days = (pd.Timestamp("2026-02-22") - pd.to_datetime(latest["Date"])).days
                result[f"{prefix}days_since_last"] = float(days)
            except Exception:
                pass

        return result

    # ── Core predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        home_team:  str,
        away_team:  str,
        b365h:      float,
        b365d:      float,
        b365a:      float,
        league:     str  = "",
        referee:    Optional[str] = None,
    ) -> Dict:
        """
        Predict P(HT draw) using both models and return full breakdown.
        """
        import xgboost as xgb_lib

        # ── Model A: log-odds features → LR ───────────────────────────────────
        log_h = np.log(b365h) if b365h > 1 else 0.0
        log_d = np.log(b365d) if b365d > 1 else 0.0
        log_a = np.log(b365a) if b365a > 1 else 0.0

        row_a = {
            "log_home_win_odds":  log_h,
            "log_draw_odds":      log_d,
            "log_away_win_odds":  log_a,
        }
        xa = np.array([[row_a.get(f, self.medians_a.get(f, 0.0)) for f in self.feat_a]],
                      dtype=np.float32)
        xa_s = self.scaler_a.transform(xa)
        pa_raw = self.lr_a.predict_proba(xa_s)[0, 1]
        pa_cal = float(np.clip(self.iso_a.predict([pa_raw])[0], 0.01, 0.99))

        # Market implied (direct, for reference)
        total = (1/b365h) + (1/b365d) + (1/b365a)
        market_implied = float((1/b365d) / total) if total > 0 else self.global_draw_rate

        # ── Model B: fundamentals ─────────────────────────────────────────────
        row_b: Dict[str, float] = {}

        # Base log-odds placeholders (set to 0; not allowed as features)
        # Get form stats from mega dataset
        home_form = self._get_team_form(home_team, as_home=True)
        away_form = self._get_team_form(away_team, as_home=False)
        row_b.update(home_form)
        row_b.update(away_form)

        # Add DC, Elo, Referee predictions
        match_row = pd.DataFrame([{
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "league": league,
            "Date": pd.Timestamp("2026-02-22"),
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

        # Build feature vector for Model B
        xb = np.array([[row_b.get(f, self.medians_b.get(f, 0.0)) for f in self.feat_b]],
                      dtype=np.float32)

        # Run best Model B
        if self.best_b == "LightGBM" and self.lgb_b is not None:
            pb_raw = float(self.lgb_b.predict(xb)[0])
        elif self.xgb_b is not None:
            dmat   = xgb_lib.DMatrix(xb, feature_names=self.feat_b)
            pb_raw = float(self.xgb_b.predict(dmat)[0])
        else:
            pb_raw = self.global_draw_rate

        pb_cal = float(np.clip(self.iso_b.predict([pb_raw])[0], 0.01, 0.99))

        # ── Edge ──────────────────────────────────────────────────────────────
        # INVERTED SIGNAL: backtesting showed that when fundamentals predict
        # FEWER draws than market, actual draw rate EXCEEDS market estimate.
        # So the exploitable edge = Model A - Model B (inverted).
        raw_edge = pb_cal - pa_cal
        inverted_edge = pa_cal - pb_cal  # positive = value bet on HT draw

        # ── Bet rating (using INVERTED edge) ──────────────────────────────────
        if inverted_edge >= self.STRONG_VALUE_EDGE:
            rating      = "STRONG VALUE"
            rating_icon = "★★★"
        elif inverted_edge >= self.VALUE_EDGE:
            rating      = "VALUE"
            rating_icon = "★★"
        elif inverted_edge >= self.MARGINAL_EDGE:
            rating      = "MARGINAL"
            rating_icon = "★"
        else:
            rating      = "PASS"
            rating_icon = "—"
        edge = inverted_edge

        # ── Kelly sizing ──────────────────────────────────────────────────────
        # Using Model B prob as our edge estimate, B365D as the odds
        b_odds = b365d - 1.0
        kelly_f = (pb_cal * b_odds - (1.0 - pb_cal)) / b_odds if b_odds > 0 else 0.0
        kelly_f = max(0.0, kelly_f)
        kelly_25 = round(kelly_f * 0.25, 4)  # Quarter Kelly
        kelly_10 = round(kelly_f * 0.10, 4)  # Tenth Kelly (conservative)

        return {
            "home_team":       home_team,
            "away_team":       away_team,
            "league":          league,
            # Model A
            "model_a_prob":    round(pa_cal, 4),
            "market_implied":  round(market_implied, 4),
            # Model B
            "model_b_prob":    round(pb_cal, 4),
            "dc_draw_prob":    round(dc_prob, 4),
            "elo_draw_prob":   round(elo_prob, 4),
            "referee_adj":     round(ref_adj, 4),
            # Edge
            "edge":            round(edge, 4),
            "edge_pct":        round(edge * 100, 2),
            "rating":          rating,
            "rating_icon":     rating_icon,
            # Kelly
            "kelly_full":      round(kelly_f, 4),
            "kelly_25pct":     kelly_25,
            "kelly_10pct":     kelly_10,
            # Odds
            "b365h":           b365h,
            "b365d":           b365d,
            "b365a":           b365a,
        }

    # ── Pretty print ─────────────────────────────────────────────────────────

    @staticmethod
    def print_result(r: Dict, wide: bool = True) -> None:
        w = 60 if wide else 50
        bar = "═" * w

        edge_color = ""
        if r["edge"] >= 0.05:    edge_str = f"  ← STRONG VALUE {r['rating_icon']}"
        elif r["edge"] >= 0.03:  edge_str = f"  ← VALUE {r['rating_icon']}"
        elif r["edge"] >= 0.01:  edge_str = f"  ← MARGINAL {r['rating_icon']}"
        else:                    edge_str = "  ← PASS"

        print(f"\n{bar}")
        print(f"  {r['home_team']} vs {r['away_team']}"
              + (f"  [{r['league']}]" if r["league"] else ""))
        print(bar)
        print(f"  ODDS: Home {r['b365h']}  |  Draw {r['b365d']}  |  Away {r['b365a']}")
        print(f"  Market implied P(HT draw): {r['market_implied']:.1%}")
        print()
        print(f"  ┌─ MODEL A — Market Estimate ───────────────────────┐")
        print(f"  │  P(HT draw) = {r['model_a_prob']:.1%}  "
              f"(logistic on B365 log-odds)               │")
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

        edge_bar_filled = int(abs(r["edge"]) * 200)
        edge_bar = ("█" * min(edge_bar_filled, 30)).ljust(30)
        sign = "+" if r["edge"] > 0 else ""
        print(f"  INVERTED EDGE (A − B): {sign}{r['edge_pct']:.2f}%{edge_str}")
        print(f"  [{edge_bar}]  Market {r['model_a_prob']:.1%} vs Fundamentals {r['model_b_prob']:.1%})")
        print(f"  (When fundamentals predict fewer draws than market → actual draws EXCEED market)")
        print()
        print(f"  BET RATING: {r['rating']} {r['rating_icon']}")

        if r["edge"] > 0:
            print(f"\n  Kelly staking (on draw @ {r['b365d']}):")
            print(f"    Full Kelly:     {r['kelly_full']:.2%} of bankroll")
            print(f"    Quarter Kelly:  {r['kelly_25pct']:.2%} of bankroll  ← recommended")
            print(f"    Tenth Kelly:    {r['kelly_10pct']:.2%} of bankroll  ← conservative")
        else:
            print(f"\n  No positive edge — skip or wait for better line.")

        print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V3 Two-Model HT Draw Predictor"
    )
    parser.add_argument("home",     help="Home team name")
    parser.add_argument("away",     help="Away team name")
    parser.add_argument("--b365h",  type=float, required=True, help="B365 home win odds")
    parser.add_argument("--b365d",  type=float, required=True, help="B365 draw odds")
    parser.add_argument("--b365a",  type=float, required=True, help="B365 away win odds")
    parser.add_argument("--league", default="",  help="League code (e.g. E0, SP1)")
    parser.add_argument("--referee", default=None, help="Referee name (optional)")
    parser.add_argument("--paths",  default="models/v3/v3_paths.json")
    args = parser.parse_args()

    print("Loading V3 predictor...")
    predictor = V3Predictor.load(paths_file=args.paths)

    result = predictor.predict(
        home_team = args.home,
        away_team = args.away,
        b365h     = args.b365h,
        b365d     = args.b365d,
        b365a     = args.b365a,
        league    = args.league,
        referee   = args.referee,
    )
    V3Predictor.print_result(result)


if __name__ == "__main__":
    main()
