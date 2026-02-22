"""
Rolling Elo Rating System — Half-Time Draw Prediction
======================================================
Updates team Elo ratings based on HALF-TIME results (not full-time).

- All teams initialised at 1500
- K-factor: tunable (default 32)
- Home advantage: 50 Elo points added to home expected score
- HT result mapping: draw=0.5, home win=1.0, away win=0.0
- P(draw) derived via logistic regression on Elo difference
- Snapshot system: preserves ratings at train/val/test boundaries

Usage:
    from src.elo import EloRatingSystem
    elo = EloRatingSystem(k=32, home_adv=50)
    elo.fit(train_df)                          # build rating history
    probs_val = elo.predict_draw(val_df)       # returns array of P(HT draw)
    elo.save("models/v2/elo.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ── Elo core ─────────────────────────────────────────────────────────────────

class EloRatingSystem:
    """
    Rolling Elo rating system based on half-time outcomes.

    Parameters
    ----------
    k : float
        K-factor (learning rate). Higher = faster adaptation.
    home_adv : float
        Virtual Elo points added to the home side's expected score.
    initial_rating : float
        Starting rating for all teams.
    """

    def __init__(self, k: float = 32, home_adv: float = 50,
                 initial_rating: float = 1500):
        self.k              = k
        self.home_adv       = home_adv
        self.initial_rating = initial_rating

        # Current ratings
        self.ratings_: Dict[str, float] = {}

        # History: team → [(date, rating)] sorted by date
        self.history_: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}

        # Meta-learner: maps (elo_diff, home_rating, away_rating) → P(draw)
        self.logit_: Optional[LogisticRegression] = None

        # Pre-computed predictions on training matches (for meta-learner training)
        self.train_preds_: Optional[np.ndarray] = None

        self.fitted_: bool = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_rating(self, team: str) -> float:
        return self.ratings_.get(team, self.initial_rating)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A vs team B (0–1 scale)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _ht_result_score(self, hthg: int, htag: int) -> Tuple[float, float]:
        """
        Map HT score to (home_score, away_score) for Elo update.
        Draw = (0.5, 0.5), home win = (1, 0), away win = (0, 1).
        """
        if hthg > htag:
            return 1.0, 0.0
        if hthg < htag:
            return 0.0, 1.0
        return 0.5, 0.5

    def _result_score(self, hg: int, ag: int) -> Tuple[float, float]:
        """Generic result → (home_score, away_score). Same logic as HT."""
        return self._ht_result_score(hg, ag)

    def _update(self, home_team: str, away_team: str,
                hthg: int, htag: int, date: pd.Timestamp) -> Tuple[float, float]:
        """
        Update ratings for one match.
        Returns (pre-match elo_diff, P(draw) from Elo) before update.
        """
        r_home = self._get_rating(home_team)
        r_away = self._get_rating(away_team)

        # Effective home rating (home advantage)
        e_home = self._expected(r_home + self.home_adv, r_away)
        e_away = 1.0 - e_home

        s_home, s_away = self._ht_result_score(hthg, htag)

        new_home = r_home + self.k * (s_home - e_home)
        new_away = r_away + self.k * (s_away - e_away)

        # Store pre-update values for history
        elo_diff = r_home + self.home_adv - r_away

        # Update
        self.ratings_[home_team] = new_home
        self.ratings_[away_team] = new_away

        # Record history
        for team, rating in [(home_team, new_home), (away_team, new_away)]:
            if team not in self.history_:
                self.history_[team] = []
            self.history_[team].append((date, rating))

        return elo_diff, e_home  # e_home = P(home win adjusted for home adv)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "EloRatingSystem":
        """
        Process all training matches chronologically.
        Builds a logistic regression to map Elo features → P(HT draw).

        df must have: HomeTeam, AwayTeam, HTHG, HTAG, Date.
        Rows where HTHG/HTAG are NaN but FTHG/FTAG are available will use
        FT result as a proxy for rating updates (but are NOT used to calibrate
        the logit, since we have no HT outcome to train against).
        """
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"]).copy()
        df = df.sort_values("Date").reset_index(drop=True)

        # Separate: rows with HT data (used for logit calibration)
        has_ht = df["HTHG"].notna() & df["HTAG"].notna()

        elo_diffs  = []
        home_probs = []
        y_draw     = []

        self.ratings_ = {}
        self.history_ = {}

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]

            r_home = self._get_rating(home)
            r_away = self._get_rating(away)
            diff   = r_home + self.home_adv - r_away

            ht_available = pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG"))
            ft_available = pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG"))

            if ht_available:
                hthg = int(row["HTHG"])
                htag = int(row["HTAG"])
                # Collect for logit calibration
                elo_diffs.append(diff)
                home_probs.append(self._expected(r_home + self.home_adv, r_away))
                y_draw.append(1 if hthg == htag else 0)
                self._update(home, away, hthg, htag, date)
            elif ft_available:
                # FT-only leagues (e.g. MLS): update ratings using FT result as proxy
                # These rows do NOT feed into logit calibration
                fthg = int(row["FTHG"])
                ftag = int(row["FTAG"])
                self._update(home, away, fthg, ftag, date)
            # else: skip (no result available)

        elo_diffs  = np.array(elo_diffs)
        home_probs = np.array(home_probs)
        y_draw     = np.array(y_draw)

        # Fit logistic regression: features = [elo_diff, |elo_diff|, home_prob]
        X_train = self._make_features(elo_diffs, home_probs)
        self.logit_ = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        self.logit_.fit(X_train, y_draw)

        train_preds = self.logit_.predict_proba(X_train)[:, 1]
        self.train_preds_ = train_preds

        train_auc = roc_auc_score(y_draw, train_preds) if len(set(y_draw)) > 1 else 0.5
        n_ht = int(has_ht.sum())
        n_ft = len(df) - n_ht
        print(f"    Elo fitted on {n_ht:,} HT matches + {n_ft:,} FT-proxy matches  |  "
              f"K={self.k}  home_adv={self.home_adv}  |  "
              f"train logit AUC={train_auc:.4f}")

        self.fitted_ = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def _make_features(self, elo_diffs: np.ndarray,
                       home_probs: np.ndarray) -> np.ndarray:
        """Feature matrix for logistic regression."""
        return np.column_stack([
            elo_diffs,
            np.abs(elo_diffs),
            elo_diffs ** 2,
            home_probs,
        ])

    def _get_pre_match_ratings(self, home: str, away: str) -> Tuple[float, float]:
        """Get current ratings (pre-match call)."""
        return self._get_rating(home), self._get_rating(away)

    def predict_draw(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict P(HT draw) for each row in df.
        Uses CURRENT ratings (ratings at time of predict call) — works for
        chronological validation/test sets where fit() has already processed
        up to the boundary.

        df must have: HomeTeam, AwayTeam.
        Applies fuzzy team-name matching when exact lookup fails.
        """
        if not self.fitted_:
            return np.full(len(df), 0.42)

        from src.utils import resolve_team_name
        known = list(self.ratings_.keys())

        elo_diffs  = []
        home_probs = []

        for _, row in df.iterrows():
            home = row.get("HomeTeam", "")
            away = row.get("AwayTeam", "")
            h = home if home in self.ratings_ else (resolve_team_name(home, known) or home)
            a = away if away in self.ratings_ else (resolve_team_name(away, known) or away)
            r_h, r_a = self._get_pre_match_ratings(h, a)
            diff = r_h + self.home_adv - r_a
            elo_diffs.append(diff)
            home_probs.append(self._expected(r_h + self.home_adv, r_a))

        X = self._make_features(np.array(elo_diffs), np.array(home_probs))
        return self.logit_.predict_proba(X)[:, 1]

    def predict_draw_single(self, home_team: str, away_team: str) -> float:
        """
        Predict P(HT draw) for a single match using current ratings.
        Falls back to fuzzy name matching when exact team lookup fails.
        """
        if not self.fitted_:
            return 0.42
        from src.utils import resolve_team_name
        known = list(self.ratings_.keys())
        h = home_team if home_team in self.ratings_ else (resolve_team_name(home_team, known) or home_team)
        a = away_team if away_team in self.ratings_ else (resolve_team_name(away_team, known) or away_team)
        r_h, r_a = self._get_pre_match_ratings(h, a)
        diff     = r_h + self.home_adv - r_a
        hp       = self._expected(r_h + self.home_adv, r_a)
        X = self._make_features(np.array([diff]), np.array([hp]))
        return float(self.logit_.predict_proba(X)[0, 1])

    def replay_and_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Walk through df chronologically, making predictions BEFORE updating.
        Useful for getting proper temporal predictions on train+val+test data.

        df must have: HomeTeam, AwayTeam, Date.
        HTHG/HTAG used when available; falls back to FTHG/FTAG for FT-only rows.
        """
        df = df.sort_values("Date").reset_index(drop=True)
        elo_diffs  = []
        home_probs = []

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]

            r_h, r_a = self._get_pre_match_ratings(home, away)
            diff = r_h + self.home_adv - r_a
            elo_diffs.append(diff)
            home_probs.append(self._expected(r_h + self.home_adv, r_a))

            # Update using HT score if available, else FT proxy
            ht_ok = pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG"))
            ft_ok = pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG"))
            if ht_ok:
                self._update(home, away, int(row["HTHG"]), int(row["HTAG"]), date)
            elif ft_ok:
                self._update(home, away, int(row["FTHG"]), int(row["FTAG"]), date)

        X = self._make_features(np.array(elo_diffs), np.array(home_probs))
        return self.logit_.predict_proba(X)[:, 1]

    def extend_ratings(self, df: pd.DataFrame) -> "EloRatingSystem":
        """
        Replay additional matches to update team ratings WITHOUT retraining
        the logistic meta-learner.  Use this to fold in FT-only leagues
        (e.g. MLS) after the model has already been fitted on European data.

        df must have: HomeTeam, AwayTeam, Date.
        Uses HTHG/HTAG when available; falls back to FTHG/FTAG for FT-only rows.
        New teams are initialised at self.initial_rating automatically.
        """
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"]).copy()
        df = df.sort_values("Date").reset_index(drop=True)

        n_new, n_updated = 0, 0
        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]

            ht_ok = pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG"))
            ft_ok = pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG"))

            if ht_ok:
                was_new = home not in self.ratings_ or away not in self.ratings_
                self._update(home, away, int(row["HTHG"]), int(row["HTAG"]), date)
            elif ft_ok:
                was_new = home not in self.ratings_ or away not in self.ratings_
                self._update(home, away, int(row["FTHG"]), int(row["FTAG"]), date)
            else:
                continue

            if was_new:
                n_new += 1
            else:
                n_updated += 1

        print(f"    Elo.extend_ratings: processed {len(df):,} matches  "
              f"(new teams: {n_new}, updated: {n_updated}, "
              f"total teams: {len(self.ratings_):,})")
        return self

    def get_top_rated(self, n: int = 20, league_filter: Optional[str] = None) -> pd.DataFrame:
        """Return top-rated teams sorted by current rating."""
        rows = [{"team": t, "elo": r} for t, r in self.ratings_.items()]
        return pd.DataFrame(rows).sort_values("elo", ascending=False).head(n)

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"    Elo model saved → {path}")

    @staticmethod
    def load(path: str) -> "EloRatingSystem":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── K-factor tuning ───────────────────────────────────────────────────────────

def tune_k_factor(
    df: pd.DataFrame,
    k_values: List[float] = [16, 20, 32, 40, 48],
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
) -> Dict[str, float]:
    """
    Grid-search K-factor on validation set. Returns best K and its AUC.
    """
    df = df.dropna(subset=["HTHG", "HTAG"]).sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(train_frac * n)
    val_end   = train_end + int(val_frac * n)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()

    results = {}
    best_k, best_auc = 32, 0.0

    for k in k_values:
        elo = EloRatingSystem(k=k)
        elo.fit(train_df)
        preds = elo.predict_draw(val_df)
        y_val = (val_df["HTHG"] == val_df["HTAG"]).astype(int).values
        auc = roc_auc_score(y_val, preds) if len(set(y_val)) > 1 else 0.5
        results[k] = auc
        print(f"    K={k:3.0f} → val AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_k   = k

    print(f"    Best K = {best_k}  (AUC = {best_auc:.4f})")
    return {"best_k": best_k, "best_auc": best_auc, "all_k_auc": results}


# ── Standalone script ─────────────────────────────────────────────────────────

def fit_from_parquet(
    parquet_path: str = "data/processed/mega_dataset.parquet",
    train_frac: float = 0.70,
    save_path: str = "models/v2/elo.pkl",
) -> EloRatingSystem:
    """Fit Elo on train split, save, return."""
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["HTHG", "HTAG"]).sort_values("Date").reset_index(drop=True)

    n = len(df)
    train_df = df.iloc[:int(train_frac * n)].copy()

    print("  Tuning K-factor...")
    tune_res = tune_k_factor(train_df)

    elo = EloRatingSystem(k=tune_res["best_k"])
    elo.fit(train_df)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    elo.save(save_path)
    return elo


if __name__ == "__main__":
    elo = fit_from_parquet()
    print(f"\nTop 10 rated teams:")
    print(elo.get_top_rated(10).to_string(index=False))

    p = elo.predict_draw_single("Leeds United", "Sheffield Utd")
    print(f"\nLeeds United vs Sheffield Utd: P(HT draw) = {p:.3f}")
