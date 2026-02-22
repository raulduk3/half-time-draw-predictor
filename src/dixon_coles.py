"""
Dixon-Coles Bivariate Poisson Model — Half-Time Draw Prediction
================================================================
Fits team attack/defense parameters on HALF-TIME scores (HTHG/HTAG).

- rho correction for low-scoring draws (0-0, 1-1 excess)
- Time-decay weighting (recent matches weighted higher)
- Per-league parameter estimation (each league independent)
- scipy.optimize.minimize (L-BFGS-B) for log-likelihood maximisation
- Output: P(HT draw) = sum_k P(HTHG=k, HTAG=k) for k=0..4

Usage:
    from src.dixon_coles import DixonColes, DixonColesEnsemble
    dc = DixonColesEnsemble()
    dc.fit(train_df)
    probs = dc.predict_draw(val_df)   # shape (n,)
    dc.save("models/v2/dixon_coles.pkl")
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ── Low-score correction (tau) ───────────────────────────────────────────────

def tau(h: float, a: float, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for scorelines 0-0, 1-0, 0-1, 1-1."""
    if h == 0 and a == 0:
        return max(1e-10, 1.0 - lam * mu * rho)
    if h == 0 and a == 1:
        return max(1e-10, 1.0 + lam * rho)
    if h == 1 and a == 0:
        return max(1e-10, 1.0 + mu * rho)
    if h == 1 and a == 1:
        return max(1e-10, 1.0 - rho)
    return 1.0


def tau_vec(h: np.ndarray, a: np.ndarray,
            lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Vectorized tau for arrays of (h, a, lam, mu)."""
    t = np.ones(len(h))
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    t[m00] = np.maximum(1e-10, 1.0 - lam[m00] * mu[m00] * rho)
    t[m01] = np.maximum(1e-10, 1.0 + lam[m01] * rho)
    t[m10] = np.maximum(1e-10, 1.0 + mu[m10] * rho)
    t[m11] = np.maximum(1e-10, 1.0 - rho)
    return t


# ── Per-league Dixon-Coles model ─────────────────────────────────────────────

class DixonColes:
    """
    Fits Dixon-Coles bivariate Poisson model for one league.

    Parameters
    ----------
    xi : float
        Time-decay parameter. Weight for a match d days ago = exp(-xi * d).
        xi=0 → uniform weights. xi=0.003 ≈ half-life of ~231 days.
    max_goals : int
        Max goals considered per team in P(draw) summation (default 5).
    """

    def __init__(self, xi: float = 0.003, max_goals: int = 5):
        self.xi = xi
        self.max_goals = max_goals
        self.params_: Optional[np.ndarray] = None
        self.teams_: List[str] = []
        self.team_idx_: Dict[str, int] = {}
        self.n_teams_: int = 0
        self.rho_: float = 0.0
        self.home_adv_: float = 0.0
        self.fitted_: bool = False
        self.league_: str = ""
        self.train_draw_rate_: float = 0.42
        # Scale factor applied to lambda/mu at prediction time.
        # Set to 0.45 for FT-only leagues (e.g. MLS) so that Poisson means
        # are calibrated to half-time scoring rates (HT goals ≈ 0.45 × FT goals).
        self.ht_scale_: float = 1.0

    def _decode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Decode parameter vector → (attack, defense, home_adv, rho)."""
        n = self.n_teams_
        attack  = x[:n]
        defense = x[n: 2 * n]
        home_adv = x[2 * n]
        rho      = x[2 * n + 1]
        return attack, defense, home_adv, rho

    def _neg_log_likelihood(self, x: np.ndarray,
                             h_arr: np.ndarray, a_arr: np.ndarray,
                             hi_arr: np.ndarray, ai_arr: np.ndarray,
                             weights: np.ndarray) -> float:
        attack, defense, home_adv, rho = self._decode(x)

        lam = np.exp(attack[hi_arr] + defense[ai_arr] + home_adv)
        mu  = np.exp(attack[ai_arr] + defense[hi_arr])

        t = tau_vec(h_arr, a_arr, lam, mu, rho)
        ll = (
            np.log(np.maximum(t, 1e-10))
            + poisson.logpmf(h_arr, lam)
            + poisson.logpmf(a_arr, mu)
        )
        return -np.sum(weights * ll)

    def fit(self, df: pd.DataFrame, use_ft_fallback: bool = False) -> "DixonColes":
        """
        Fit on a DataFrame with columns: HomeTeam, AwayTeam, HTHG, HTAG, Date.
        Temporal ordering: uses match Date to compute time-decay weights.

        If use_ft_fallback=True and HTHG/HTAG are missing, uses FTHG/FTAG as
        a proxy (for FT-only leagues like MLS). The model will capture team
        attack/defence strengths but draw probabilities will reflect FT rates.
        """
        if use_ft_fallback:
            # For FT-only leagues: use FTHG/FTAG where HTHG/HTAG unavailable
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"]).copy()
            has_ht = df["HTHG"].notna() & df["HTAG"].notna()
            has_ft = df["FTHG"].notna() & df["FTAG"].notna()
            df.loc[~has_ht & has_ft, "HTHG"] = df.loc[~has_ht & has_ft, "FTHG"]
            df.loc[~has_ht & has_ft, "HTAG"] = df.loc[~has_ht & has_ft, "FTAG"]

        df = df.dropna(subset=["HomeTeam", "AwayTeam", "HTHG", "HTAG", "Date"]).copy()
        df = df.sort_values("Date").reset_index(drop=True)

        # Collect teams
        all_teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.teams_ = all_teams
        self.team_idx_ = {t: i for i, t in enumerate(all_teams)}
        self.n_teams_ = len(all_teams)
        self.train_draw_rate_ = (df["HTHG"] == df["HTAG"]).mean()

        # Encode
        h_arr  = df["HTHG"].values.astype(int)
        a_arr  = df["HTAG"].values.astype(int)
        hi_arr = np.array([self.team_idx_[t] for t in df["HomeTeam"]])
        ai_arr = np.array([self.team_idx_[t] for t in df["AwayTeam"]])

        # Time-decay weights
        ref_date = df["Date"].max()
        days_ago = (ref_date - df["Date"]).dt.days.values.astype(float)
        weights  = np.exp(-self.xi * days_ago)
        weights /= weights.sum()

        # Initial parameters: log(1) = 0 for all, slight home advantage
        n = self.n_teams_
        x0 = np.zeros(2 * n + 2)
        x0[2 * n]   = 0.1   # home_adv
        x0[2 * n + 1] = -0.1  # rho

        # Bounds: rho in (-1, 1), rest unbounded
        bounds = [(None, None)] * (2 * n)
        bounds += [(None, None)]  # home_adv
        bounds += [(-0.99, 0.99)]  # rho

        # Constraint: sum of attack params = 0 (identifiability)
        # We enforce this via a penalised objective instead of hard constraint
        def obj(x):
            nll = self._neg_log_likelihood(x, h_arr, a_arr, hi_arr, ai_arr, weights)
            # Soft constraint on sum of attack params
            penalty = 1e3 * (x[:n].sum() ** 2)
            return nll + penalty

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                obj, x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
            )

        self.params_   = result.x
        attack, defense, home_adv, rho = self._decode(result.x)
        self.home_adv_ = home_adv
        self.rho_      = rho
        self.fitted_   = True
        return self

    def predict_draw_proba(
        self,
        home_team: str,
        away_team: str,
        max_goals: Optional[int] = None,
    ) -> float:
        """
        P(HTHG == HTAG) = sum_{k=0}^{max_goals} P(HTHG=k, HTAG=k).
        Returns league average draw rate if teams not seen in training.
        """
        if not self.fitted_:
            return self.train_draw_rate_

        if home_team not in self.team_idx_ or away_team not in self.team_idx_:
            return self.train_draw_rate_

        attack, defense, home_adv, rho = self._decode(self.params_)
        hi = self.team_idx_[home_team]
        ai = self.team_idx_[away_team]

        scale = getattr(self, "ht_scale_", 1.0)
        lam = float(np.exp(attack[hi] + defense[ai] + home_adv)) * scale
        mu  = float(np.exp(attack[ai] + defense[hi])) * scale
        mg  = max_goals or self.max_goals

        p_draw = 0.0
        for k in range(mg + 1):
            t   = tau(k, k, lam, mu, rho)
            p_k = t * poisson.pmf(k, lam) * poisson.pmf(k, mu)
            p_draw += p_k

        return float(np.clip(p_draw, 0.01, 0.99))

    def predict_score_matrix(self, home_team: str, away_team: str,
                             max_goals: int = 6) -> np.ndarray:
        """Return (max_goals+1) × (max_goals+1) score probability matrix."""
        if not self.fitted_ or home_team not in self.team_idx_ or away_team not in self.team_idx_:
            return np.full((max_goals + 1, max_goals + 1), 1.0 / (max_goals + 1) ** 2)

        attack, defense, home_adv, rho = self._decode(self.params_)
        hi = self.team_idx_[home_team]
        ai = self.team_idx_[away_team]
        scale = getattr(self, "ht_scale_", 1.0)
        lam = float(np.exp(attack[hi] + defense[ai] + home_adv)) * scale
        mu  = float(np.exp(attack[ai] + defense[hi])) * scale

        mat = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                t = tau(h, a, lam, mu, rho)
                mat[h, a] = t * poisson.pmf(h, lam) * poisson.pmf(a, mu)
        return mat / mat.sum()

    def get_team_strengths(self) -> pd.DataFrame:
        """Return DataFrame with team attack/defense ratings."""
        if not self.fitted_:
            return pd.DataFrame()
        attack, defense, _, _ = self._decode(self.params_)
        return pd.DataFrame({
            "team":    self.teams_,
            "attack":  attack,
            "defense": defense,
        }).sort_values("attack", ascending=False)


# ── Multi-league ensemble ─────────────────────────────────────────────────────

class DixonColesEnsemble:
    """
    Fits one DixonColes model per league, then provides a unified predict interface.
    Falls back to global draw rate for unrecognised team/league pairs.
    """

    def __init__(self, xi: float = 0.003, max_goals: int = 5):
        self.xi = xi
        self.max_goals = max_goals
        self.league_models_: Dict[str, DixonColes] = {}
        self.global_draw_rate_: float = 0.42

    def fit(self, df: pd.DataFrame, min_matches: int = 50,
            ft_only_leagues: Optional[List[str]] = None) -> "DixonColesEnsemble":
        """
        Fit per-league models.

        Parameters
        ----------
        df : DataFrame with HomeTeam, AwayTeam, HTHG, HTAG, Date, league.
             Rows without HTHG/HTAG are automatically handled via FT fallback
             for leagues listed in ft_only_leagues.
        min_matches : leagues with fewer matches are skipped.
        ft_only_leagues : list of league codes that have no HT data; these use
                          FTHG/FTAG as a proxy for goal scoring patterns.
        """
        # For global draw rate, use only rows with real HT data
        ht_df = df.dropna(subset=["HomeTeam", "AwayTeam", "HTHG", "HTAG", "Date"]).copy()
        self.global_draw_rate_ = (ht_df["HTHG"] == ht_df["HTAG"]).mean()

        ft_only = set(ft_only_leagues or [])
        leagues = df["league"].unique() if "league" in df.columns else ["default"]

        fitted, skipped = 0, 0
        for league in leagues:
            sub = df[df["league"] == league] if "league" in df.columns else df
            is_ft_only = league in ft_only

            # For FT-only leagues, count valid rows differently
            if is_ft_only:
                valid_sub = sub[sub["FTHG"].notna() & sub["FTAG"].notna()]
            else:
                valid_sub = sub[sub["HTHG"].notna() & sub["HTAG"].notna()]

            if len(valid_sub) < min_matches:
                skipped += 1
                continue

            tag = " [FT-proxy]" if is_ft_only else ""
            print(f"    Fitting Dixon-Coles: {league:30s}  n={len(valid_sub):6,}{tag}", end="")
            try:
                model = DixonColes(xi=self.xi, max_goals=self.max_goals)
                model.fit(sub, use_ft_fallback=is_ft_only)
                model.league_ = league
                # For FT-only leagues, scale Poisson means to HT-equivalent rates
                # (HT goals ≈ 0.45 × FT goals).
                if is_ft_only:
                    model.ht_scale_ = 0.45
                self.league_models_[league] = model
                ht_note = f"  ht_scale={model.ht_scale_:.2f}" if is_ft_only else ""
                print(f"  rho={model.rho_:.3f}  home_adv={model.home_adv_:.3f}{ht_note}")
                fitted += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                skipped += 1

        print(f"\n    Dixon-Coles: fitted {fitted} leagues, skipped {skipped}")
        return self

    def predict_draw(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict P(HT draw) for each row. Returns array of shape (n,).
        Looks up the correct league model per row, with fuzzy team-name fallback.
        """
        from src.utils import resolve_team_name
        probs = np.full(len(df), self.global_draw_rate_)

        for i, row in df.iterrows():
            league = row.get("league", None)
            home   = row.get("HomeTeam", "")
            away   = row.get("AwayTeam", "")
            idx    = df.index.get_loc(i)

            if league and league in self.league_models_:
                model = self.league_models_[league]
                # Try exact, then fuzzy within league model
                h = home if home in model.team_idx_ else (resolve_team_name(home, model.teams_) or "")
                a = away if away in model.team_idx_ else (resolve_team_name(away, model.teams_) or "")
                if h and a:
                    probs[idx] = model.predict_draw_proba(h, a)
                else:
                    probs[idx] = model.train_draw_rate_
            elif not league and len(self.league_models_) == 1:
                model = next(iter(self.league_models_.values()))
                probs[idx] = model.predict_draw_proba(home, away)
            else:
                # No league specified — try exact match across all league models first
                found = False
                for model in self.league_models_.values():
                    if home in model.team_idx_ and away in model.team_idx_:
                        probs[idx] = model.predict_draw_proba(home, away)
                        found = True
                        break
                if not found:
                    # Fuzzy match across all league models
                    for model in self.league_models_.values():
                        h = resolve_team_name(home, model.teams_)
                        a = resolve_team_name(away, model.teams_)
                        if h and a:
                            probs[idx] = model.predict_draw_proba(h, a)
                            break

        return probs

    def predict_draw_single(self, home_team: str, away_team: str,
                            league: Optional[str] = None) -> float:
        """
        Predict P(HT draw) for a single match.
        Falls back to fuzzy team-name matching when exact lookup fails.
        """
        from src.utils import resolve_team_name

        def _lookup(model: DixonColes, home: str, away: str) -> Optional[float]:
            all_teams = model.teams_
            h = resolve_team_name(home, all_teams) if home not in model.team_idx_ else home
            a = resolve_team_name(away, all_teams) if away not in model.team_idx_ else away
            if h and a:
                return model.predict_draw_proba(h, a)
            return None

        if league and league in self.league_models_:
            p = _lookup(self.league_models_[league], home_team, away_team)
            return p if p is not None else self.global_draw_rate_

        # Try exact match first across all leagues
        for model in self.league_models_.values():
            if home_team in model.team_idx_ and away_team in model.team_idx_:
                return model.predict_draw_proba(home_team, away_team)

        # Fuzzy match across all leagues
        for model in self.league_models_.values():
            p = _lookup(model, home_team, away_team)
            if p is not None:
                return p

        return self.global_draw_rate_

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"    Dixon-Coles ensemble saved → {path}")

    @staticmethod
    def load(path: str) -> "DixonColesEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Standalone fitting script ─────────────────────────────────────────────────

def fit_from_parquet(
    parquet_path: str = "data/processed/mega_dataset.parquet",
    train_frac: float = 0.70,
    xi: float = 0.003,
    save_path: str = "models/v2/dixon_coles.pkl",
) -> DixonColesEnsemble:
    """Load mega dataset, split 70/15/15, fit on train, return ensemble."""
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["HTHG", "HTAG"]).sort_values("Date").reset_index(drop=True)

    n         = len(df)
    train_end = int(train_frac * n)
    train_df  = df.iloc[:train_end].copy()

    print(f"  Fitting Dixon-Coles on {len(train_df):,} training matches...")
    dc = DixonColesEnsemble(xi=xi)
    dc.fit(train_df)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    dc.save(save_path)
    return dc


if __name__ == "__main__":
    dc = fit_from_parquet()
    print("\nExample prediction:")
    p = dc.predict_draw_single("Leeds United", "Sheffield Utd", league="E1")
    print(f"  Leeds United vs Sheffield Utd (EFL): P(HT draw) = {p:.3f}")
