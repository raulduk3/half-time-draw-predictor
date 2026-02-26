"""
Referee Impact Model — Half-Time Draw Prediction
=================================================
Analyses referee effects on half-time draw rates from EFL Championship raw data.

- Loads all data/raw/E1*.csv files
- Computes per-referee: HT draw rate, avg HT goals, cards, fouls
- Statistical testing: chi-squared (draw rate vs baseline), permutation test
- Outputs adjustment factor = ratio of ref draw rate to league baseline
- Capped at [0.6, 1.7] to avoid extreme adjustments on small samples

Usage:
    from src.referee_model import RefereeModel
    rm = RefereeModel()
    rm.fit(train_df_raw)
    adj = rm.get_adjustment("M Dean")   # e.g. 1.12
    rm.save("models/v2/referee_model.pkl")
"""

from __future__ import annotations

import glob
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm


# ── Data loading ─────────────────────────────────────────────────────────────

def load_efl_raw(
    raw_dir: str = "data/raw",
    pattern: str = "E1*.csv",
) -> pd.DataFrame:
    """Load all EFL Championship raw CSV files into a single DataFrame."""
    files = sorted(glob.glob(str(Path(raw_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {raw_dir}/{pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
            df.columns = df.columns.str.replace("ï»¿", "", regex=False).str.strip()
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)

    # Normalise date
    for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
        try:
            combined["Date"] = pd.to_datetime(combined["Date"], format=fmt)
            break
        except Exception:
            continue

    # Ensure HT columns present
    for col in ["HTHG", "HTAG"]:
        if col not in combined.columns:
            combined[col] = np.nan

    combined["y_ht_draw"] = (combined["HTHG"] == combined["HTAG"]).astype(float)
    combined["ht_goals"]  = combined["HTHG"].fillna(0) + combined["HTAG"].fillna(0)

    return combined.sort_values("Date").reset_index(drop=True)


# ── Statistical tests ─────────────────────────────────────────────────────────

def chi2_draw_test(
    ref_draws: int,
    ref_total: int,
    global_draws: int,
    global_total: int,
) -> Tuple[float, float]:
    """
    Chi-squared test: is this referee's draw rate significantly different?
    Returns (chi2_stat, p_value).
    """
    table = np.array([
        [ref_draws,              ref_total - ref_draws],
        [global_draws - ref_draws, global_total - global_total // 1 - ref_total + ref_total],
    ])
    # Simplified 2x2 contingency
    contingency = np.array([
        [ref_draws,         ref_total - ref_draws],
        [global_draws - ref_draws, (global_total - ref_total) - (global_draws - ref_draws)],
    ])
    if (contingency < 0).any():
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p, _, _ = chi2_contingency(contingency, correction=False)
    return float(chi2), float(p)


def permutation_draw_rate(
    ref_mask: np.ndarray,
    y_draw: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """
    Permutation test p-value: how often does a random subset this size
    have a draw rate as extreme as this referee?
    """
    rng = np.random.default_rng(seed)
    ref_rate = y_draw[ref_mask].mean()
    n        = ref_mask.sum()

    extreme = 0
    for _ in range(n_permutations):
        perm_idx = rng.choice(len(y_draw), size=n, replace=False)
        perm_rate = y_draw[perm_idx].mean()
        if abs(perm_rate - y_draw.mean()) >= abs(ref_rate - y_draw.mean()):
            extreme += 1
    return extreme / n_permutations


# ── Referee Model ─────────────────────────────────────────────────────────────

class RefereeModel:
    """
    Builds a per-referee adjustment factor for HT draw probability.

    Parameters
    ----------
    min_matches : int
        Minimum matches for a referee to have their own profile (default 20).
    adj_cap : tuple
        (min, max) bounds on the adjustment factor (default 0.6, 1.7).
    confidence_threshold : float
        p-value threshold for a referee's effect to be considered significant.
    """

    def __init__(
        self,
        min_matches: int = 20,
        adj_cap: Tuple[float, float] = (0.6, 1.7),
        confidence_threshold: float = 0.10,
        prior_strength: int = 30,
    ):
        self.min_matches          = min_matches
        self.adj_cap              = adj_cap
        self.confidence_threshold = confidence_threshold
        self.prior_strength       = prior_strength

        self.profiles_: Dict[str, Dict] = {}
        self.global_draw_rate_: float   = 0.42
        self.global_stats_: Dict        = {}
        self.fitted_: bool              = False

    def fit(self, df: pd.DataFrame) -> "RefereeModel":
        """
        Build referee profiles from training data.

        df must have: Referee, HTHG, HTAG, y_ht_draw, and optionally
        HY, AY, HF, AF (cards/fouls).
        """
        df = df.copy()

        if "Referee" not in df.columns:
            print("    Warning: no Referee column — referee model will return 1.0 always")
            self.fitted_ = True
            return self

        df = df.dropna(subset=["Referee", "HTHG", "HTAG"])
        df["Referee"] = df["Referee"].str.strip().str.title()
        df["y_ht_draw"] = (df["HTHG"] == df["HTAG"]).astype(float)
        df["ht_goals"]  = df["HTHG"] + df["HTAG"]

        y_all = df["y_ht_draw"].values
        self.global_draw_rate_ = float(y_all.mean())
        global_total  = len(df)
        global_draws  = int(y_all.sum())

        # Global averages
        self.global_stats_ = {
            "draw_rate":    self.global_draw_rate_,
            "avg_ht_goals": float(df["ht_goals"].mean()),
            "avg_hy":       float(df["HY"].mean()) if "HY" in df.columns else None,
            "avg_ay":       float(df["AY"].mean()) if "AY" in df.columns else None,
            "avg_hf":       float(df["HF"].mean()) if "HF" in df.columns else None,
            "avg_af":       float(df["AF"].mean()) if "AF" in df.columns else None,
            "n_matches":    global_total,
        }

        profiles = {}
        for referee, grp in df.groupby("Referee"):
            n = len(grp)
            if n < self.min_matches:
                continue

            ref_draws = int(grp["y_ht_draw"].sum())
            ref_rate  = grp["y_ht_draw"].mean()

            # Statistical test
            chi2, p_chi2 = chi2_draw_test(
                ref_draws, n, global_draws, global_total
            )

            # Bayesian-smoothed draw rate (weighted towards global)
            alpha = self.prior_strength  # decoupled from min_matches threshold
            smoothed_rate = (ref_draws + alpha * self.global_draw_rate_) / (n + alpha)

            # Adjustment factor (clipped)
            adj_raw = smoothed_rate / self.global_draw_rate_
            adj     = float(np.clip(adj_raw, self.adj_cap[0], self.adj_cap[1]))

            profiles[referee] = {
                "n_matches":        n,
                "n_draws":          ref_draws,
                "draw_rate":        float(ref_rate),
                "smoothed_rate":    float(smoothed_rate),
                "adjustment":       adj,
                "chi2":             float(chi2),
                "p_chi2":           float(p_chi2),
                "significant":      p_chi2 < self.confidence_threshold,
                "avg_ht_goals":     float(grp["ht_goals"].mean()),
                "avg_hy":           float(grp["HY"].mean()) if "HY" in grp.columns else None,
                "avg_ay":           float(grp["AY"].mean()) if "AY" in grp.columns else None,
                "avg_hf":           float(grp["HF"].mean()) if "HF" in grp.columns else None,
                "avg_af":           float(grp["AF"].mean()) if "AF" in grp.columns else None,
            }

        self.profiles_ = profiles

        n_sig = sum(1 for p in profiles.values() if p["significant"])
        print(f"    Referee model: {len(profiles)} referees profiled, "
              f"{n_sig} statistically significant (p<{self.confidence_threshold})")
        print(f"    Global HT draw rate: {self.global_draw_rate_:.3f}")

        # Print top/bottom referees by draw rate
        sig_refs = [(r, p) for r, p in profiles.items() if p["significant"]]
        if sig_refs:
            sig_refs_sorted = sorted(sig_refs, key=lambda x: x[1]["draw_rate"], reverse=True)
            print(f"\n    Most draw-prone referees (significant):")
            for r, p in sig_refs_sorted[:5]:
                print(f"      {r:<25} rate={p['draw_rate']:.3f}  adj={p['adjustment']:.3f}  "
                      f"n={p['n_matches']}  p={p['p_chi2']:.3f}")
            print(f"\n    Least draw-prone referees (significant):")
            for r, p in sig_refs_sorted[-5:]:
                print(f"      {r:<25} rate={p['draw_rate']:.3f}  adj={p['adjustment']:.3f}  "
                      f"n={p['n_matches']}  p={p['p_chi2']:.3f}")

        self.fitted_ = True
        return self

    def get_adjustment(self, referee: Optional[str]) -> float:
        """
        Return adjustment factor for a referee.
        Returns 1.0 if referee unknown or not significant.
        """
        if not self.fitted_ or referee is None:
            return 1.0

        ref = str(referee).strip().title()
        profile = self.profiles_.get(ref)
        if profile is None:
            return 1.0

        # Only use adjustment if statistically significant
        if not profile["significant"]:
            return 1.0

        return profile["adjustment"]

    def get_profile(self, referee: str) -> Optional[Dict]:
        """Return full profile for a referee, or None."""
        ref = str(referee).strip().title()
        return self.profiles_.get(ref)

    def get_all_profiles_df(self) -> pd.DataFrame:
        """Return all referee profiles as a sorted DataFrame."""
        rows = []
        for ref, p in self.profiles_.items():
            rows.append({"referee": ref, **p})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("draw_rate", ascending=False)

    def predict_draw_adjustment(self, df: pd.DataFrame) -> np.ndarray:
        """
        For a DataFrame with a 'Referee' column, return an array of
        adjustment factors (one per row).
        """
        if "Referee" not in df.columns:
            return np.ones(len(df))

        return np.array([
            self.get_adjustment(ref)
            for ref in df["Referee"].fillna("").values
        ])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # Also save human-readable JSON
        json_path = str(path).replace(".pkl", "_profiles.json")
        with open(json_path, "w") as f:
            json.dump({
                "global_stats": self.global_stats_,
                "profiles": self.profiles_,
            }, f, indent=2)
        print(f"    Referee model saved → {path}")
        print(f"    Referee profiles JSON → {json_path}")

    @staticmethod
    def load(path: str) -> "RefereeModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Standalone script ─────────────────────────────────────────────────────────

def fit_from_raw_csv(
    raw_dir: str = "data/raw",
    train_frac: float = 0.70,
    save_path: str = "models/v2/referee_model.pkl",
) -> RefereeModel:
    """Load EFL raw CSVs, split 70%, fit referee model on train."""
    print("  Loading EFL raw CSV files...")
    df = load_efl_raw(raw_dir)
    print(f"  Loaded {len(df):,} EFL matches")

    n        = len(df)
    train_df = df.iloc[:int(train_frac * n)].copy()
    print(f"  Training on {len(train_df):,} matches "
          f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")

    rm = RefereeModel()
    rm.fit(train_df)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    rm.save(save_path)
    return rm


if __name__ == "__main__":
    rm = fit_from_raw_csv()

    # Example
    adj = rm.get_adjustment("M Dean")
    print(f"\nM Dean adjustment: {adj:.3f}")
    print(f"Global draw rate: {rm.global_draw_rate_:.3f}")

    print("\nAll significant referees:")
    df_profiles = rm.get_all_profiles_df()
    sig = df_profiles[df_profiles["significant"]]
    print(sig[["referee", "draw_rate", "adjustment", "n_matches", "p_chi2"]].to_string(index=False))
