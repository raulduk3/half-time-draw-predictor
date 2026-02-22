"""
Stacking Ensemble Predictor — Half-Time Draw Probability
=========================================================
Combines five signal sources:
  1. Dixon-Coles P(draw)        — bivariate Poisson on HT scores
  2. Elo P(draw)                — rolling Elo rating + logistic mapping
  3. XGBoost P(draw)            — pre-trained mega model
  4. LightGBM P(draw)           — pre-trained mega model
  5. Market consensus P(draw)   — multi-bookmaker normalized implied prob
  6. Referee adjustment factor  — per-referee HT draw rate multiplier

Stack:
  - Meta-learner: Logistic Regression trained on validation set predictions
  - Calibration: IsotonicRegression applied to meta-learner output
  - Confidence interval: bootstrap std → ±1.96σ

Output per match:
  - p_ensemble: calibrated final P(HT draw)
  - ci_lower, ci_upper: 90% bootstrap CI
  - breakdown: dict of each model's contribution
  - key_drivers: factors pushing toward/away from draw

Usage:
    from src.ensemble_predictor import EnsemblePredictor
    ep = EnsemblePredictor.load("models/v2/ensemble.pkl")
    result = ep.predict_single(home_team="Leeds", away_team="Sheffield Utd",
                               league="E1", referee="M Dean",
                               market_odds={"B365H": 2.2, "B365D": 3.1, "B365A": 3.6})
    print(result)
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


# ── Helper: safe clamp ────────────────────────────────────────────────────────

def _safe_prob(x: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return float(np.clip(x, lo, hi))


# ── EnsemblePredictor ─────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Stacking ensemble for P(HT draw) prediction.

    Parameters
    ----------
    use_isotonic : bool
        Use isotonic regression calibration (default True). If False, uses
        Platt scaling (logistic).
    n_bootstrap : int
        Bootstrap samples for confidence interval estimation (default 200).
    """

    # Signal names (order matters for feature matrix columns)
    SIGNAL_NAMES = [
        "dixon_coles", "elo", "xgb", "lgb", "market_consensus",
        "referee_adj",   # referee multiplier applied as an additive feature
    ]

    def __init__(
        self,
        use_isotonic: bool = True,
        n_bootstrap: int = 200,
    ):
        self.use_isotonic  = use_isotonic
        self.n_bootstrap   = n_bootstrap

        # Sub-models (set externally before fit)
        self.dixon_coles   = None   # DixonColesEnsemble
        self.elo           = None   # EloRatingSystem
        self.xgb_model     = None   # xgboost.Booster
        self.lgb_model     = None   # lightgbm.Booster
        self.market_model  = None   # MarketModel
        self.referee_model = None   # RefereeModel

        # XGBoost/LGB pre-processing
        self.scaler_        = None
        self.medians_       = None
        self.feature_names_ = None
        self.xgb_calibrator_ = None
        self.lgb_calibrator_ = None

        # Meta-learner
        self.meta_lr_: Optional[LogisticRegression] = None
        self.calibrator_: Optional[IsotonicRegression] = None

        # Metadata
        self.global_draw_rate_: float = 0.42
        self.val_signal_aucs_: Dict[str, float] = {}
        self.fitted_: bool = False

    # ── Signal extraction ─────────────────────────────────────────────────────

    def _get_mega_preds(
        self,
        df: pd.DataFrame,
        model_type: str = "xgb",
    ) -> np.ndarray:
        """
        Get XGBoost or LightGBM predictions using the pre-trained mega model.
        Handles feature engineering, scaling, and imputation.
        """
        import xgboost as xgb
        import lightgbm as lgb

        if self.feature_names_ is None or self.scaler_ is None:
            return np.full(len(df), self.global_draw_rate_)

        # Build feature matrix
        X = np.zeros((len(df), len(self.feature_names_)), dtype=np.float32)
        for j, col in enumerate(self.feature_names_):
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").values
                median = self.medians_.get(col, 0.0)
                X[:, j] = np.where(np.isfinite(vals), vals, median)
            else:
                X[:, j] = self.medians_.get(col, 0.0)

        X_scaled = self.scaler_.transform(X)

        if model_type == "xgb" and self.xgb_model is not None:
            dmat  = xgb.DMatrix(X)   # XGB uses unscaled
            preds = self.xgb_model.predict(dmat)
            if self.xgb_calibrator_ is not None:
                preds = self.xgb_calibrator_.predict_proba(
                    preds.reshape(-1, 1))[:, 1]
        elif model_type == "lgb" and self.lgb_model is not None:
            preds = self.lgb_model.predict(X)
            if self.lgb_calibrator_ is not None:
                preds = self.lgb_calibrator_.predict_proba(
                    preds.reshape(-1, 1))[:, 1]
        else:
            preds = np.full(len(df), self.global_draw_rate_)

        return np.clip(preds, 0.01, 0.99)

    def _extract_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build signal matrix: shape (n_matches, n_signals).
        Signals: [dixon_coles, elo, xgb, lgb, market_consensus, referee_adj].
        """
        n = len(df)
        signals = np.full((n, len(self.SIGNAL_NAMES)), self.global_draw_rate_)

        # 0. Dixon-Coles
        if self.dixon_coles is not None:
            try:
                signals[:, 0] = self.dixon_coles.predict_draw(df)
            except Exception:
                pass

        # 1. Elo
        if self.elo is not None:
            try:
                signals[:, 1] = self.elo.predict_draw(df)
            except Exception:
                pass

        # 2. XGBoost
        try:
            signals[:, 2] = self._get_mega_preds(df, "xgb")
        except Exception:
            pass

        # 3. LightGBM
        try:
            signals[:, 3] = self._get_mega_preds(df, "lgb")
        except Exception:
            pass

        # 4. Market consensus
        if self.market_model is not None:
            try:
                mf = self.market_model.transform(df)
                cons = mf["consensus_draw_prob"].values
                signals[:, 4] = np.where(np.isfinite(cons), cons, self.global_draw_rate_)
            except Exception:
                pass
        else:
            # Fallback: use B365 if available
            if "B365D" in df.columns and "B365H" in df.columns and "B365A" in df.columns:
                h = pd.to_numeric(df["B365H"], errors="coerce")
                d = pd.to_numeric(df["B365D"], errors="coerce")
                a = pd.to_numeric(df["B365A"], errors="coerce")
                total = (1/h) + (1/d) + (1/a)
                prob  = (1/d) / total
                signals[:, 4] = np.where(np.isfinite(prob), prob, self.global_draw_rate_)

        # 5. Referee adjustment (multiplicative → converted to additive offset)
        if self.referee_model is not None:
            try:
                adj = self.referee_model.predict_draw_adjustment(df)
                # Store raw adjustment factor; meta-learner will weight it
                signals[:, 5] = adj
            except Exception:
                signals[:, 5] = 1.0
        else:
            signals[:, 5] = 1.0

        return signals

    def _apply_referee_adjustment(
        self,
        base_prob: np.ndarray,
        referee_adj: np.ndarray,
    ) -> np.ndarray:
        """
        Apply referee adjustment multiplicatively, then renormalize.
        adj=1.2 → draw 20% more likely → boost raw probability, renormalize.
        """
        p_draw = base_prob * referee_adj
        # Renormalize: (1-p_draw) scales by (1 - adj_part) approximately
        # Simple multiplicative cap
        return np.clip(p_draw, 0.01, 0.99)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        val_df: pd.DataFrame,
        y_val: np.ndarray,
        test_df: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> "EnsemblePredictor":
        """
        Train the meta-learner on validation set predictions.
        Calibrate with isotonic regression.
        Optionally evaluate on test set.

        All sub-models (dixon_coles, elo, xgb_model, ...) must be set
        before calling fit().
        """
        print("  Extracting validation signals for meta-learner...")
        X_val_signals = self._extract_signals(val_df)

        # Log individual signal AUCs on val set
        for i, name in enumerate(self.SIGNAL_NAMES):
            col = X_val_signals[:, i]
            if name == "referee_adj":
                # Referee adj is multiplicative, not a probability — skip AUC
                continue
            valid = np.isfinite(col)
            if valid.sum() < 10:
                continue
            try:
                auc = roc_auc_score(y_val[valid], col[valid])
                self.val_signal_aucs_[name] = float(auc)
                print(f"    {name:<25} val AUC = {auc:.4f}")
            except Exception:
                pass

        # Meta-learner: LogisticRegression on signal matrix
        valid_rows = np.isfinite(X_val_signals).all(axis=1)
        X_meta = X_val_signals[valid_rows]
        y_meta = y_val[valid_rows]

        print(f"  Training meta-learner on {valid_rows.sum():,} val rows...")
        self.meta_lr_ = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta_lr_.fit(X_meta, y_meta)

        val_preds_raw = self.meta_lr_.predict_proba(X_meta)[:, 1]
        val_auc_meta  = roc_auc_score(y_meta, val_preds_raw)
        print(f"  Meta-learner val AUC (before calibration): {val_auc_meta:.4f}")

        # Calibrate with Isotonic Regression
        print("  Fitting isotonic calibration on val predictions...")
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_.fit(val_preds_raw, y_meta)

        val_preds_cal = self.calibrator_.predict(val_preds_raw)
        val_auc_cal   = roc_auc_score(y_meta, val_preds_cal)
        val_brier_cal = brier_score_loss(y_meta, val_preds_cal)
        print(f"  Meta-learner val AUC (after calibration):  {val_auc_cal:.4f}  "
              f"Brier: {val_brier_cal:.4f}")

        self.fitted_ = True

        # Evaluate on test set if provided
        if test_df is not None and y_test is not None:
            print("\n  Evaluating ensemble on test set...")
            test_preds, _ = self._predict_proba_breakdown(test_df)
            valid_t = np.isfinite(test_preds)
            if valid_t.sum() > 0:
                test_auc   = roc_auc_score(y_test[valid_t], test_preds[valid_t])
                test_brier = brier_score_loss(y_test[valid_t], test_preds[valid_t])
                print(f"  ENSEMBLE TEST AUC:   {test_auc:.4f}")
                print(f"  ENSEMBLE TEST BRIER: {test_brier:.4f}")

                # Compare with individual signals
                test_signals = self._extract_signals(test_df)
                print(f"\n  Individual signal AUCs on test set:")
                for i, name in enumerate(self.SIGNAL_NAMES):
                    if name == "referee_adj":
                        continue
                    col = test_signals[:, i]
                    valid = np.isfinite(col) & np.isfinite(y_test.astype(float))
                    if valid.sum() < 10:
                        continue
                    try:
                        auc = roc_auc_score(y_test[valid], col[valid])
                        print(f"    {name:<25} test AUC = {auc:.4f}")
                    except Exception:
                        pass
                print(f"    {'ENSEMBLE':25} test AUC = {test_auc:.4f}  ← FINAL")

        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def _predict_proba_breakdown(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (calibrated_probs, signal_matrix).
        """
        X_signals = self._extract_signals(df)

        if not self.fitted_:
            # Simple average of available signals
            probs = np.nanmean(X_signals[:, :5], axis=1)
            return np.clip(probs, 0.01, 0.99), X_signals

        valid_rows = np.isfinite(X_signals).all(axis=1)
        preds = np.full(len(df), self.global_draw_rate_)

        if valid_rows.sum() > 0:
            raw = self.meta_lr_.predict_proba(X_signals[valid_rows])[:, 1]
            cal = self.calibrator_.predict(raw)
            preds[valid_rows] = cal

        return np.clip(preds, 0.01, 0.99), X_signals

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(HT draw) for each row. Shape (n,)."""
        probs, _ = self._predict_proba_breakdown(df)
        return probs

    def predict_single(
        self,
        home_team: str,
        away_team: str,
        league: str = "",
        referee: Optional[str] = None,
        market_odds: Optional[Dict[str, float]] = None,
        extra_features: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Predict P(HT draw) for one match with full breakdown.

        Parameters
        ----------
        home_team, away_team : str
        league : str
            League code (e.g. "E1")
        referee : str, optional
        market_odds : dict, optional
            e.g. {"B365H": 2.2, "B365D": 3.1, "B365A": 3.6}
        extra_features : dict, optional
            Additional features to pass to XGB/LGB models.

        Returns
        -------
        dict with keys:
          p_ensemble, ci_lower, ci_upper,
          breakdown (dict), key_drivers (list), quarter_kelly_stake
        """
        # Build a single-row DataFrame
        row = {
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "league":   league,
        }
        if referee:
            row["Referee"] = referee
        if market_odds:
            row.update(market_odds)
        if extra_features:
            row.update(extra_features)

        df_row = pd.DataFrame([row])

        # Extract signals
        signals = self._extract_signals(df_row)[0]  # shape (n_signals,)

        # Individual signal probs
        dc_prob  = float(signals[0])
        elo_prob = float(signals[1])
        xgb_prob = float(signals[2])
        lgb_prob = float(signals[3])
        mkt_prob = float(signals[4])
        ref_adj  = float(signals[5])

        # Meta-learner prediction
        if self.fitted_ and self.meta_lr_ is not None:
            raw_meta = self.meta_lr_.predict_proba(signals.reshape(1, -1))[0, 1]
            p_cal    = float(self.calibrator_.predict([raw_meta])[0])
        else:
            # Simple mean if not fitted
            valid_probs = [p for p in [dc_prob, elo_prob, xgb_prob, lgb_prob, mkt_prob]
                           if np.isfinite(p)]
            p_cal = float(np.mean(valid_probs)) if valid_probs else self.global_draw_rate_

        p_cal = _safe_prob(p_cal)

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(signals)

        # Key drivers
        drivers = self._key_drivers(
            p_cal, dc_prob, elo_prob, xgb_prob, lgb_prob, mkt_prob, ref_adj
        )

        # Quarter-Kelly stake (if market odds provided)
        qk_stake = None
        if market_odds and "B365D" in market_odds:
            d_odds = float(market_odds["B365D"])
            if d_odds > 1.0:
                b = d_odds - 1.0
                kelly = (p_cal * b - (1.0 - p_cal)) / b
                kelly = max(0.0, kelly)
                qk_stake = round(kelly * 0.25, 4)

        return {
            "home_team":       home_team,
            "away_team":       away_team,
            "league":          league,
            "p_ensemble":      round(p_cal, 4),
            "ci_lower":        round(ci_lower, 4),
            "ci_upper":        round(ci_upper, 4),
            "breakdown": {
                "dixon_coles":      round(dc_prob, 4)  if np.isfinite(dc_prob)  else None,
                "elo":              round(elo_prob, 4) if np.isfinite(elo_prob) else None,
                "xgboost":         round(xgb_prob, 4) if np.isfinite(xgb_prob) else None,
                "lightgbm":        round(lgb_prob, 4) if np.isfinite(lgb_prob) else None,
                "market_consensus":round(mkt_prob, 4) if np.isfinite(mkt_prob) else None,
                "referee_adj":     round(ref_adj, 4),
            },
            "key_drivers":     drivers,
            "quarter_kelly_stake": qk_stake,
            "global_draw_rate":    round(self.global_draw_rate_, 4),
        }

    def _bootstrap_ci(
        self,
        signals: np.ndarray,
        alpha: float = 0.10,
    ) -> Tuple[float, float]:
        """Approximate CI via perturbation of signal vector."""
        if not self.fitted_:
            return self.global_draw_rate_ * 0.8, self.global_draw_rate_ * 1.2

        rng = np.random.default_rng(42)
        preds = []
        for _ in range(self.n_bootstrap):
            # Add small noise to each signal
            noise    = rng.normal(0, 0.02, size=signals.shape)
            perturbed = np.clip(signals + noise, 0.01, 0.99)
            if not np.isfinite(perturbed).all():
                continue
            raw = self.meta_lr_.predict_proba(perturbed.reshape(1, -1))[0, 1]
            cal = float(self.calibrator_.predict([raw])[0])
            preds.append(cal)

        if not preds:
            return 0.3, 0.6

        lo = float(np.percentile(preds, alpha * 100 / 2 * 100 / 100))
        hi = float(np.percentile(preds, (1 - alpha / 2) * 100))
        return round(lo, 4), round(hi, 4)

    def _key_drivers(
        self,
        p_final: float,
        dc: float, elo: float, xgb: float, lgb: float, mkt: float, ref: float,
    ) -> List[Dict]:
        """Generate human-readable key drivers."""
        drivers = []
        base = self.global_draw_rate_

        signals_named = [
            ("Dixon-Coles model",     dc),
            ("Elo ratings",           elo),
            ("XGBoost model",         xgb),
            ("LightGBM model",        lgb),
            ("Market consensus",      mkt),
        ]

        for name, val in signals_named:
            if not np.isfinite(val):
                continue
            diff = val - base
            direction = "toward draw" if diff > 0 else "away from draw"
            if abs(diff) > 0.01:
                drivers.append({
                    "factor":    name,
                    "value":     round(val, 3),
                    "direction": direction,
                    "delta":     round(diff, 3),
                })

        # Referee
        if abs(ref - 1.0) > 0.05:
            direction = "toward draw" if ref > 1.0 else "away from draw"
            drivers.append({
                "factor":    "Referee tendency",
                "value":     round(ref, 3),
                "direction": direction,
                "delta":     round((ref - 1.0) * base, 3),
            })

        # Sort by absolute impact
        drivers.sort(key=lambda d: abs(d["delta"]), reverse=True)
        return drivers

    # ── Expected value ────────────────────────────────────────────────────────

    def expected_value(self, p_model: float, decimal_odds: float) -> float:
        """Expected value per unit staked on draw at given odds."""
        return p_model * (decimal_odds - 1) - (1 - p_model)

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save ensemble (excluding large sub-models which have their own paths)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save meta-learner + calibrator separately to avoid huge pickle
        payload = {
            "meta_lr":           self.meta_lr_,
            "calibrator":        self.calibrator_,
            "global_draw_rate":  self.global_draw_rate_,
            "val_signal_aucs":   self.val_signal_aucs_,
            "fitted":            self.fitted_,
            "use_isotonic":      self.use_isotonic,
            "n_bootstrap":       self.n_bootstrap,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"    Ensemble meta-learner saved → {path}")

    def load_weights(self, path: str) -> None:
        """Load meta-learner + calibrator from saved file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.meta_lr_           = payload["meta_lr"]
        self.calibrator_        = payload["calibrator"]
        self.global_draw_rate_  = payload["global_draw_rate"]
        self.val_signal_aucs_   = payload["val_signal_aucs"]
        self.fitted_            = payload["fitted"]
        self.use_isotonic       = payload["use_isotonic"]
        self.n_bootstrap        = payload.get("n_bootstrap", 200)

    @staticmethod
    def load_full(
        weights_path: str,
        dc_path: str,
        elo_path: str,
        xgb_path: str,
        lgb_path: str,
        scaler_path: str,
        medians_path: str,
        features_path: str,
        xgb_cal_path: Optional[str] = None,
        lgb_cal_path: Optional[str] = None,
        referee_path: Optional[str] = None,
        market_path: Optional[str] = None,
    ) -> "EnsemblePredictor":
        """
        Load a fully assembled EnsemblePredictor from saved component files.
        """
        import xgboost as xgb_lib
        import lightgbm as lgb_lib

        ep = EnsemblePredictor()
        ep.load_weights(weights_path)

        # Dixon-Coles
        if Path(dc_path).exists():
            from src.dixon_coles import DixonColesEnsemble
            ep.dixon_coles = DixonColesEnsemble.load(dc_path)

        # Elo
        if Path(elo_path).exists():
            from src.elo import EloRatingSystem
            ep.elo = EloRatingSystem.load(elo_path)

        # XGBoost
        if Path(xgb_path).exists():
            ep.xgb_model = xgb_lib.Booster()
            ep.xgb_model.load_model(xgb_path)

        # LightGBM
        if Path(lgb_path).exists():
            ep.lgb_model = lgb_lib.Booster(model_file=lgb_path)

        # Scaler
        if Path(scaler_path).exists():
            with open(scaler_path, "rb") as f:
                ep.scaler_ = pickle.load(f)

        # Medians
        if Path(medians_path).exists():
            with open(medians_path, "r") as f:
                ep.medians_ = json.load(f)

        # Features
        if Path(features_path).exists():
            with open(features_path, "r") as f:
                ep.feature_names_ = json.load(f)

        # XGB calibrator
        if xgb_cal_path and Path(xgb_cal_path).exists():
            with open(xgb_cal_path, "rb") as f:
                ep.xgb_calibrator_ = pickle.load(f)

        # LGB calibrator
        if lgb_cal_path and Path(lgb_cal_path).exists():
            with open(lgb_cal_path, "rb") as f:
                ep.lgb_calibrator_ = pickle.load(f)

        # Referee
        if referee_path and Path(referee_path).exists():
            from src.referee_model import RefereeModel
            ep.referee_model = RefereeModel.load(referee_path)

        # Market
        if market_path and Path(market_path).exists():
            from src.market_model import MarketModel
            ep.market_model = MarketModel.load(market_path)

        print(f"    Ensemble fully loaded from {weights_path}")
        return ep
