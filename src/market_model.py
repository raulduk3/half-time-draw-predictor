"""
Multi-Bookmaker Market Signal Extraction
=========================================
Extracts wisdom-of-crowds signals from multiple bookmaker odds.

Available books in raw CSVs:
  Opening: B365, BW, IW, LB, WH, SJ, VC, PS (Pinnacle opening)
  Closing: B365C/BWCH/IWCH/WHCH/VCCH/PSCH (via newer files)
  Consensus: Max, Avg

Features computed per match:
  1. consensus_draw_prob  — average normalized implied P(draw)
  2. max_draw_prob        — highest draw prob across books
  3. std_draw_probs       — spread of implied draw probs (uncertainty)
  4. n_books              — number of books available
  5. pinnacle_draw_prob   — Pinnacle (sharpest book) P(draw) if available
  6. line_movement        — Pinnacle closing vs opening P(draw)
  7. max_minus_min        — range of draw probs across books
  8. log_consensus_draw   — log of consensus P(draw) (for LR features)

All probabilities are normalized (overround removed via multiplicative norm).

Usage:
    from src.market_model import MarketModel, extract_multi_book_features
    mm = MarketModel()
    mm.fit(raw_df)
    features = mm.transform(df)   # returns DataFrame of features
    mm.save("models/v2/market_model.pkl")
"""

from __future__ import annotations

import glob
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Bookmaker definitions ─────────────────────────────────────────────────────

# Each tuple: (home_col, draw_col, away_col, label)
BOOKMAKER_TRIPLES: List[Tuple[str, str, str, str]] = [
    ("B365H", "B365D", "B365A", "B365"),
    ("BWH",   "BWD",   "BWA",   "BW"),
    ("IWH",   "IWD",   "IWA",   "IW"),
    ("LBH",   "LBD",   "LBA",   "LB"),
    ("WHH",   "WHD",   "WHA",   "WH"),
    ("VCH",   "VCD",   "VCA",   "VC"),
    ("PSH",   "PSD",   "PSA",   "PS"),   # Pinnacle opening
    ("SJH",   "SJD",   "SJA",   "SJ"),
    ("MaxH",  "MaxD",  "MaxA",  "Max"),
    ("AvgH",  "AvgD",  "AvgA",  "Avg"),
]

# Closing odds (Pinnacle closing = sharpest market signal)
CLOSING_TRIPLES: List[Tuple[str, str, str, str]] = [
    ("PSCH",  "PSCD",  "PSCA",  "PS_close"),
    ("B365CH", "B365CD", "B365CA", "B365_close"),
    ("BWCH",  "BWCD",  "BWCA",  "BW_close"),
    ("MaxCH", "MaxCD", "MaxCA", "Max_close"),
    ("AvgCH", "AvgCD", "AvgCA", "Avg_close"),
]


def normalize_odds(h_odds: float, d_odds: float, a_odds: float) -> Tuple[float, float, float]:
    """
    Convert 3-way odds to normalized probabilities (overround removed).
    Returns (p_home, p_draw, p_away). Returns NaN tuple if any odds are invalid.
    """
    try:
        ph = 1.0 / float(h_odds)
        pd_ = 1.0 / float(d_odds)
        pa = 1.0 / float(a_odds)
        total = ph + pd_ + pa
        if total <= 0 or not np.isfinite(total):
            return np.nan, np.nan, np.nan
        return ph / total, pd_ / total, pa / total
    except Exception:
        return np.nan, np.nan, np.nan


def extract_row_features(row: pd.Series) -> Dict[str, float]:
    """
    Extract all market features from a single match row.
    Returns a dict of feature_name → value.
    """
    draw_probs: Dict[str, float] = {}

    for h_col, d_col, a_col, label in BOOKMAKER_TRIPLES:
        if h_col in row.index and d_col in row.index and a_col in row.index:
            h, d, a = row.get(h_col), row.get(d_col), row.get(a_col)
            if pd.notna(h) and pd.notna(d) and pd.notna(a):
                _, pd_norm, _ = normalize_odds(h, d, a)
                if np.isfinite(pd_norm):
                    draw_probs[label] = pd_norm

    close_draw_probs: Dict[str, float] = {}
    for h_col, d_col, a_col, label in CLOSING_TRIPLES:
        if h_col in row.index and d_col in row.index and a_col in row.index:
            h, d, a = row.get(h_col), row.get(d_col), row.get(a_col)
            if pd.notna(h) and pd.notna(d) and pd.notna(a):
                _, pd_norm, _ = normalize_odds(h, d, a)
                if np.isfinite(pd_norm):
                    close_draw_probs[label] = pd_norm

    # Exclude consensus books from individual book stats
    indiv_probs = {k: v for k, v in draw_probs.items()
                   if k not in ("Max", "Avg", "Max_close", "Avg_close")}

    all_probs = list(draw_probs.values())
    indiv_vals = list(indiv_probs.values())

    feat: Dict[str, float] = {}

    if all_probs:
        feat["consensus_draw_prob"]  = float(np.mean(all_probs))
        feat["max_draw_prob"]        = float(np.max(all_probs))
        feat["min_draw_prob"]        = float(np.min(all_probs))
        feat["std_draw_probs"]       = float(np.std(all_probs)) if len(all_probs) > 1 else 0.0
        feat["n_books"]              = float(len(all_probs))
        feat["max_minus_min"]        = feat["max_draw_prob"] - feat["min_draw_prob"]
        feat["log_consensus_draw"]   = float(np.log(max(feat["consensus_draw_prob"], 1e-6)))
    else:
        feat["consensus_draw_prob"]  = np.nan
        feat["max_draw_prob"]        = np.nan
        feat["min_draw_prob"]        = np.nan
        feat["std_draw_probs"]       = np.nan
        feat["n_books"]              = 0.0
        feat["max_minus_min"]        = np.nan
        feat["log_consensus_draw"]   = np.nan

    # Pinnacle (sharpest signal)
    feat["pinnacle_draw_prob"] = float(draw_probs.get("PS", np.nan))
    feat["pinnacle_close_prob"] = float(close_draw_probs.get("PS_close", np.nan))

    # Line movement: closing vs opening Pinnacle
    if "pinnacle_draw_prob" in feat and "pinnacle_close_prob" in feat:
        po, pc = feat["pinnacle_draw_prob"], feat["pinnacle_close_prob"]
        if np.isfinite(po) and np.isfinite(pc) and po > 0:
            feat["line_movement"] = float(pc - po)
            feat["line_movement_pct"] = float((pc - po) / po)
        else:
            feat["line_movement"]     = np.nan
            feat["line_movement_pct"] = np.nan
    else:
        feat["line_movement"]     = np.nan
        feat["line_movement_pct"] = np.nan

    # Disagreement among individual books
    if len(indiv_vals) >= 2:
        feat["inter_book_std"] = float(np.std(indiv_vals))
        feat["inter_book_range"] = float(max(indiv_vals) - min(indiv_vals))
    else:
        feat["inter_book_std"]   = np.nan
        feat["inter_book_range"] = np.nan

    # B365 normalized (for backward compatibility with mega model)
    if "B365" in draw_probs:
        feat["b365_draw_prob"] = float(draw_probs["B365"])
    else:
        feat["b365_draw_prob"] = np.nan

    return feat


# ── MarketModel class ─────────────────────────────────────────────────────────

class MarketModel:
    """
    Extracts and stores multi-bookmaker market features.

    The MarketModel acts as a feature transformer: it does not learn
    per se, but it tracks global statistics from training data for
    imputation purposes.
    """

    FEATURE_COLS = [
        "consensus_draw_prob", "max_draw_prob", "min_draw_prob",
        "std_draw_probs", "n_books", "max_minus_min",
        "log_consensus_draw", "pinnacle_draw_prob", "pinnacle_close_prob",
        "line_movement", "line_movement_pct",
        "inter_book_std", "inter_book_range", "b365_draw_prob",
    ]

    def __init__(self):
        self.train_medians_: Dict[str, float] = {}
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "MarketModel":
        """Compute training medians for imputation."""
        feats = self.transform(df, impute=False)
        self.train_medians_ = {
            col: float(feats[col].median())
            for col in self.FEATURE_COLS
            if col in feats.columns
        }
        self.fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        impute: bool = True,
    ) -> pd.DataFrame:
        """
        Transform a DataFrame of matches into market feature DataFrame.
        Returns DataFrame of shape (n_matches, n_features).
        """
        rows = []
        for _, row in df.iterrows():
            rows.append(extract_row_features(row))

        feat_df = pd.DataFrame(rows, index=df.index)

        # Ensure all expected columns present
        for col in self.FEATURE_COLS:
            if col not in feat_df.columns:
                feat_df[col] = np.nan

        # Impute with training medians
        if impute and self.train_medians_:
            for col, median_val in self.train_medians_.items():
                feat_df[col] = feat_df[col].fillna(median_val)

        return feat_df[self.FEATURE_COLS]

    def get_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return % of rows where each feature is non-null."""
        feats = self.transform(df, impute=False)
        return {
            col: float(feats[col].notna().mean())
            for col in self.FEATURE_COLS
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"    Market model saved → {path}")

    @staticmethod
    def load(path: str) -> "MarketModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Vectorised batch extraction (faster than row-by-row) ─────────────────────

def extract_multi_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast batch extraction of all market features.
    Handles missing columns gracefully.
    """
    out = pd.DataFrame(index=df.index)

    # Collect all book draw probabilities in one pass
    all_book_probs = {}
    for h_col, d_col, a_col, label in BOOKMAKER_TRIPLES + CLOSING_TRIPLES:
        if all(c in df.columns for c in [h_col, d_col, a_col]):
            h = pd.to_numeric(df[h_col], errors="coerce")
            d = pd.to_numeric(df[d_col], errors="coerce")
            a = pd.to_numeric(df[a_col], errors="coerce")
            total_imp = (1.0 / h) + (1.0 / d) + (1.0 / a)
            prob_d = (1.0 / d) / total_imp
            valid = (h > 1) & (d > 1) & (a > 1) & np.isfinite(total_imp)
            all_book_probs[label] = prob_d.where(valid)

    # Stack all opening probs
    open_labels  = [label for _, _, _, label in BOOKMAKER_TRIPLES]
    close_labels = [label for _, _, _, label in CLOSING_TRIPLES]

    open_probs  = pd.DataFrame({l: all_book_probs.get(l, pd.Series(np.nan, index=df.index))
                                 for l in open_labels}, index=df.index)
    close_probs = pd.DataFrame({l: all_book_probs.get(l, pd.Series(np.nan, index=df.index))
                                 for l in close_labels}, index=df.index)

    out["consensus_draw_prob"] = open_probs.mean(axis=1)
    out["max_draw_prob"]       = open_probs.max(axis=1)
    out["min_draw_prob"]       = open_probs.min(axis=1)
    out["std_draw_probs"]      = open_probs.std(axis=1)
    out["n_books"]             = open_probs.notna().sum(axis=1).astype(float)
    out["max_minus_min"]       = out["max_draw_prob"] - out["min_draw_prob"]
    out["log_consensus_draw"]  = np.log(out["consensus_draw_prob"].clip(lower=1e-6))

    out["pinnacle_draw_prob"]  = all_book_probs.get("PS", pd.Series(np.nan, index=df.index))
    out["pinnacle_close_prob"] = all_book_probs.get("PS_close", pd.Series(np.nan, index=df.index))

    # Line movement
    po = out["pinnacle_draw_prob"]
    pc = out["pinnacle_close_prob"]
    out["line_movement"]     = pc - po
    out["line_movement_pct"] = (pc - po) / po.clip(lower=1e-6)

    # Inter-book disagreement (only individual books, not Max/Avg)
    indiv_labels = [l for _, _, _, l in BOOKMAKER_TRIPLES
                    if l not in ("Max", "Avg")]
    indiv_df = open_probs[[l for l in indiv_labels if l in open_probs.columns]]
    out["inter_book_std"]   = indiv_df.std(axis=1)
    out["inter_book_range"] = indiv_df.max(axis=1) - indiv_df.min(axis=1)

    out["b365_draw_prob"] = all_book_probs.get("B365", pd.Series(np.nan, index=df.index))

    return out


# ── Standalone enrichment (merges market features with mega dataset) ──────────

def enrich_mega_dataset(
    mega_path: str = "data/processed/mega_dataset.parquet",
    raw_dir: str   = "data/raw",
    output_path: str = "data/processed/mega_with_market.parquet",
) -> pd.DataFrame:
    """
    Merge multi-bookmaker features into the mega dataset
    by joining on (Date, HomeTeam, AwayTeam).
    Only EFL rows will get the enriched market features.
    """
    print("  Loading mega dataset...")
    mega = pd.read_parquet(mega_path)

    print("  Loading EFL raw CSVs...")
    from src.referee_model import load_efl_raw
    efl_raw = load_efl_raw(raw_dir)

    print("  Extracting multi-book features from EFL raw data...")
    efl_raw["Date"] = pd.to_datetime(efl_raw["Date"])
    market_feats = extract_multi_book_features(efl_raw)
    efl_enriched = pd.concat([
        efl_raw[["Date", "HomeTeam", "AwayTeam", "Referee"]].copy(),
        market_feats,
    ], axis=1)

    print("  Merging into mega dataset...")
    mega["Date"] = pd.to_datetime(mega["Date"])
    merged = mega.merge(efl_enriched, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    cov = market_feats.notna().mean().mean()
    print(f"  Market feature coverage in EFL: {cov:.1%}")

    merged.to_parquet(output_path, index=False)
    print(f"  Saved enriched dataset → {output_path}  shape={merged.shape}")
    return merged


if __name__ == "__main__":
    from src.referee_model import load_efl_raw
    df = load_efl_raw()
    feats = extract_multi_book_features(df)
    print("Market feature coverage:")
    for col in feats.columns:
        print(f"  {col:<30}  {feats[col].notna().mean():.1%}")
    print(feats.head(3))
