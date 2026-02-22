"""
Build American Leagues Data for Mega Dataset V2
================================================
Downloads and normalises MLS (USA) and Liga MX (Mexico) data from
football-data.co.uk new format into the standard mega dataset schema.

Key differences vs European mmz4281 format:
  - Home/Away instead of HomeTeam/AwayTeam
  - HG/AG = full-time goals (NO half-time scores available)
  - B365CH/D/A = B365 closing (only from ~2025); use PSCH/D/A (Pinnacle) instead
  - MaxCH/D/A and AvgCH/D/A available throughout
  - HTHG/HTAG are set to NaN (not available from this source)

The resulting rows are appended to the existing mega dataset.
Elo will update on FT results as proxy; Dixon-Coles will fit per-league on FT goals.
"""

import pandas as pd
import numpy as np
from pathlib import Path


AMERICAN_LEAGUES = {
    "USA_MLS": {
        "path":    "data/raw_all/USA_MLS/all_seasons.csv",
        "league":  "USA_MLS",
        "country": "USA",
        "name":    "MLS",
        "tier":    1,
    },
    "MEX_LigaMX": {
        "path":    "data/raw_all/MEX_LigaMX/all_seasons.csv",
        "league":  "MEX_LigaMX",
        "country": "Mexico",
        "name":    "Liga MX",
        "tier":    1,
    },
}


def load_american_league(cfg: dict) -> pd.DataFrame:
    """Load and normalise one American league CSV into mega dataset schema."""
    path = Path(cfg["path"])
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  Loaded {len(df):,} rows from {path.name}")

    # ── Column renames ─────────────────────────────────────────────────────────
    rename = {
        "Home":   "HomeTeam",
        "Away":   "AwayTeam",
        "HG":     "FTHG",
        "AG":     "FTAG",
        # Pinnacle closing → standard names used by market_model
        "PSCH":   "PSCH",
        "PSCD":   "PSCD",
        "PSCA":   "PSCA",
        # Max/Avg closing
        "MaxCH":  "MaxCH",
        "MaxCD":  "MaxCD",
        "MaxCA":  "MaxCA",
        "AvgCH":  "AvgCH",
        "AvgCD":  "AvgCD",
        "AvgCA":  "AvgCA",
        # Betfair Exchange closing
        "BFECH":  "BFECH",
        "BFECD":  "BFECD",
        "BFECA":  "BFECA",
    }

    # Handle B365 closing (USA has typo B36CA instead of B365CA)
    if "B365CH" in df.columns:
        rename["B365CH"] = "B365H"
    if "B365CD" in df.columns:
        rename["B365CD"] = "B365D"
    # USA typo
    if "B36CA" in df.columns and "B365CA" not in df.columns:
        rename["B36CA"] = "B365A"
    elif "B365CA" in df.columns:
        rename["B365CA"] = "B365A"

    df = df.rename(columns=rename)

    # ── Parse dates ────────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    # ── Require FT goals ───────────────────────────────────────────────────────
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    # ── HT scores: not available ───────────────────────────────────────────────
    df["HTHG"] = np.nan
    df["HTAG"] = np.nan

    # ── Metadata ───────────────────────────────────────────────────────────────
    df["league"]      = cfg["league"]
    df["country"]     = cfg["country"]
    df["league_tier"] = cfg["tier"]
    df["league_name"] = cfg["name"]
    df["season"]      = df.get("Season", pd.Series("unknown", index=df.index)).astype(str)
    df["ft_only"]     = True   # flag: no HT data available

    # Numeric odds columns
    for col in ["PSCH", "PSCD", "PSCA", "MaxCH", "MaxCD", "MaxCA",
                "AvgCH", "AvgCD", "AvgCA", "BFECH", "BFECD", "BFECA",
                "B365H", "B365D", "B365A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Select only columns present in both datasets ───────────────────────────
    keep = [
        "Date", "HomeTeam", "AwayTeam",
        "HTHG", "HTAG",        # NaN for American leagues
        "FTHG", "FTAG",
        "B365H", "B365D", "B365A",       # mostly NaN pre-2025
        "PSCH", "PSCD", "PSCA",          # Pinnacle closing (main odds signal)
        "MaxCH", "MaxCD", "MaxCA",
        "AvgCH", "AvgCD", "AvgCA",
        "BFECH", "BFECD", "BFECA",
        "league", "season", "country", "league_tier", "league_name", "ft_only",
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()

    # Fill missing B365 from AvgC when B365 is NaN (for market model fallback)
    for side, avg_col in [("B365H", "AvgCH"), ("B365D", "AvgCD"), ("B365A", "AvgCA")]:
        if side not in df.columns and avg_col in df.columns:
            df[side] = df[avg_col]
        elif side in df.columns and avg_col in df.columns:
            df[side] = df[side].fillna(df[avg_col])

    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  → {len(df):,} valid matches  ({df['Date'].min().date()} to {df['Date'].max().date()})")
    return df


def build_american_dataframe() -> pd.DataFrame:
    """Return a combined American leagues DataFrame."""
    dfs = []
    for key, cfg in AMERICAN_LEAGUES.items():
        print(f"\nProcessing {key}...")
        sub = load_american_league(cfg)
        if not sub.empty:
            dfs.append(sub)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    print(f"\nTotal American rows: {len(combined):,}")
    return combined


def build_mega_v2(
    existing_parquet: str = "data/processed/mega_dataset.parquet",
    output_parquet:   str = "data/processed/mega_dataset_v2.parquet",
) -> pd.DataFrame:
    """Append American leagues to the existing mega dataset and save v2."""

    # Load existing
    print(f"Loading existing mega dataset from {existing_parquet}...")
    existing = pd.read_parquet(existing_parquet)
    existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
    print(f"  Existing rows: {len(existing):,}")

    # Mark existing rows as having HT data
    if "ft_only" not in existing.columns:
        existing["ft_only"] = False

    # Load American data
    american = build_american_dataframe()
    if american.empty:
        print("No American data loaded — saving existing as v2.")
        existing.to_parquet(output_parquet, index=False)
        return existing

    # Align columns: add any missing cols as NaN
    all_cols = list(dict.fromkeys(list(existing.columns) + list(american.columns)))
    for col in all_cols:
        if col not in existing.columns:
            existing[col] = np.nan
        if col not in american.columns:
            american[col] = np.nan

    # Combine
    combined = pd.concat([existing[all_cols], american[all_cols]], ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)

    # Refresh league_encoded and country_encoded
    from sklearn.preprocessing import LabelEncoder
    le_l = LabelEncoder()
    le_c = LabelEncoder()
    combined["league_encoded"]  = le_l.fit_transform(combined["league"].fillna("unknown"))
    combined["country_encoded"] = le_c.fit_transform(combined["country"].fillna("unknown"))

    print(f"\nCombined dataset: {len(combined):,} rows")
    print(f"Leagues: {sorted(combined['league'].unique())}")
    print(f"Countries: {sorted(combined['country'].unique())}")
    ht_rows = combined["HTHG"].notna().sum()
    ft_only = combined["ft_only"].eq(True).sum()
    print(f"Rows with HT data: {ht_rows:,}  |  FT-only rows: {ft_only:,}")

    combined.to_parquet(output_parquet, index=False)
    print(f"\nSaved → {output_parquet}")
    return combined


if __name__ == "__main__":
    df = build_mega_v2()
    print(f"\nDone. {len(df):,} total matches in mega_dataset_v2.parquet")
