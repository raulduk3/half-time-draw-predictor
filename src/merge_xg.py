"""
Merge FBref xG data into mega dataset.
=======================================
Reads all fbref_{league}_{season}.csv files from data/xg/,
fuzzy-matches by (home_team, away_team, date) with team name normalization,
then computes rolling-5 xG features per team.

Usage:
    python src/merge_xg.py

Reads:  data/xg/fbref_*.csv
        data/processed/mega_dataset_v2.parquet
Writes: data/processed/mega_dataset_v3.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from unidecode import unidecode
import re

# ── Team name mapping: FBref → football-data.co.uk ───────────────────────────

TEAM_MAP = {
    # EPL
    "Manchester Utd":          "Man United",
    "Manchester United":       "Man United",
    "Manchester City":         "Man City",
    "Tottenham":               "Tottenham",
    "Tottenham Hotspur":       "Tottenham",
    "Newcastle Utd":           "Newcastle",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves":                  "Wolves",
    "West Ham":                "West Ham",
    "West Ham United":         "West Ham",
    "Leicester City":          "Leicester",
    "Leicester":               "Leicester",
    "Leeds United":            "Leeds",
    "Leeds":                   "Leeds",
    "Sheffield Utd":           "Sheffield United",
    "Sheffield United":        "Sheffield United",
    "West Brom":               "West Brom",
    "West Bromwich Albion":    "West Brom",
    "Nott'ham Forest":         "Nott'm Forest",
    "Nottingham Forest":       "Nott'm Forest",
    "Nottm Forest":            "Nott'm Forest",
    "Ipswich Town":            "Ipswich",
    "Ipswich":                 "Ipswich",
    "Bournemouth":             "Bournemouth",
    "Brighton":                "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Huddersfield":            "Huddersfield",
    "Huddersfield Town":       "Huddersfield",
    "Cardiff City":            "Cardiff",
    "Norwich City":            "Norwich",
    "Watford":                 "Watford",
    "Luton Town":              "Luton",
    "Luton":                   "Luton",
    "Burnley":                 "Burnley",
    "QPR":                     "QPR",
    "Queens Park Rangers":     "QPR",
    "Fulham":                  "Fulham",
    "Crystal Palace":          "Crystal Palace",
    "Brentford":               "Brentford",
    "Arsenal":                 "Arsenal",
    "Chelsea":                 "Chelsea",
    "Liverpool":               "Liverpool",
    "Everton":                 "Everton",
    "Southampton":             "Southampton",
    "Aston Villa":             "Aston Villa",
    "Sunderland":              "Sunderland",
    "Stoke City":              "Stoke",
    "Swansea City":            "Swansea",

    # La Liga
    "Atlético Madrid":         "Ath Madrid",
    "Atletico Madrid":         "Ath Madrid",
    "Athletic Club":           "Ath Bilbao",
    "Betis":                   "Betis",
    "Real Betis":              "Betis",
    "Celta Vigo":              "Celta",
    "Celta":                   "Celta",
    "Deportivo Alavés":        "Alaves",
    "Alavés":                  "Alaves",
    "Alaves":                  "Alaves",
    "Rayo Vallecano":          "Vallecano",
    "Real Sociedad":           "Sociedad",
    "Sociedad":                "Sociedad",
    "Real Valladolid":         "Valladolid",
    "Valladolid":              "Valladolid",
    "Eibar":                   "Eibar",
    "Huesca":                  "Huesca",
    "Las Palmas":              "Las Palmas",
    "Leganés":                 "Leganes",
    "Leganes":                 "Leganes",
    "RCD Mallorca":            "Mallorca",
    "Mallorca":                "Mallorca",
    "Cádiz":                   "Cadiz",
    "Cadiz":                   "Cadiz",
    "Elche":                   "Elche",
    "Espanyol":                "Espanol",
    "RCD Espanyol":            "Espanol",
    "Almería":                 "Almeria",
    "Almeria":                 "Almeria",
    "Girona":                  "Girona",
    "Getafe":                  "Getafe",
    "Osasuna":                 "Osasuna",
    "Granada":                 "Granada",
    "Sevilla":                 "Sevilla",
    "Real Madrid":             "Real Madrid",
    "Barcelona":               "Barcelona",
    "Valencia":                "Valencia",
    "Villarreal":              "Villarreal",

    # Serie A
    "AC Milan":                "Milan",
    "Inter":                   "Inter",
    "Hellas Verona":           "Verona",
    "SPAL":                    "Spal",
    "Parma":                   "Parma",
    "Benevento":               "Benevento",
    "Crotone":                 "Crotone",
    "Frosinone":               "Frosinone",
    "Brescia":                 "Brescia",
    "Lecce":                   "Lecce",
    "Spezia":                  "Spezia",
    "Salernitana":             "Salernitana",
    "Venezia":                 "Venezia",
    "Monza":                   "Monza",
    "Cremonese":               "Cremonese",
    "Empoli":                  "Empoli",
    "Udinese":                 "Udinese",
    "Bologna":                 "Bologna",
    "Torino":                  "Torino",
    "Fiorentina":              "Fiorentina",
    "Napoli":                  "Napoli",
    "Roma":                    "Roma",
    "Lazio":                   "Lazio",
    "Juventus":                "Juventus",
    "Atalanta":                "Atalanta",
    "Cagliari":                "Cagliari",
    "Sassuolo":                "Sassuolo",
    "Genoa":                   "Genoa",
    "Sampdoria":               "Sampdoria",

    # Bundesliga
    "Bayer Leverkusen":        "Leverkusen",
    "Leverkusen":              "Leverkusen",
    "Bayern Munich":           "Bayern Munich",
    "Borussia Dortmund":       "Dortmund",
    "Dortmund":                "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Mönchengladbach":         "M'gladbach",
    "RB Leipzig":              "RB Leipzig",
    "Eintracht Frankfurt":     "Ein Frankfurt",
    "Frankfurt":               "Ein Frankfurt",
    "Hertha BSC":              "Hertha",
    "Hertha":                  "Hertha",
    "Arminia Bielefeld":       "Bielefeld",
    "Bielefeld":               "Bielefeld",
    "Greuther Fürth":          "Greuther Furth",
    "VfL Bochum":              "Bochum",
    "Bochum":                  "Bochum",
    "Köln":                    "FC Koln",
    "1. FC Köln":              "FC Koln",
    "FC Köln":                 "FC Koln",
    "Paderborn":               "Paderborn",
    "Hamburger SV":            "Hamburg",
    "Darmstadt 98":            "Darmstadt",
    "1. FC Heidenheim 1846":   "Heidenheim",
    "Heidenheim":              "Heidenheim",
    "FC St. Pauli":            "St Pauli",
    "St. Pauli":               "St Pauli",
    "Holstein Kiel":           "Holstein Kiel",
    "Hannover 96":             "Hannover",
    "Mainz 05":                "Mainz",
    "Mainz":                   "Mainz",
    "1. FSV Mainz 05":         "Mainz",
    "Nürnberg":                "Nurnberg",
    "VfB Stuttgart":           "Stuttgart",
    "Stuttgart":               "Stuttgart",
    "Wolfsburg":               "Wolfsburg",
    "VfL Wolfsburg":           "Wolfsburg",
    "Augsburg":                "Augsburg",
    "FC Augsburg":             "Augsburg",
    "Union Berlin":            "Union Berlin",
    "1. FC Union Berlin":      "Union Berlin",
    "Freiburg":                "Freiburg",
    "SC Freiburg":             "Freiburg",
    "Hoffenheim":              "Hoffenheim",
    "TSG Hoffenheim":          "Hoffenheim",

    # Ligue 1
    "Paris S-G":               "Paris SG",
    "Paris Saint-Germain":     "Paris SG",
    "Lyon":                    "Lyon",
    "Olympique Lyonnais":      "Lyon",
    "Marseille":               "Marseille",
    "Olympique de Marseille":  "Marseille",
    "Monaco":                  "Monaco",
    "AS Monaco":               "Monaco",
    "Saint-Étienne":           "St Etienne",
    "Saint-Etienne":           "St Etienne",
    "Rennes":                  "Rennes",
    "Stade Rennais":           "Rennes",
    "Strasbourg":              "Strasbourg",
    "Reims":                   "Reims",
    "Lens":                    "Lens",
    "Brest":                   "Brest",
    "Lorient":                 "Lorient",
    "Clermont Foot":           "Clermont",
    "Troyes":                  "Troyes",
    "Metz":                    "Metz",
    "Nantes":                  "Nantes",
    "Montpellier":             "Montpellier",
    "Nice":                    "Nice",
    "OGC Nice":                "Nice",
    "Angers":                  "Angers",
    "Dijon":                   "Dijon",
    "Amiens":                  "Amiens",
    "Le Havre":                "Le Havre",
    "Auxerre":                 "Auxerre",
    "Toulouse":                "Toulouse",
    "Lille":                   "Lille",
    "Bordeaux":                "Bordeaux",
}


def normalize_name(name: str) -> str:
    """Normalize team name for fuzzy matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = TEAM_MAP.get(name, name)
    name = unidecode(name).lower().strip()
    # Only strip generic non-distinguishing suffixes
    for suffix in [" fc", " cf", " sc", " afc"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return name.strip()


def load_fbref_data(xg_dir: str = "data/xg") -> pd.DataFrame:
    """Load and combine all FBref CSVs from data/xg/."""
    xg_path = Path(xg_dir)
    csv_files = sorted(xg_path.glob("fbref_*.csv"))
    # Exclude the combined file
    csv_files = [f for f in csv_files if f.name != "fbref_all_xg.csv"]

    if not csv_files:
        raise FileNotFoundError(
            f"No fbref_*.csv files found in {xg_dir}. "
            "Run src/fbref_scraper.py first."
        )

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(csv_files)} FBref CSV files → {len(combined):,} rows")
    return combined


def compute_rolling_xg(
    mega: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add per-team rolling-5 xG features to mega dataset.

    Requires columns: Date, HomeTeam, AwayTeam, home_xg, away_xg.
    Optionally uses home_1h_xg, away_1h_xg if present.

    Adds columns:
        home_xg_r5, home_xga_r5, home_xg_diff_r5
        away_xg_r5, away_xga_r5, away_xg_diff_r5
        home_1h_xg_r5, home_1h_xga_r5   (if 1h data available)
        away_1h_xg_r5, away_1h_xga_r5   (if 1h data available)
        home_goals_minus_xg_r5, away_goals_minus_xg_r5
    """
    df = mega.sort_values("Date").reset_index(drop=True)

    has_1h = "home_1h_xg" in df.columns and df["home_1h_xg"].notna().mean() > 0.01

    # Allocate output columns
    new_cols = [
        "home_xg_r5", "home_xga_r5", "home_xg_diff_r5",
        "away_xg_r5", "away_xga_r5", "away_xg_diff_r5",
        "home_goals_minus_xg_r5", "away_goals_minus_xg_r5",
    ]
    if has_1h:
        new_cols += [
            "home_1h_xg_r5", "home_1h_xga_r5",
            "away_1h_xg_r5", "away_1h_xga_r5",
        ]

    for col in new_cols:
        df[col] = np.nan

    # Build per-team history: date → (xg_for, xg_against, 1h_xg_for, 1h_xg_against,
    #                                  goals_for, goals_against)
    # We iterate chronologically and use the last `window` rows before current match.
    from collections import defaultdict
    team_hist: dict = defaultdict(list)  # team → list of dicts

    print(f"  Computing rolling xG features (window={window}) for {len(df):,} matches...")

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]
        h_xg  = row.get("home_xg",  np.nan)
        a_xg  = row.get("away_xg",  np.nan)
        h1_xg = row.get("home_1h_xg", np.nan) if has_1h else np.nan
        a1_xg = row.get("away_1h_xg", np.nan) if has_1h else np.nan
        h_g   = row.get("FTHG", np.nan)
        a_g   = row.get("FTAG", np.nan)

        def _rolling_mean(records, key, last_n=window):
            vals = [r[key] for r in records[-last_n:] if not np.isnan(r.get(key, np.nan))]
            return float(np.mean(vals)) if vals else np.nan

        # ── Home team stats ────────────────────────────────────────────────────
        h_hist = team_hist[home]
        if h_hist:
            df.at[idx, "home_xg_r5"]  = _rolling_mean(h_hist, "xg_for")
            df.at[idx, "home_xga_r5"] = _rolling_mean(h_hist, "xg_against")
            xg_f = _rolling_mean(h_hist, "xg_for")
            xg_a = _rolling_mean(h_hist, "xg_against")
            if not np.isnan(xg_f) and not np.isnan(xg_a):
                df.at[idx, "home_xg_diff_r5"] = xg_f - xg_a
            gf = _rolling_mean(h_hist, "goals_for")
            if not np.isnan(gf) and not np.isnan(xg_f):
                df.at[idx, "home_goals_minus_xg_r5"] = gf - xg_f
            if has_1h:
                df.at[idx, "home_1h_xg_r5"]  = _rolling_mean(h_hist, "xg_1h_for")
                df.at[idx, "home_1h_xga_r5"] = _rolling_mean(h_hist, "xg_1h_against")

        # ── Away team stats ────────────────────────────────────────────────────
        a_hist = team_hist[away]
        if a_hist:
            df.at[idx, "away_xg_r5"]  = _rolling_mean(a_hist, "xg_for")
            df.at[idx, "away_xga_r5"] = _rolling_mean(a_hist, "xg_against")
            xg_f = _rolling_mean(a_hist, "xg_for")
            xg_a = _rolling_mean(a_hist, "xg_against")
            if not np.isnan(xg_f) and not np.isnan(xg_a):
                df.at[idx, "away_xg_diff_r5"] = xg_f - xg_a
            gf = _rolling_mean(a_hist, "goals_for")
            if not np.isnan(gf) and not np.isnan(xg_f):
                df.at[idx, "away_goals_minus_xg_r5"] = gf - xg_f
            if has_1h:
                df.at[idx, "away_1h_xg_r5"]  = _rolling_mean(a_hist, "xg_1h_for")
                df.at[idx, "away_1h_xga_r5"] = _rolling_mean(a_hist, "xg_1h_against")

        # ── Update history after using it (no look-ahead) ─────────────────────
        team_hist[home].append({
            "date":        date,
            "xg_for":      h_xg,
            "xg_against":  a_xg,
            "xg_1h_for":   h1_xg,
            "xg_1h_against": a1_xg,
            "goals_for":   h_g,
            "goals_against": a_g,
        })
        team_hist[away].append({
            "date":        date,
            "xg_for":      a_xg,
            "xg_against":  h_xg,
            "xg_1h_for":   a1_xg,
            "xg_1h_against": h1_xg,
            "goals_for":   a_g,
            "goals_against": h_g,
        })

        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx:,} / {len(df):,} processed...")

    print("  Done computing rolling xG features.")
    return df


def merge_xg_data(
    mega_path: str = "data/processed/mega_dataset_v2.parquet",
    xg_dir: str = "data/xg",
    output_path: str = "data/processed/mega_dataset_v3.parquet",
) -> pd.DataFrame:
    """Merge FBref xG data into mega dataset and compute rolling features."""

    print("Loading mega dataset...")
    mega = pd.read_parquet(mega_path)
    mega["Date"] = pd.to_datetime(mega["Date"])
    print(f"  Mega: {len(mega):,} matches")

    print("Loading FBref xG data...")
    xg = load_fbref_data(xg_dir)

    # Normalize team names for matching
    xg["date"] = pd.to_datetime(xg["date"], errors="coerce")
    xg = xg.dropna(subset=["date", "home_team", "away_team"])

    mega["_home_norm"] = mega["HomeTeam"].apply(normalize_name)
    mega["_away_norm"] = mega["AwayTeam"].apply(normalize_name)
    mega["_date"]      = mega["Date"].dt.date

    xg["_home_norm"] = xg["home_team"].apply(normalize_name)
    xg["_away_norm"] = xg["away_team"].apply(normalize_name)
    xg["_date"]      = xg["date"].dt.date

    # Build lookup dict from xG data
    xg_lookup = {}
    for _, row in xg.iterrows():
        key = (row["_home_norm"], row["_away_norm"], row["_date"])
        xg_lookup[key] = {
            "home_xg":    row.get("home_xg"),
            "away_xg":    row.get("away_xg"),
            "home_1h_xg": row.get("home_1h_xg"),
            "away_1h_xg": row.get("away_1h_xg"),
        }
    print(f"  xG lookup keys: {len(xg_lookup):,}")

    # Match and merge
    matched = 0
    unmatched = 0
    home_xg_vals, away_xg_vals = [], []
    home_1h_vals, away_1h_vals = [], []

    from datetime import timedelta

    for _, row in mega.iterrows():
        key = (row["_home_norm"], row["_away_norm"], row["_date"])

        found_data = None
        if key in xg_lookup:
            found_data = xg_lookup[key]
            matched += 1
        else:
            for delta in [timedelta(days=1), timedelta(days=-1)]:
                alt_key = (row["_home_norm"], row["_away_norm"], row["_date"] + delta)
                if alt_key in xg_lookup:
                    found_data = xg_lookup[alt_key]
                    matched += 1
                    break
            if found_data is None:
                unmatched += 1

        if found_data:
            home_xg_vals.append(found_data.get("home_xg"))
            away_xg_vals.append(found_data.get("away_xg"))
            home_1h_vals.append(found_data.get("home_1h_xg"))
            away_1h_vals.append(found_data.get("away_1h_xg"))
        else:
            home_xg_vals.append(np.nan)
            away_xg_vals.append(np.nan)
            home_1h_vals.append(np.nan)
            away_1h_vals.append(np.nan)

    mega["home_xg"]    = home_xg_vals
    mega["away_xg"]    = away_xg_vals
    mega["home_1h_xg"] = home_1h_vals
    mega["away_1h_xg"] = away_1h_vals

    # Derived match-level xG features
    mega["home_xg_diff"] = mega["home_xg"] - mega["away_xg"]
    mega["away_xg_diff"] = mega["away_xg"] - mega["home_xg"]
    mega["total_xg"]     = mega["home_xg"] + mega["away_xg"]

    # Clean up temp columns
    mega.drop(columns=["_home_norm", "_away_norm", "_date"], inplace=True)

    # Stats
    xg_coverage = mega["home_xg"].notna().sum()
    print(f"\n{'='*50}")
    print(f"  Matched: {matched:,} ({matched/len(mega):.1%})")
    print(f"  Unmatched: {unmatched:,}")
    print(f"  xG coverage: {xg_coverage:,} ({xg_coverage/len(mega):.1%})")

    if "league" in mega.columns:
        print(f"\n  Per-league xG coverage:")
        for league in sorted(mega["league"].unique()):
            league_df = mega[mega["league"] == league]
            cov = league_df["home_xg"].notna().sum()
            if cov > 0:
                print(f"    {league}: {cov}/{len(league_df)} ({cov/len(league_df):.1%})")

    # Compute rolling xG features
    print(f"\nComputing rolling xG features...")
    mega = compute_rolling_xg(mega, window=5)

    # Report rolling feature coverage
    rolling_cols = [c for c in mega.columns if c.endswith("_r5") and "xg" in c]
    print(f"\n  Rolling xG feature coverage:")
    for col in rolling_cols:
        cov = mega[col].notna().sum()
        print(f"    {col}: {cov:,} ({cov/len(mega):.1%})")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mega.to_parquet(output_path, index=False)
    print(f"\n  Saved to {output_path}")
    print(f"  Columns: {len(mega.columns)} (+{len(rolling_cols)+4} xG columns)")
    print(f"{'='*50}")

    return mega


if __name__ == "__main__":
    merge_xg_data()
