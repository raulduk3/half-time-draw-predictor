"""
Merge Understat xG data into mega dataset.
==========================================
Fuzzy-matches by (home_team, away_team, date) with team name normalization.

Usage:
    python src/merge_xg.py
    
Reads:  data/xg/understat_all_xg.csv
        data/processed/mega_dataset_v2.parquet
Writes: data/processed/mega_dataset_v3.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from unidecode import unidecode
import re

# ── Team name mapping: Understat → football-data.co.uk ────────────────────────

TEAM_MAP = {
    # EPL
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham": "Tottenham",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham": "West Ham",
    "Leicester": "Leicester",
    "Leeds": "Leeds",
    "Sheffield United": "Sheffield United",
    "West Bromwich Albion": "West Brom",
    "Nottingham Forest": "Nott'm Forest",
    "Ipswich": "Ipswich",
    "Bournemouth": "Bournemouth",
    "Brighton": "Brighton",
    "Huddersfield": "Huddersfield",
    "Cardiff": "Cardiff",
    "Norwich": "Norwich",
    "Watford": "Watford",
    "Luton": "Luton",
    "Burnley": "Burnley",
    
    # La Liga
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Deportivo La Coruna": "La Coruna",
    "Deportivo Alaves": "Alaves",
    "Rayo Vallecano": "Vallecano",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "SD Eibar": "Eibar",
    "SD Huesca": "Huesca",
    "Sporting Gijon": "Sp Gijon",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "RCD Mallorca": "Mallorca",
    "Cadiz": "Cadiz",
    "Elche": "Elche",
    
    # Serie A
    "AC Milan": "Milan",
    "Inter": "Inter",
    "Hellas Verona": "Verona",
    "SPAL 2013": "Spal",
    "Parma Calcio 1913": "Parma",
    "Benevento": "Benevento",
    "ChievoVerona": "Chievo",
    "Crotone": "Crotone",
    "Pescara": "Pescara",
    "Frosinone": "Frosinone",
    "Brescia": "Brescia",
    "Lecce": "Lecce",
    "Spezia": "Spezia",
    "Salernitana": "Salernitana",
    "Venezia": "Venezia",
    "Monza": "Monza",
    "Cremonese": "Cremonese",
    
    # Bundesliga
    "Bayer Leverkusen": "Leverkusen",
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Hertha Berlin": "Hertha",
    "Fortuna Dusseldorf": "Fortuna Dusseldorf",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "Arminia Bielefeld": "Bielefeld",
    "Greuther Fuerth": "Greuther Furth",
    "VfL Bochum": "Bochum",
    "FC Cologne": "FC Koln",
    "SC Paderborn 07": "Paderborn",
    "Hamburger SV": "Hamburg",
    "SV Darmstadt 98": "Darmstadt",
    "1. FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "St Pauli",
    "St. Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "1. FC Heidenheim": "Heidenheim",
    "FC Heidenheim": "Heidenheim",
    
    # Bundesliga continued
    "Hannover 96": "Hannover",
    "Mainz 05": "Mainz",
    "Nuernberg": "Nurnberg",
    "VfB Stuttgart": "Stuttgart",
    
    # EPL continued
    "Queens Park Rangers": "QPR",
    "Nottingham Forest": "Nott'm Forest",
    
    # La Liga continued
    "Espanyol": "Espanol",
    
    # Serie A continued
    "SPAL 2013": "Spal",
    "Parma Calcio 1913": "Parma",
    
    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "GFC Ajaccio": "Ajaccio GFCO",
    "SC Bastia": "Bastia",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "AS Monaco": "Monaco",
    "Saint-Etienne": "St Etienne",
    "Stade Rennais": "Rennes",
    "Racing Club de Strasbourg": "Strasbourg",
    "Stade de Reims": "Reims",
    "RC Lens": "Lens",
    "Stade Brestois 29": "Brest",
    "FC Lorient": "Lorient",
    "Clermont Foot": "Clermont",
    "Troyes": "Troyes",
    "Metz": "Metz",
    "FC Nantes": "Nantes",
    "Montpellier": "Montpellier",
    "OGC Nice": "Nice",
    "Angers": "Angers",
    "Dijon": "Dijon",
    "Amiens": "Amiens",
    "Le Havre": "Le Havre",
    "AJ Auxerre": "Auxerre",
    "Toulouse": "Toulouse",
}

# League mapping: Understat → our codes
LEAGUE_MAP = {
    "ENG-Premier League": "E0",
    "ESP-La Liga": "SP1",
    "ITA-Serie A": "I1",
    "GER-Bundesliga": "D1",
    "FRA-Ligue 1": "F1",
}


def normalize_name(name: str) -> str:
    """Normalize team name for fuzzy matching."""
    if pd.isna(name):
        return ""
    name = TEAM_MAP.get(name, name)
    name = unidecode(str(name)).lower().strip()
    # Remove common suffixes
    for suffix in [" fc", " cf", " sc", " afc"]:
        name = name.replace(suffix, "")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return name.strip()


def merge_xg_data(
    mega_path: str = "data/processed/mega_dataset_v2.parquet",
    xg_path: str = "data/xg/understat_all_xg.csv",
    output_path: str = "data/processed/mega_dataset_v3.parquet",
) -> pd.DataFrame:
    """Merge xG data into mega dataset."""
    
    print("Loading mega dataset...")
    mega = pd.read_parquet(mega_path)
    mega["Date"] = pd.to_datetime(mega["Date"])
    print(f"  Mega: {len(mega):,} matches")
    
    print("Loading xG data...")
    xg = pd.read_csv(xg_path)
    xg["date"] = pd.to_datetime(xg["date"])
    print(f"  xG: {len(xg):,} matches")
    
    # Normalize team names for matching
    mega["_home_norm"] = mega["HomeTeam"].apply(normalize_name)
    mega["_away_norm"] = mega["AwayTeam"].apply(normalize_name)
    mega["_date"] = mega["Date"].dt.date
    
    xg["_home_norm"] = xg["home_team"].apply(normalize_name)
    xg["_away_norm"] = xg["away_team"].apply(normalize_name)
    xg["_date"] = xg["date"].dt.date
    
    # Build lookup dict from xG data
    xg_lookup = {}
    for _, row in xg.iterrows():
        key = (row["_home_norm"], row["_away_norm"], row["_date"])
        xg_lookup[key] = {
            "understat_home_xg": row["home_xg"],
            "understat_away_xg": row["away_xg"],
            "understat_game_id": row.get("game_id"),
        }
    
    print(f"  xG lookup keys: {len(xg_lookup):,}")
    
    # Match and merge
    matched = 0
    unmatched = 0
    
    home_xg = []
    away_xg = []
    
    for _, row in mega.iterrows():
        key = (row["_home_norm"], row["_away_norm"], row["_date"])
        
        if key in xg_lookup:
            data = xg_lookup[key]
            home_xg.append(data["understat_home_xg"])
            away_xg.append(data["understat_away_xg"])
            matched += 1
        else:
            # Try date ±1 day (timezone mismatches)
            from datetime import timedelta
            found = False
            for delta in [timedelta(days=1), timedelta(days=-1)]:
                alt_key = (row["_home_norm"], row["_away_norm"], row["_date"] + delta)
                if alt_key in xg_lookup:
                    data = xg_lookup[alt_key]
                    home_xg.append(data["understat_home_xg"])
                    away_xg.append(data["understat_away_xg"])
                    matched += 1
                    found = True
                    break
            if not found:
                home_xg.append(np.nan)
                away_xg.append(np.nan)
                unmatched += 1
    
    mega["home_xg"] = home_xg
    mega["away_xg"] = away_xg
    
    # Derived features
    mega["home_xg_diff"] = mega["home_xg"] - mega["away_xg"]
    mega["away_xg_diff"] = mega["away_xg"] - mega["home_xg"]
    mega["total_xg"] = mega["home_xg"] + mega["away_xg"]
    mega["xg_draw_proxy"] = 1.0 / (1.0 + abs(mega["home_xg"] - mega["away_xg"]))
    
    # Clean up temp columns
    mega.drop(columns=["_home_norm", "_away_norm", "_date"], inplace=True)
    
    # Stats
    xg_coverage = mega["home_xg"].notna().sum()
    print(f"\n{'='*50}")
    print(f"  Matched: {matched:,} ({matched/len(mega):.1%})")
    print(f"  Unmatched: {unmatched:,}")
    print(f"  xG coverage: {xg_coverage:,} ({xg_coverage/len(mega):.1%})")
    
    # Per-league coverage
    print(f"\n  Per-league xG coverage:")
    for league in sorted(mega["league"].unique()):
        league_df = mega[mega["league"] == league]
        cov = league_df["home_xg"].notna().sum()
        print(f"    {league}: {cov}/{len(league_df)} ({cov/len(league_df):.1%})")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mega.to_parquet(output_path, index=False)
    print(f"\n  Saved to {output_path}")
    print(f"  Columns: {len(mega.columns)} (was {len(mega.columns) - 5} + 5 new xG)")
    print(f"{'='*50}")
    
    return mega


if __name__ == "__main__":
    merge_xg_data()
