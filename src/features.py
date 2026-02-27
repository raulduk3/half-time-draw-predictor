"""
Feature engineering utilities for soccer half-time draw prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def compute_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute rolling form statistics for each team.
    
    Args:
        df: DataFrame with match data sorted by date
        window: Rolling window size (default: 5 matches)
    
    Returns:
        DataFrame with rolling form features added
    """
    df_copy = df.copy()
    
    # Ensure data is sorted by date
    df_copy = df_copy.sort_values('Date').reset_index(drop=True)
    
    # Initialize rolling form columns
    form_cols = [
        'home_gf_r5', 'home_ga_r5', 'home_gd_r5',
        'away_gf_r5', 'away_ga_r5', 'away_gd_r5'
    ]
    
    for col in form_cols:
        df_copy[col] = np.nan
    
    # Get all unique teams (excluding NaN values)
    home_teams = set(df_copy['HomeTeam'].dropna().unique())
    away_teams = set(df_copy['AwayTeam'].dropna().unique())
    all_teams = sorted(home_teams | away_teams)
    
    print(f"Computing rolling form for {len(all_teams)} teams over {len(df_copy)} matches...")
    
    # For each match, compute rolling form features
    for idx, row in df_copy.iterrows():
        match_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get historical matches for home team (before current match)
        home_history = df_copy[
            (df_copy['Date'] < match_date) & 
            ((df_copy['HomeTeam'] == home_team) | (df_copy['AwayTeam'] == home_team))
        ].tail(window)
        
        # Get historical matches for away team (before current match)
        away_history = df_copy[
            (df_copy['Date'] < match_date) & 
            ((df_copy['HomeTeam'] == away_team) | (df_copy['AwayTeam'] == away_team))
        ].tail(window)
        
        # Compute home team rolling form
        if len(home_history) > 0:
            # Goals for and against from home team's perspective
            home_gf = []
            home_ga = []
            
            for _, hist_match in home_history.iterrows():
                if hist_match['HomeTeam'] == home_team:
                    # Team played at home
                    home_gf.append(hist_match['HTHG'])
                    home_ga.append(hist_match['HTAG'])
                else:
                    # Team played away
                    home_gf.append(hist_match['HTAG'])
                    home_ga.append(hist_match['HTHG'])
            
            df_copy.at[idx, 'home_gf_r5'] = np.mean(home_gf)
            df_copy.at[idx, 'home_ga_r5'] = np.mean(home_ga)
            df_copy.at[idx, 'home_gd_r5'] = np.mean(home_gf) - np.mean(home_ga)
        
        # Compute away team rolling form
        if len(away_history) > 0:
            # Goals for and against from away team's perspective
            away_gf = []
            away_ga = []
            
            for _, hist_match in away_history.iterrows():
                if hist_match['HomeTeam'] == away_team:
                    # Team played at home
                    away_gf.append(hist_match['HTHG'])
                    away_ga.append(hist_match['HTAG'])
                else:
                    # Team played away
                    away_gf.append(hist_match['HTAG'])
                    away_ga.append(hist_match['HTHG'])
            
            df_copy.at[idx, 'away_gf_r5'] = np.mean(away_gf)
            df_copy.at[idx, 'away_ga_r5'] = np.mean(away_ga)
            df_copy.at[idx, 'away_gd_r5'] = np.mean(away_gf) - np.mean(away_ga)
        
        # Progress indicator
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx:,} matches...")
    
    print(f"✅ Rolling form computation completed!")
    
    # Show statistics
    valid_matches = df_copy[form_cols].dropna()
    if len(valid_matches) > 0:
        print(f"  Valid matches with rolling form: {len(valid_matches):,}")
        print(f"  Matches without history: {len(df_copy) - len(valid_matches):,}")
    
    return df_copy

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add days since last match for each team.
    
    Args:
        df: DataFrame with match data sorted by date
    
    Returns:
        DataFrame with rest day features added
    """
    df_copy = df.copy()
    
    # Ensure data is sorted by date
    df_copy = df_copy.sort_values('Date').reset_index(drop=True)
    
    # Initialize rest day columns
    df_copy['home_days_since_last'] = np.nan
    df_copy['away_days_since_last'] = np.nan
    
    print("Computing rest days for each team...")
    
    # For each match, find the days since last match for each team
    for idx, row in df_copy.iterrows():
        match_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Find home team's last match before this one
        home_last_matches = df_copy[
            (df_copy['Date'] < match_date) & 
            ((df_copy['HomeTeam'] == home_team) | (df_copy['AwayTeam'] == home_team))
        ]
        
        if len(home_last_matches) > 0:
            last_home_date = home_last_matches['Date'].max()
            days_since_home = (match_date - last_home_date).days
            df_copy.at[idx, 'home_days_since_last'] = days_since_home
        
        # Find away team's last match before this one
        away_last_matches = df_copy[
            (df_copy['Date'] < match_date) & 
            ((df_copy['HomeTeam'] == away_team) | (df_copy['AwayTeam'] == away_team))
        ]
        
        if len(away_last_matches) > 0:
            last_away_date = away_last_matches['Date'].max()
            days_since_away = (match_date - last_away_date).days
            df_copy.at[idx, 'away_days_since_last'] = days_since_away
        
        # Progress indicator
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx:,} matches...")
    
    print(f"✅ Rest days computation completed!")
    
    # Show statistics
    valid_rest = df_copy[['home_days_since_last', 'away_days_since_last']].dropna()
    if len(valid_rest) > 0:
        print(f"  Valid matches with rest data: {len(valid_rest):,}")
        print(f"  Average rest days - Home: {valid_rest['home_days_since_last'].mean():.1f}, Away: {valid_rest['away_days_since_last'].mean():.1f}")
    
    return df_copy

def transform_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform betting odds to log scale.
    
    Args:
        df: DataFrame with odds columns
    
    Returns:
        DataFrame with log-transformed odds
    """
    df_copy = df.copy()
    
    # Log transform odds, handling invalid values
    if 'B365H' in df_copy.columns:
        # Replace invalid odds (<=1 or missing) with median valid odds
        valid_home_odds = df_copy['B365H'][(df_copy['B365H'] > 1) & (df_copy['B365H'].notna())]
        if len(valid_home_odds) > 0:
            median_home_odds = valid_home_odds.median()
            df_copy['B365H'] = df_copy['B365H'].fillna(median_home_odds)
            df_copy.loc[df_copy['B365H'] <= 1, 'B365H'] = median_home_odds
        df_copy['log_home_win_odds'] = np.log(df_copy['B365H'])
    
    if 'B365D' in df_copy.columns:
        # Replace invalid odds (<=1 or missing) with median valid odds  
        valid_draw_odds = df_copy['B365D'][(df_copy['B365D'] > 1) & (df_copy['B365D'].notna())]
        if len(valid_draw_odds) > 0:
            median_draw_odds = valid_draw_odds.median()
            df_copy['B365D'] = df_copy['B365D'].fillna(median_draw_odds)
            df_copy.loc[df_copy['B365D'] <= 1, 'B365D'] = median_draw_odds
        df_copy['log_draw_odds'] = np.log(df_copy['B365D'])
    
    if 'B365A' in df_copy.columns:
        # Replace invalid odds (<=1 or missing) with median valid odds
        valid_away_odds = df_copy['B365A'][(df_copy['B365A'] > 1) & (df_copy['B365A'].notna())]
        if len(valid_away_odds) > 0:
            median_away_odds = valid_away_odds.median()
            df_copy['B365A'] = df_copy['B365A'].fillna(median_away_odds)
            df_copy.loc[df_copy['B365A'] <= 1, 'B365A'] = median_away_odds
        df_copy['log_away_win_odds'] = np.log(df_copy['B365A'])
    
    return df_copy

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the half-time draw target variable.

    Args:
        df: DataFrame with HTHG and HTAG columns

    Returns:
        DataFrame with y_ht_draw target column
    """
    df_copy = df.copy()
    df_copy['y_ht_draw'] = (df_copy['HTHG'] == df_copy['HTAG']).astype(int)
    return df_copy


# ── xG rolling feature names (for reference in train_v4.py) ──────────────────

XG_ROLLING_FEATURES = [
    # Full-match xG rolling averages
    "home_xg_r5",          # home team's avg xG over last 5 matches
    "home_xga_r5",         # home team's avg xG against over last 5 matches
    "home_xg_diff_r5",     # home team's avg xG differential over last 5 matches
    "away_xg_r5",          # away team's avg xG over last 5 matches
    "away_xga_r5",         # away team's avg xG against over last 5 matches
    "away_xg_diff_r5",     # away team's avg xG differential over last 5 matches
    # xG overperformance (regression signal)
    "home_goals_minus_xg_r5",  # home team's avg goals − xG (positive = overperforming)
    "away_goals_minus_xg_r5",  # away team's avg goals − xG
    # First-half xG (key new feature — requires match report scraping)
    "home_1h_xg_r5",      # home team's avg 1st-half xG over last 5 matches
    "home_1h_xga_r5",     # home team's avg 1st-half xG against over last 5 matches
    "away_1h_xg_r5",      # away team's avg 1st-half xG over last 5 matches
    "away_1h_xga_r5",     # away team's avg 1st-half xG against over last 5 matches
]


def compute_rolling_xg_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute rolling xG features per team from per-match xG columns.

    Expects columns: Date, HomeTeam, AwayTeam, home_xg, away_xg.
    Optionally uses: home_1h_xg, away_1h_xg (first-half splits).

    Returns DataFrame with XG_ROLLING_FEATURES columns added.
    Missing xG values are handled gracefully (rolling over available data only).
    """
    from collections import defaultdict

    df_out = df.sort_values("Date").reset_index(drop=True)

    has_1h = (
        "home_1h_xg" in df_out.columns
        and df_out["home_1h_xg"].notna().mean() > 0.01
    )

    for col in XG_ROLLING_FEATURES:
        df_out[col] = np.nan

    team_hist: dict = defaultdict(list)

    def _rolling_mean(records, key):
        vals = [
            r[key] for r in records[-window:]
            if not np.isnan(r.get(key, np.nan))
        ]
        return float(np.mean(vals)) if vals else np.nan

    for idx, row in df_out.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        h_xg  = row.get("home_xg",  np.nan)
        a_xg  = row.get("away_xg",  np.nan)
        h1_xg = row.get("home_1h_xg", np.nan) if has_1h else np.nan
        a1_xg = row.get("away_1h_xg", np.nan) if has_1h else np.nan
        h_g   = row.get("FTHG", np.nan)
        a_g   = row.get("FTAG", np.nan)

        # ── Home team rolling xG ───────────────────────────────────────────────
        h_hist = team_hist[home]
        if h_hist:
            hxf = _rolling_mean(h_hist, "xg_for")
            hxa = _rolling_mean(h_hist, "xg_against")
            df_out.at[idx, "home_xg_r5"]  = hxf
            df_out.at[idx, "home_xga_r5"] = hxa
            if not np.isnan(hxf) and not np.isnan(hxa):
                df_out.at[idx, "home_xg_diff_r5"] = hxf - hxa
            hgf = _rolling_mean(h_hist, "goals_for")
            if not np.isnan(hgf) and not np.isnan(hxf):
                df_out.at[idx, "home_goals_minus_xg_r5"] = hgf - hxf
            if has_1h:
                df_out.at[idx, "home_1h_xg_r5"]  = _rolling_mean(h_hist, "xg_1h_for")
                df_out.at[idx, "home_1h_xga_r5"] = _rolling_mean(h_hist, "xg_1h_against")

        # ── Away team rolling xG ───────────────────────────────────────────────
        a_hist = team_hist[away]
        if a_hist:
            axf = _rolling_mean(a_hist, "xg_for")
            axa = _rolling_mean(a_hist, "xg_against")
            df_out.at[idx, "away_xg_r5"]  = axf
            df_out.at[idx, "away_xga_r5"] = axa
            if not np.isnan(axf) and not np.isnan(axa):
                df_out.at[idx, "away_xg_diff_r5"] = axf - axa
            agf = _rolling_mean(a_hist, "goals_for")
            if not np.isnan(agf) and not np.isnan(axf):
                df_out.at[idx, "away_goals_minus_xg_r5"] = agf - axf
            if has_1h:
                df_out.at[idx, "away_1h_xg_r5"]  = _rolling_mean(a_hist, "xg_1h_for")
                df_out.at[idx, "away_1h_xga_r5"] = _rolling_mean(a_hist, "xg_1h_against")

        # ── Update history (no look-ahead) ─────────────────────────────────────
        team_hist[home].append({
            "xg_for":       h_xg,
            "xg_against":   a_xg,
            "xg_1h_for":    h1_xg,
            "xg_1h_against": a1_xg,
            "goals_for":    h_g,
            "goals_against": a_g,
        })
        team_hist[away].append({
            "xg_for":       a_xg,
            "xg_against":   h_xg,
            "xg_1h_for":    a1_xg,
            "xg_1h_against": h1_xg,
            "goals_for":    a_g,
            "goals_against": h_g,
        })

        if idx % 10000 == 0 and idx > 0:
            print(f"  Rolling xG: {idx:,}/{len(df_out):,} processed...")

    return df_out