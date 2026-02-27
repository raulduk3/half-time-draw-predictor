"""Fast vectorized merge of Understat xG into mega dataset."""
import pandas as pd
import numpy as np
from pathlib import Path

TEAM_MAP = {
    'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'ham Forest",
    'Queens Park Rangers': 'QPR', 'West Bromwich Albion': 'West Brom',
    'Wolverhampton Wanderers': 'Wolves',
    'Athletic Club': 'Ath Bilbao', 'Atletico Madrid': 'Ath Madrid',
    'Celta Vigo': 'Celta', 'Deportivo La Coruna': 'La Coruna',
    'Espanyol': 'Espanol', 'Rayo Vallecano': 'Vallecano',
    'Real Betis': 'Betis', 'Real Sociedad': 'Sociedad',
    'Real Valladolid': 'Valladolid', 'SD Huesca': 'Huesca',
    'Sporting Gijon': 'Sp Gijon',
    'Arminia Bielefeld': 'Bielefeld', 'Bayer Leverkusen': 'Leverkusen',
    'Borussia Dortmund': 'Dortmund', 'Borussia M.Gladbach': "M'gladbach",
    'Eintracht Frankfurt': 'Ein Frankfurt', 'FC Cologne': 'FC Koln',
    'FC Heidenheim': 'Heidenheim', 'Fortuna Duesseldorf': 'Fortuna Dusseldorf',
    'Greuther Fuerth': 'Greuther Furth', 'Hamburger SV': 'Hamburg',
    'Hannover 96': 'Hannover', 'Hertha Berlin': 'Hertha',
    'Mainz 05': 'Mainz', 'Nuernberg': 'Nurnberg',
    'RasenBallsport Leipzig': 'RB Leipzig', 'St. Pauli': 'St Pauli',
    'VfB Stuttgart': 'Stuttgart',
    'Clermont Foot': 'Clermont', 'GFC Ajaccio': 'Ajaccio',
    'Paris Saint Germain': 'Paris SG', 'SC Bastia': 'Bastia',
    'Saint-Etienne': 'St Etienne',
    'AC Milan': 'Milan', 'Parma Calcio 1913': 'Parma', 'SPAL 2013': 'Spal',
}

def main():
    xg = pd.read_csv('data/xg/understat_all_xg.csv')
    mega = pd.read_parquet('data/processed/mega_dataset_v2.parquet')
    
    # Map team names and dates
    xg['HomeTeam'] = xg['home_team'].map(TEAM_MAP).fillna(xg['home_team'])
    xg['AwayTeam'] = xg['away_team'].map(TEAM_MAP).fillna(xg['away_team'])
    xg['Date'] = pd.to_datetime(xg['date']).dt.normalize()
    mega['Date'] = pd.to_datetime(mega['Date']).dt.normalize()
    
    # Merge on exact date + teams
    xg_slim = xg[['HomeTeam', 'AwayTeam', 'Date', 'home_xg', 'away_xg']].copy()
    xg_slim = xg_slim.rename(columns={'home_xg': 'xg_home', 'away_xg': 'xg_away'})
    
    # Drop any existing xg columns from mega
    xg_cols_existing = [c for c in mega.columns if 'xg' in c.lower()]
    if xg_cols_existing:
        mega = mega.drop(columns=xg_cols_existing)
    
    merged = mega.merge(xg_slim, on=['HomeTeam', 'AwayTeam', 'Date'], how='left')
    
    # Try ±1 day for unmatched
    unmatched = merged['xg_home'].isna()
    print(f'Exact match: {(~unmatched).sum()}/{len(xg)} xG records')
    
    if unmatched.sum() > 0:
        for offset in [-1, 1]:
            xg_shifted = xg_slim.copy()
            xg_shifted['Date'] = xg_shifted['Date'] + pd.Timedelta(days=offset)
            xg_shifted = xg_shifted.rename(columns={'xg_home': 'xg_home_shifted', 'xg_away': 'xg_away_shifted'})
            
            merged = merged.merge(xg_shifted, on=['HomeTeam', 'AwayTeam', 'Date'], how='left')
            fill_mask = merged['xg_home'].isna() & merged['xg_home_shifted'].notna()
            merged.loc[fill_mask, 'xg_home'] = merged.loc[fill_mask, 'xg_home_shifted']
            merged.loc[fill_mask, 'xg_away'] = merged.loc[fill_mask, 'xg_away_shifted']
            merged = merged.drop(columns=['xg_home_shifted', 'xg_away_shifted'])
    
    matched = merged['xg_home'].notna().sum()
    print(f'After fuzzy: {matched}/{len(xg)} xG records matched ({matched/len(xg)*100:.1f}%)')
    
    # Compute rolling xG features per team
    print('Computing rolling xG features...')
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    # Melt to per-team perspective
    home_view = merged[['Date', 'HomeTeam', 'xg_home', 'xg_away', 'FTHG']].rename(
        columns={'HomeTeam': 'team', 'xg_home': 'team_xg', 'xg_away': 'team_xga', 'FTHG': 'team_goals'})
    away_view = merged[['Date', 'AwayTeam', 'xg_away', 'xg_home', 'FTAG']].rename(
        columns={'AwayTeam': 'team', 'xg_away': 'team_xg', 'xg_home': 'team_xga', 'FTAG': 'team_goals'})
    home_view['mega_idx'] = merged.index
    away_view['mega_idx'] = merged.index
    home_view['is_home'] = True
    away_view['is_home'] = False
    
    team_matches = pd.concat([home_view, away_view]).sort_values('Date')
    
    # Rolling 5-match per team (shift 1 for no lookahead)
    for col_name, source in [('xg_r5', 'team_xg'), ('xga_r5', 'team_xga')]:
        team_matches[col_name] = (
            team_matches.groupby('team')[source]
            .transform(lambda x: x.rolling(5, min_periods=3).mean().shift(1))
        )
    
    team_matches['xg_diff_r5'] = team_matches['xg_r5'] - team_matches['xga_r5']
    team_matches['_gmxg'] = team_matches['team_goals'] - team_matches['team_xg']
    team_matches['goals_minus_xg_r5'] = (
        team_matches.groupby('team')['_gmxg']
        .transform(lambda x: x.rolling(5, min_periods=3).mean().shift(1))
    )
    team_matches = team_matches.drop(columns=['_gmxg'])
    
    # Assign back with home/away prefix
    for prefix, is_home in [('home', True), ('away', False)]:
        view = team_matches[team_matches['is_home'] == is_home]
        for feat in ['xg_r5', 'xga_r5', 'xg_diff_r5', 'goals_minus_xg_r5']:
            merged.loc[view['mega_idx'], f'{prefix}_{feat}'] = view[feat].values
    
    # Coverage stats
    new_cols = [c for c in merged.columns if 'xg' in c.lower()]
    for col in new_cols:
        n = merged[col].notna().sum()
        print(f'  {col}: {n:,} non-null ({n/len(merged)*100:.1f}%)')
    
    out = Path('data/processed/mega_dataset_v3.parquet')
    merged.to_parquet(out, index=False)
    print(f'\nSaved {out}: {len(merged):,} rows × {len(merged.columns)} cols')

if __name__ == '__main__':
    main()
