"""
Scrape xG from Understat using Selenium headless.
Extracts match-level xG for all 5 leagues Understat covers.

Usage:
    python src/scrape_understat_current.py                  # current season (2025-26)
    python src/scrape_understat_current.py --season 2024    # specific season
    python src/scrape_understat_current.py --league EPL     # specific league
    python src/scrape_understat_current.py --all-seasons    # all seasons 2014-2025
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

LEAGUES = {
    'EPL': 'EPL',
    'La_Liga': 'La liga',
    'Bundesliga': 'Bundesliga',
    'Serie_A': 'Serie A',
    'Ligue_1': 'Ligue 1',
}

OUT_DIR = Path('data/xg')
EXISTING = OUT_DIR / 'understat_all_xg.csv'


def scrape_league_season(league_understat: str, season: int, driver) -> list:
    """Scrape all matches for one league/season from Understat via JS execution."""
    url = f'https://understat.com/league/{league_understat}/{season}'
    print(f'  Fetching {url}...', end=' ', flush=True)
    
    driver.get(url)
    time.sleep(4)
    
    # Extract datesData directly from JS context
    try:
        raw = driver.execute_script(
            'return typeof datesData !== "undefined" ? JSON.stringify(datesData) : null'
        )
    except Exception as e:
        print(f'JS error: {e}')
        return []
    
    if not raw:
        print('no datesData')
        return []
    
    data = json.loads(raw)
    
    results = []
    for match in data:
        if not isinstance(match, dict):
            continue
        
        is_result = match.get('isResult', False)
        if not is_result:
            continue
        
        home = match.get('h', {})
        away = match.get('a', {})
        goals = match.get('goals', {})
        xg = match.get('xG', {})
        
        try:
            results.append({
                'league': league_understat,
                'season': season,
                'game_id': match.get('id', ''),
                'date': match.get('datetime', ''),
                'home_team': home.get('title', ''),
                'away_team': away.get('title', ''),
                'home_goals': int(goals.get('h', 0)),
                'away_goals': int(goals.get('a', 0)),
                'home_xg': float(xg.get('h', 0)),
                'away_xg': float(xg.get('a', 0)),
                'is_result': True,
            })
        except (ValueError, TypeError):
            continue
    
    print(f'{len(results)} matches')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=2025)
    parser.add_argument('--league', type=str, default=None)
    parser.add_argument('--all-seasons', action='store_true')
    args = parser.parse_args()
    
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        all_results = []
        
        leagues = {args.league: LEAGUES[args.league]} if args.league else LEAGUES
        seasons = list(range(2014, 2026)) if args.all_seasons else [args.season]
        
        for season in seasons:
            print(f'\nSeason {season}/{season+1}:')
            for key, understat_name in leagues.items():
                results = scrape_league_season(understat_name, season, driver)
                all_results.extend(results)
                time.sleep(2)
        
        if not all_results:
            print('No matches scraped.')
            return
        
        df = pd.DataFrame(all_results)
        
        # Map league names
        league_map = {
            'EPL': 'ENG-Premier League',
            'La liga': 'ESP-La Liga',
            'Bundesliga': 'GER-Bundesliga',
            'Serie A': 'ITA-Serie A',
            'Ligue 1': 'FRA-Ligue 1',
        }
        df['league_full'] = df['league'].map(league_map)
        
        # Save season-specific file
        out_file = OUT_DIR / f'understat_{args.season if not args.all_seasons else "all"}.csv'
        df.to_csv(out_file, index=False)
        print(f'\nSaved {len(df)} matches to {out_file}')
        
        # Append to main file
        if EXISTING.exists():
            existing = pd.read_csv(EXISTING)
            
            new_rows = []
            for _, row in df.iterrows():
                new_rows.append({
                    'league': row['league_full'],
                    'season': f'{row["season"]}{(row["season"]+1) % 100:02d}',
                    'game': f'{str(row["date"])[:10]} {row["home_team"]}-{row["away_team"]}',
                    'league_id': '',
                    'season_id': row['season'],
                    'game_id': row['game_id'],
                    'date': row['date'],
                    'home_team_id': '',
                    'away_team_id': '',
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'away_team_code': '',
                    'home_team_code': '',
                    'home_goals': row['home_goals'],
                    'away_goals': row['away_goals'],
                    'home_xg': row['home_xg'],
                    'away_xg': row['away_xg'],
                    'is_result': True,
                    'has_data': True,
                    'url': f'https://understat.com/match/{row["game_id"]}',
                })
            
            new_df = pd.DataFrame(new_rows)
            existing_ids = set(existing['game_id'].astype(str))
            new_df = new_df[~new_df['game_id'].astype(str).isin(existing_ids)]
            
            if len(new_df) > 0:
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined.to_csv(EXISTING, index=False)
                print(f'Appended {len(new_df)} new matches to {EXISTING} (total: {len(combined)})')
            else:
                print(f'All matches already in {EXISTING}')
        else:
            # Create the file in the expected format
            new_rows = []
            for _, row in df.iterrows():
                new_rows.append({
                    'league': row['league_full'],
                    'season': f'{row["season"]}{(row["season"]+1) % 100:02d}',
                    'game': f'{str(row["date"])[:10]} {row["home_team"]}-{row["away_team"]}',
                    'league_id': '', 'season_id': row['season'],
                    'game_id': row['game_id'], 'date': row['date'],
                    'home_team_id': '', 'away_team_id': '',
                    'home_team': row['home_team'], 'away_team': row['away_team'],
                    'away_team_code': '', 'home_team_code': '',
                    'home_goals': row['home_goals'], 'away_goals': row['away_goals'],
                    'home_xg': row['home_xg'], 'away_xg': row['away_xg'],
                    'is_result': True, 'has_data': True,
                    'url': f'https://understat.com/match/{row["game_id"]}',
                })
            pd.DataFrame(new_rows).to_csv(EXISTING, index=False)
            print(f'Created {EXISTING} with {len(new_rows)} matches')
        
        # Summary
        print(f'\nSummary:')
        for league in df['league'].unique():
            ldf = df[df['league'] == league]
            print(f'  {league}: {len(ldf)} matches, '
                  f'avg xG: {ldf["home_xg"].mean():.2f} H / {ldf["away_xg"].mean():.2f} A')
    
    finally:
        driver.quit()


if __name__ == '__main__':
    main()
