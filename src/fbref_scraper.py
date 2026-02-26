"""
FBref Match-Level xG Scraper
==============================
Scrapes per-match xG data from FBref for top European leagues.
Includes first-half / second-half xG splits when available.

Rate limited to 3 req/s (FBref's limit).
Checkpointed — safe to interrupt and resume.

Usage:
    python src/fbref_scraper.py                    # scrape all leagues/seasons
    python src/fbref_scraper.py --league EPL        # single league
    python src/fbref_scraper.py --season 2024-2025  # single season

Output: data/xg/fbref_{league}_{season}.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

RATE_LIMIT = 3.5  # seconds between requests (FBref limit is ~3/s, be conservative)

# FBref league IDs and our internal codes
LEAGUES = {
    "EPL":        {"fbref_id": "9",  "fbref_name": "Premier-League",     "country": "ENG", "our_code": "E0"},
    "La_Liga":    {"fbref_id": "12", "fbref_name": "La-Liga",            "country": "ESP", "our_code": "SP1"},
    "Serie_A":    {"fbref_id": "11", "fbref_name": "Serie-A",            "country": "ITA", "our_code": "I1"},
    "Bundesliga": {"fbref_id": "20", "fbref_name": "Bundesliga",         "country": "GER", "our_code": "D1"},
    "Ligue_1":    {"fbref_id": "13", "fbref_name": "Ligue-1",            "country": "FRA", "our_code": "F1"},
    "Championship": {"fbref_id": "10", "fbref_name": "Championship",     "country": "ENG", "our_code": "E1"},
}

# Seasons available on FBref with xG data (2017-18 onwards for top 5)
SEASONS = [
    "2017-2018", "2018-2019", "2019-2020", "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025",
]

DATA_DIR = Path("data/xg")
CHECKPOINT_FILE = DATA_DIR / "fbref_checkpoint.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_checkpoint(cp: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)

# ── Scraping ──────────────────────────────────────────────────────────────────

def get_season_fixtures_url(league_key: str, season: str) -> str:
    """Build FBref scores & fixtures URL for a league-season."""
    info = LEAGUES[league_key]
    # FBref URL format: /en/comps/{id}/{season}/schedule/{season}-{league}-Scores-and-Fixtures
    return (
        f"https://fbref.com/en/comps/{info['fbref_id']}/{season}/schedule/"
        f"{season}-{info['fbref_name']}-Scores-and-Fixtures"
    )

def get_match_report_url(match_id: str) -> str:
    """Build FBref match report URL."""
    return f"https://fbref.com/en/matches/{match_id}"


def scrape_season_fixtures(league_key: str, season: str) -> List[Dict]:
    """
    Scrape the scores & fixtures page for a league-season.
    Returns list of match dicts with basic info + match report links.
    """
    url = get_season_fixtures_url(league_key, season)
    print(f"  Fetching fixtures: {url}")
    
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT)
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Find the scores table
    table = soup.find("table", {"id": re.compile(r"sched_\d+_\d+")})
    if not table:
        # Try alternate ID format
        table = soup.find("table", class_="stats_table")
    
    if not table:
        print(f"    WARNING: No fixtures table found for {league_key} {season}")
        return []
    
    tbody = table.find("tbody")
    if not tbody:
        return []
    
    matches = []
    rows = tbody.find_all("tr", class_=lambda c: c != "thead")
    
    for row in rows:
        # Skip spacer/header rows
        if row.get("class") and "spacer" in row.get("class", []):
            continue
        if row.find("th", {"scope": "col"}):
            continue
            
        cells = row.find_all(["td", "th"])
        if len(cells) < 8:
            continue
        
        # Extract data from cells
        match = {}
        
        for cell in cells:
            stat = cell.get("data-stat", "")
            
            if stat == "date":
                match["date"] = cell.get_text(strip=True)
            elif stat == "home_team":
                match["home_team"] = cell.get_text(strip=True)
            elif stat == "away_team":
                match["away_team"] = cell.get_text(strip=True)
            elif stat == "home_xg":
                val = cell.get_text(strip=True)
                match["home_xg"] = float(val) if val else None
            elif stat == "away_xg":
                val = cell.get_text(strip=True)
                match["away_xg"] = float(val) if val else None
            elif stat == "score":
                score_text = cell.get_text(strip=True)
                match["score"] = score_text
                # Extract match report link
                link = cell.find("a")
                if link and link.get("href"):
                    href = link["href"]
                    # Extract match ID from href like /en/matches/abc123/...
                    m = re.search(r"/matches/([a-f0-9]+)/", href)
                    if m:
                        match["match_id"] = m.group(1)
                        match["match_url"] = f"https://fbref.com{href}"
            elif stat == "referee":
                match["referee"] = cell.get_text(strip=True)
            elif stat == "venue":
                match["venue"] = cell.get_text(strip=True)
        
        if match.get("home_team") and match.get("away_team") and match.get("date"):
            match["league"] = league_key
            match["season"] = season
            matches.append(match)
    
    print(f"    Found {len(matches)} matches, {sum(1 for m in matches if m.get('home_xg') is not None)} with xG")
    return matches


def scrape_match_halftime_xg(match_url: str) -> Optional[Dict]:
    """
    Scrape a single match report for first-half xG splits.
    Returns dict with 1H xG for home/away, or None if unavailable.
    
    FBref match reports have shot logs with minute data.
    We sum xG for shots in minutes 1-45 for first-half xG.
    """
    try:
        resp = requests.get(match_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT)
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Look for shot tables (both home and away)
        # FBref has "Shots" tables with minute, xG per shot
        result = {"home_1h_xg": 0.0, "away_1h_xg": 0.0,
                  "home_2h_xg": 0.0, "away_2h_xg": 0.0,
                  "home_shots_1h": 0, "away_shots_1h": 0}
        
        # Find all shot tables
        shot_tables = soup.find_all("table", {"id": re.compile(r"shots_")})
        
        if not shot_tables:
            return None
        
        for table in shot_tables:
            table_id = table.get("id", "")
            is_home = "home" in table_id.lower() or table_id.endswith("_sh")
            
            # Determine if home or away from table context
            # The first shots table is typically "shots_all" (combined)
            # Individual team tables are "shots_{team_id}"
            
            tbody = table.find("tbody")
            if not tbody:
                continue
                
            for row in tbody.find_all("tr"):
                cells = row.find_all(["td", "th"])
                minute_cell = None
                xg_cell = None
                squad_cell = None
                
                for cell in cells:
                    stat = cell.get("data-stat", "")
                    if stat == "minute":
                        minute_cell = cell
                    elif stat == "xg_shot":
                        xg_cell = cell
                    elif stat == "team":  # In combined table
                        squad_cell = cell
                
                if minute_cell and xg_cell:
                    try:
                        minute_text = minute_cell.get_text(strip=True)
                        # Handle "45+2" style minutes
                        base_minute = int(re.match(r"(\d+)", minute_text).group(1))
                        xg_val = float(xg_cell.get_text(strip=True) or 0)
                        
                        is_first_half = base_minute <= 45
                        
                        # For combined table, check squad
                        if squad_cell:
                            # We'd need to match team names — skip combined for now
                            pass
                        else:
                            # Individual team table
                            prefix = "home" if is_home else "away"
                            half = "1h" if is_first_half else "2h"
                            result[f"{prefix}_{half}_xg"] += xg_val
                            if is_first_half:
                                result[f"{prefix}_shots_1h"] += 1
                    except (ValueError, AttributeError):
                        continue
        
        # Only return if we got meaningful data
        if result["home_1h_xg"] > 0 or result["away_1h_xg"] > 0:
            return result
        return None
        
    except Exception as e:
        print(f"    Error scraping match: {e}")
        return None


def scrape_league_season(league_key: str, season: str, checkpoint: Dict,
                         skip_match_reports: bool = True) -> pd.DataFrame:
    """
    Scrape a full league-season. Returns DataFrame.
    
    If skip_match_reports=True, only scrapes the fixtures page (fast, gets match xG).
    If False, also scrapes individual match reports for 1H xG splits (slow, ~3s/match).
    """
    key = f"{league_key}_{season}"
    csv_path = DATA_DIR / f"fbref_{key}.csv"
    
    # Check if already done
    if key in checkpoint.get("completed", []):
        if csv_path.exists():
            print(f"  ✓ {key} already completed, loading from cache")
            return pd.read_csv(csv_path)
        # File missing but marked complete — rescrape
        checkpoint["completed"].remove(key)
    
    print(f"\n{'='*60}")
    print(f"  Scraping {league_key} {season}")
    print(f"{'='*60}")
    
    matches = scrape_season_fixtures(league_key, season)
    
    if not matches:
        print(f"  No matches found for {key}")
        checkpoint.setdefault("failed", []).append(key)
        save_checkpoint(checkpoint)
        return pd.DataFrame()
    
    # Optionally scrape individual match reports for 1H xG
    if not skip_match_reports:
        n_with_url = sum(1 for m in matches if m.get("match_url"))
        print(f"  Scraping {n_with_url} match reports for 1H xG splits...")
        
        for i, match in enumerate(matches):
            if not match.get("match_url"):
                continue
            
            halftime_data = scrape_match_halftime_xg(match["match_url"])
            if halftime_data:
                match.update(halftime_data)
            
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{n_with_url} match reports scraped")
    
    df = pd.DataFrame(matches)
    
    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} matches to {csv_path}")
    
    checkpoint.setdefault("completed", []).append(key)
    save_checkpoint(checkpoint)
    
    return df


def scrape_all(leagues: Optional[List[str]] = None,
               seasons: Optional[List[str]] = None,
               skip_match_reports: bool = True) -> pd.DataFrame:
    """
    Scrape all specified leagues and seasons.
    Returns combined DataFrame.
    """
    leagues = leagues or list(LEAGUES.keys())
    seasons = seasons or SEASONS
    
    checkpoint = load_checkpoint()
    all_dfs = []
    
    total = len(leagues) * len(seasons)
    done = 0
    
    for league in leagues:
        if league not in LEAGUES:
            print(f"  Unknown league: {league}. Available: {list(LEAGUES.keys())}")
            continue
            
        for season in seasons:
            done += 1
            print(f"\n[{done}/{total}]", end="")
            
            try:
                df = scrape_league_season(league, season, checkpoint,
                                         skip_match_reports=skip_match_reports)
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                print(f"  ERROR: {league} {season}: {e}")
                checkpoint.setdefault("failed", []).append(f"{league}_{season}")
                save_checkpoint(checkpoint)
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = DATA_DIR / "fbref_all_xg.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*60}")
        print(f"  TOTAL: {len(combined)} matches across {len(all_dfs)} league-seasons")
        print(f"  With xG: {combined['home_xg'].notna().sum()}")
        print(f"  Saved to {combined_path}")
        print(f"{'='*60}")
        return combined
    
    return pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FBref xG Scraper")
    parser.add_argument("--league", default=None,
                        help=f"Single league to scrape. Options: {list(LEAGUES.keys())}")
    parser.add_argument("--season", default=None,
                        help="Single season (e.g. 2024-2025)")
    parser.add_argument("--match-reports", action="store_true",
                        help="Also scrape individual match reports for 1H xG (slow)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset checkpoint and rescrape everything")
    args = parser.parse_args()
    
    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint reset.")
    
    leagues = [args.league] if args.league else None
    seasons = [args.season] if args.season else None
    
    scrape_all(leagues=leagues, seasons=seasons,
               skip_match_reports=not args.match_reports)


if __name__ == "__main__":
    main()
