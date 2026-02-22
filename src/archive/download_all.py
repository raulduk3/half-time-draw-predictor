"""
Download ALL available CSV data from football-data.co.uk to build the largest possible dataset.
"""
import os
import requests
import time
from pathlib import Path
from typing import List, Dict

def download_all_data():
    """Download all available CSV data from football-data.co.uk"""

    # Base URL pattern
    base_url = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

    # All seasons in format: last 2 digits of each year
    seasons = [
        "9394", "9495", "9596", "9697", "9798", "9899", "9900",
        "0001", "0102", "0203", "0304", "0405", "0506", "0607",
        "0708", "0809", "0910", "1011", "1112", "1213", "1314",
        "1415", "1516", "1617", "1718", "1819", "1920", "2021",
        "2122", "2223", "2324", "2425", "2526"
    ]

    # All leagues with their countries
    leagues = {
        # England
        "E0": "England",  # Premier League
        "E1": "England",  # Championship
        "E2": "England",  # League 1
        "E3": "England",  # League 2
        "EC": "England",  # Conference
        # Scotland
        "SC0": "Scotland",
        "SC1": "Scotland",
        "SC2": "Scotland",
        "SC3": "Scotland",
        # Germany
        "D1": "Germany",
        "D2": "Germany",
        # Italy
        "I1": "Italy",
        "I2": "Italy",
        # Spain
        "SP1": "Spain",
        "SP2": "Spain",
        # France
        "F1": "France",
        "F2": "France",
        # Netherlands
        "N1": "Netherlands",
        # Belgium
        "B1": "Belgium",
        # Portugal
        "P1": "Portugal",
        # Turkey
        "T1": "Turkey",
        # Greece
        "G1": "Greece"
    }

    # Create base directory
    base_dir = Path("data/raw_all")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "downloaded": 0,
        "skipped_existing": 0,
        "not_found": 0,
        "errors": 0,
        "by_league": {},
        "by_country": {}
    }

    total_combinations = len(seasons) * len(leagues)
    current = 0

    print(f"🏈 Starting download of ALL football-data.co.uk CSV files")
    print(f"📊 Attempting {total_combinations:,} season/league combinations")
    print(f"💾 Saving to: {base_dir.absolute()}")
    print()

    for season in seasons:
        for league, country in leagues.items():
            current += 1

            # Create league directory
            league_dir = base_dir / league
            league_dir.mkdir(exist_ok=True)

            # Target file path
            file_path = league_dir / f"{season}.csv"

            # Skip if file already exists
            if file_path.exists():
                stats["skipped_existing"] += 1
                print(f"⏭️  [{current:4d}/{total_combinations}] SKIP: {league}/{season}.csv (already exists)")
                continue

            # Construct URL
            url = base_url.format(season=season, league=league)

            try:
                # Make request with timeout
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    # Try different encodings
                    content = None
                    encodings = ['utf-8', 'latin-1', 'cp1252']

                    for encoding in encodings:
                        try:
                            content = response.content.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue

                    if content is None:
                        # Fall back to bytes if all encodings fail
                        content = response.content.decode('utf-8', errors='replace')
                        print(f"⚠️  [{current:4d}/{total_combinations}] WARN: {league}/{season}.csv (encoding issues)")

                    # Save file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    # Update stats
                    stats["downloaded"] += 1
                    stats["by_league"][league] = stats["by_league"].get(league, 0) + 1
                    stats["by_country"][country] = stats["by_country"].get(country, 0) + 1

                    file_size = file_path.stat().st_size / 1024  # KB
                    print(f"✅ [{current:4d}/{total_combinations}] DOWN: {league}/{season}.csv ({file_size:.1f}KB)")

                elif response.status_code == 404:
                    stats["not_found"] += 1
                    print(f"❌ [{current:4d}/{total_combinations}] 404:  {league}/{season}.csv")

                else:
                    stats["errors"] += 1
                    print(f"💥 [{current:4d}/{total_combinations}] ERR:  {league}/{season}.csv (HTTP {response.status_code})")

            except requests.RequestException as e:
                stats["errors"] += 1
                print(f"💥 [{current:4d}/{total_combinations}] ERR:  {league}/{season}.csv ({str(e)[:50]}...)")

            # Small delay to be polite to the server
            time.sleep(0.1)

    print()
    print("🎯 Download Summary:")
    print(f"✅ Downloaded: {stats['downloaded']:,} files")
    print(f"⏭️  Skipped (existing): {stats['skipped_existing']:,} files")
    print(f"❌ Not found (404): {stats['not_found']:,} files")
    print(f"💥 Errors: {stats['errors']:,} files")
    print(f"📁 Total combinations tried: {total_combinations:,}")

    if stats["downloaded"] > 0:
        print()
        print("📊 Downloads by Country:")
        for country, count in sorted(stats["by_country"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {country}: {count:,} files")

        print()
        print("🏆 Downloads by League:")
        for league, count in sorted(stats["by_league"].items(), key=lambda x: x[1], reverse=True):
            country = leagues[league]
            print(f"  {league} ({country}): {count:,} files")

    # Calculate total disk usage
    total_size = sum(f.stat().st_size for f in base_dir.rglob("*.csv")) / (1024 * 1024)  # MB
    print(f"💾 Total disk usage: {total_size:.1f} MB")

    return stats

if __name__ == "__main__":
    stats = download_all_data()

    if stats["downloaded"] == 0 and stats["skipped_existing"] == 0:
        print("\n❌ No data was downloaded or found. Check your internet connection.")
        exit(1)
    else:
        print(f"\n🚀 Ready to build mega dataset from {stats['downloaded'] + stats['skipped_existing']:,} CSV files!")