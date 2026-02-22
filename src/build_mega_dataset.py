"""
Build the ultimate mega dataset from ALL downloaded football-data.co.uk CSV files.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import warnings

# Import existing feature functions
from features import compute_rolling_form, add_rest_days, transform_odds, create_target

warnings.filterwarnings('ignore')

def parse_date_robust(date_str: str) -> Optional[pd.Timestamp]:
    """Robustly parse dates from multiple formats"""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return None

    # Clean the string
    date_str = date_str.strip()

    # Common formats to try
    formats = [
        '%d/%m/%y',     # 01/08/94
        '%d/%m/%Y',     # 01/08/1994
        '%Y-%m-%d',     # 1994-08-01
        '%d-%m-%y',     # 01-08-94
        '%d-%m-%Y',     # 01-08-1994
        '%d.%m.%y',     # 01.08.94
        '%d.%m.%Y',     # 01.08.1994
    ]

    for fmt in formats:
        try:
            parsed = pd.to_datetime(date_str, format=fmt)
            # Handle 2-digit years - assume anything > 50 is 19xx, else 20xx
            if fmt.endswith('/%y') or fmt.endswith('-%y') or fmt.endswith('.%y'):
                if parsed.year > 2050:
                    parsed = parsed.replace(year=parsed.year - 100)
            return parsed
        except:
            continue

    # Try pandas default parser as last resort
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return None

def normalize_team_name(name: str) -> str:
    """Normalize team names to handle encoding issues and variations"""
    if not isinstance(name, str):
        return str(name)

    # Clean up common encoding issues and normalize
    name = name.strip()

    # Replace common character encoding issues
    replacements = {
        'Ãa': 'ã',
        'Ã§': 'ç',
        'Ã©': 'é',
        'Ã¡': 'á',
        'Ã³': 'ó',
        'Ã­': 'í',
        'Ãº': 'ú',
        'Ã¢': 'â',
        'Ã´': 'ô',
        'Ã¨': 'è',
        'Ã¬': 'ì',
        'Ã¹': 'ù',
        'Ã«': 'ë',
    }

    for bad, good in replacements.items():
        name = name.replace(bad, good)

    return name

def compute_rolling_stats(df: pd.DataFrame, stat_cols: List[str], window: int = 5) -> pd.DataFrame:
    """Compute rolling statistics for match stats like shots, corners, fouls"""
    df_copy = df.copy()

    # Sort by date
    df_copy = df_copy.sort_values('Date').reset_index(drop=True)

    # Initialize rolling stat columns
    for stat in stat_cols:
        if stat in df_copy.columns:
            df_copy[f'home_{stat.lower()}_r{window}'] = np.nan
            df_copy[f'away_{stat.lower()}_r{window}'] = np.nan
            if stat.endswith('T') and stat[:-1] in df_copy.columns:  # e.g., HST and HS
                # Also compute ratio (shots on target / shots)
                base_stat = stat[:-1]
                df_copy[f'home_{base_stat.lower()}_ratio_r{window}'] = np.nan
                df_copy[f'away_{base_stat.lower()}_ratio_r{window}'] = np.nan

    print(f"Computing rolling stats for {stat_cols} over {len(df_copy)} matches...")

    # For each match, compute rolling stats
    for idx, row in df_copy.iterrows():
        match_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Get historical matches for teams
        home_history = df_copy[
            (df_copy['Date'] < match_date) &
            ((df_copy['HomeTeam'] == home_team) | (df_copy['AwayTeam'] == home_team))
        ].tail(window)

        away_history = df_copy[
            (df_copy['Date'] < match_date) &
            ((df_copy['HomeTeam'] == away_team) | (df_copy['AwayTeam'] == away_team))
        ].tail(window)

        # Compute stats for each team
        for team, history, prefix in [(home_team, home_history, 'home'), (away_team, away_history, 'away')]:
            if len(history) > 0:
                for stat in stat_cols:
                    if stat in df_copy.columns:
                        values = []
                        ratio_numerator = []
                        ratio_denominator = []

                        for _, hist_match in history.iterrows():
                            if hist_match['HomeTeam'] == team:
                                # Team played at home - use home stats
                                stat_col = stat.replace('A', 'H')  # AS -> HS, AC -> HC, etc.
                                if stat_col in hist_match and pd.notna(hist_match[stat_col]):
                                    values.append(hist_match[stat_col])

                                    # For ratio calculation (e.g., HST/HS)
                                    if stat.endswith('T'):
                                        base_stat = stat[:-1].replace('A', 'H')  # AST -> HS
                                        if base_stat in hist_match and pd.notna(hist_match[base_stat]):
                                            ratio_numerator.append(hist_match[stat_col])
                                            ratio_denominator.append(hist_match[base_stat])
                            else:
                                # Team played away - use away stats
                                stat_col = stat.replace('H', 'A')  # HS -> AS, HC -> AC, etc.
                                if stat_col in hist_match and pd.notna(hist_match[stat_col]):
                                    values.append(hist_match[stat_col])

                                    # For ratio calculation
                                    if stat.endswith('T'):
                                        base_stat = stat[:-1].replace('H', 'A')  # HST -> AS
                                        if base_stat in hist_match and pd.notna(hist_match[base_stat]):
                                            ratio_numerator.append(hist_match[stat_col])
                                            ratio_denominator.append(hist_match[base_stat])

                        # Store rolling average
                        if values:
                            df_copy.at[idx, f'{prefix}_{stat.lower()}_r{window}'] = np.mean(values)

                        # Store rolling ratio
                        if stat.endswith('T') and len(ratio_numerator) > 0:
                            base_stat = stat[:-1]
                            total_numerator = sum(ratio_numerator)
                            total_denominator = sum(ratio_denominator)
                            if total_denominator > 0:
                                df_copy.at[idx, f'{prefix}_{base_stat.lower()}_ratio_r{window}'] = total_numerator / total_denominator

        # Progress indicator
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx:,} matches...")

    print("✅ Rolling stats computation completed!")
    return df_copy

def compute_league_draw_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical league average half-time draw rates"""
    df_copy = df.copy()

    # Sort by date
    df_copy = df_copy.sort_values('Date').reset_index(drop=True)

    # Initialize column
    df_copy['league_ht_draw_rate_historical'] = np.nan

    print("Computing historical league draw rates...")

    # For each match, compute draw rate from prior season(s)
    for idx, row in df_copy.iterrows():
        match_date = row['Date']
        league = row['league']

        # Get historical data from same league, prior to current match
        # Use data from at least 30 days ago to avoid recent bias
        historical_data = df_copy[
            (df_copy['Date'] < match_date - pd.Timedelta(days=30)) &
            (df_copy['league'] == league) &
            (df_copy['y_ht_draw'].notna())
        ]

        if len(historical_data) >= 50:  # Need reasonable sample size
            draw_rate = historical_data['y_ht_draw'].mean()
            df_copy.at[idx, 'league_ht_draw_rate_historical'] = draw_rate

        # Progress indicator
        if idx % 2000 == 0 and idx > 0:
            print(f"  Processed {idx:,} matches...")

    print("✅ League draw rates computation completed!")
    return df_copy

def load_and_process_all_data():
    """Load and process all downloaded CSV files into mega dataset"""

    # League metadata
    league_metadata = {
        # England
        "E0": {"name": "Premier League", "country": "England", "tier": 1},
        "E1": {"name": "Championship", "country": "England", "tier": 2},
        "E2": {"name": "League One", "country": "England", "tier": 3},
        "E3": {"name": "League Two", "country": "England", "tier": 4},
        "EC": {"name": "National League", "country": "England", "tier": 5},
        # Scotland
        "SC0": {"name": "Premiership", "country": "Scotland", "tier": 1},
        "SC1": {"name": "Championship", "country": "Scotland", "tier": 2},
        "SC2": {"name": "League One", "country": "Scotland", "tier": 3},
        "SC3": {"name": "League Two", "country": "Scotland", "tier": 4},
        # Germany
        "D1": {"name": "Bundesliga", "country": "Germany", "tier": 1},
        "D2": {"name": "2. Bundesliga", "country": "Germany", "tier": 2},
        # Italy
        "I1": {"name": "Serie A", "country": "Italy", "tier": 1},
        "I2": {"name": "Serie B", "country": "Italy", "tier": 2},
        # Spain
        "SP1": {"name": "La Liga", "country": "Spain", "tier": 1},
        "SP2": {"name": "Segunda División", "country": "Spain", "tier": 2},
        # France
        "F1": {"name": "Ligue 1", "country": "France", "tier": 1},
        "F2": {"name": "Ligue 2", "country": "France", "tier": 2},
        # Netherlands
        "N1": {"name": "Eredivisie", "country": "Netherlands", "tier": 1},
        # Belgium
        "B1": {"name": "Pro League", "country": "Belgium", "tier": 1},
        # Portugal
        "P1": {"name": "Primeira Liga", "country": "Portugal", "tier": 1},
        # Turkey
        "T1": {"name": "Süper Lig", "country": "Turkey", "tier": 1},
        # Greece
        "G1": {"name": "Super League", "country": "Greece", "tier": 1},
    }

    # Required columns for a valid match
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HTHG', 'HTAG']

    # Optional columns to grab if available
    optional_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
                     'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']

    # Stats to compute rolling averages for
    rolling_stats = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']

    base_dir = Path("data/raw_all")
    all_matches = []
    load_stats = {
        "total_files": 0,
        "loaded_files": 0,
        "skipped_files": 0,
        "total_matches": 0,
        "by_league": {},
        "by_country": {},
        "date_range": {"min": None, "max": None},
        "errors": []
    }

    print(f"🔍 Scanning for CSV files in {base_dir.absolute()}")

    # Find all CSV files
    csv_files = list(base_dir.rglob("*.csv"))
    load_stats["total_files"] = len(csv_files)

    print(f"📁 Found {len(csv_files):,} CSV files")
    print("📊 Loading and processing data...")

    for csv_file in csv_files:
        # Extract league and season from path
        league = csv_file.parent.name
        season = csv_file.stem

        try:
            # Try different encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                load_stats["errors"].append(f"Encoding error: {csv_file}")
                continue

            # Check if required columns exist
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                load_stats["skipped_files"] += 1
                load_stats["errors"].append(f"Missing columns {missing_required}: {csv_file}")
                continue

            # Filter to valid matches only (non-null required columns)
            valid_matches = df[required_cols + [col for col in optional_cols if col in df.columns]].copy()

            # Filter out rows where required columns are null
            initial_count = len(valid_matches)
            for col in required_cols:
                valid_matches = valid_matches[valid_matches[col].notna()]

            if len(valid_matches) == 0:
                load_stats["skipped_files"] += 1
                continue

            # Add metadata
            valid_matches['league'] = league
            valid_matches['season'] = season
            if league in league_metadata:
                valid_matches['country'] = league_metadata[league]['country']
                valid_matches['league_tier'] = league_metadata[league]['tier']
                valid_matches['league_name'] = league_metadata[league]['name']
            else:
                valid_matches['country'] = 'Unknown'
                valid_matches['league_tier'] = np.nan
                valid_matches['league_name'] = league

            # Normalize team names
            valid_matches['HomeTeam'] = valid_matches['HomeTeam'].apply(normalize_team_name)
            valid_matches['AwayTeam'] = valid_matches['AwayTeam'].apply(normalize_team_name)

            # Parse dates robustly
            valid_matches['Date'] = valid_matches['Date'].apply(parse_date_robust)

            # Remove matches where date parsing failed
            valid_matches = valid_matches[valid_matches['Date'].notna()]

            if len(valid_matches) > 0:
                all_matches.append(valid_matches)
                load_stats["loaded_files"] += 1
                load_stats["total_matches"] += len(valid_matches)

                # Update stats
                country = valid_matches['country'].iloc[0]
                load_stats["by_league"][league] = load_stats["by_league"].get(league, 0) + len(valid_matches)
                load_stats["by_country"][country] = load_stats["by_country"].get(country, 0) + len(valid_matches)

                # Update date range
                min_date = valid_matches['Date'].min()
                max_date = valid_matches['Date'].max()

                if load_stats["date_range"]["min"] is None or min_date < load_stats["date_range"]["min"]:
                    load_stats["date_range"]["min"] = min_date
                if load_stats["date_range"]["max"] is None or max_date > load_stats["date_range"]["max"]:
                    load_stats["date_range"]["max"] = max_date

                print(f"✅ {league}/{season}: {len(valid_matches):,} matches ({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})")

        except Exception as e:
            load_stats["errors"].append(f"Error loading {csv_file}: {str(e)}")
            load_stats["skipped_files"] += 1

    if not all_matches:
        print("❌ No valid match data found!")
        return None, load_stats

    # Combine all matches
    print(f"\n🔄 Combining {len(all_matches)} datasets...")
    combined_df = pd.concat(all_matches, ignore_index=True)

    # Sort by date for feature computation
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)

    print(f"✅ Combined dataset: {len(combined_df):,} matches")
    print(f"📅 Date range: {load_stats['date_range']['min'].strftime('%Y-%m-%d')} to {load_stats['date_range']['max'].strftime('%Y-%m-%d')}")

    # Create target variable first
    print("\n🎯 Creating target variable...")
    combined_df = create_target(combined_df)

    # Transform odds if available
    odds_cols = ['B365H', 'B365D', 'B365A']
    if any(col in combined_df.columns for col in odds_cols):
        print("💰 Transforming betting odds...")
        combined_df = transform_odds(combined_df)

    # Compute rolling form features
    print("\n📈 Computing rolling form features...")
    combined_df = compute_rolling_form(combined_df, window=5)

    # Compute rest days
    print("\n😴 Computing rest days...")
    combined_df = add_rest_days(combined_df)

    # Compute rolling stats for match statistics
    available_stats = [stat for stat in rolling_stats if stat in combined_df.columns or stat.replace('H', 'A') in combined_df.columns]
    if available_stats:
        print(f"\n📊 Computing rolling stats for: {', '.join(available_stats)}")
        combined_df = compute_rolling_stats(combined_df, available_stats, window=5)

    # Compute league draw rates
    print("\n🏆 Computing historical league draw rates...")
    combined_df = compute_league_draw_rates(combined_df)

    # Add categorical encodings
    print("\n🔢 Adding categorical features...")

    # League as categorical
    from sklearn.preprocessing import LabelEncoder
    le_league = LabelEncoder()
    combined_df['league_encoded'] = le_league.fit_transform(combined_df['league'])

    # Country as categorical
    le_country = LabelEncoder()
    combined_df['country_encoded'] = le_country.fit_transform(combined_df['country'])

    return combined_df, load_stats, {
        'league_encoder': le_league,
        'country_encoder': le_country,
        'league_metadata': league_metadata
    }

def save_mega_dataset(df: pd.DataFrame, stats: Dict, encoders: Dict):
    """Save the processed mega dataset and metadata"""

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset as parquet (efficient for large datasets)
    dataset_path = output_dir / "mega_dataset.parquet"
    print(f"💾 Saving mega dataset to {dataset_path}")
    df.to_parquet(dataset_path, index=False)

    # Prepare metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_matches": len(df),
        "total_files_processed": stats["loaded_files"],
        "total_files_found": stats["total_files"],
        "date_range": {
            "min": stats["date_range"]["min"].isoformat() if stats["date_range"]["min"] else None,
            "max": stats["date_range"]["max"].isoformat() if stats["date_range"]["max"] else None
        },
        "countries": list(stats["by_country"].keys()),
        "leagues": list(stats["by_league"].keys()),
        "match_counts_by_country": stats["by_country"],
        "match_counts_by_league": stats["by_league"],
        "columns": list(df.columns),
        "feature_completeness": {},
        "target_distribution": {},
        "errors": stats["errors"][:10],  # Only save first 10 errors
        "league_metadata": encoders["league_metadata"]
    }

    # Compute feature completeness
    for col in df.columns:
        if col not in ['league', 'season', 'country', 'HomeTeam', 'AwayTeam']:
            completeness = (1 - df[col].isnull().mean()) * 100
            metadata["feature_completeness"][col] = round(completeness, 1)

    # Target distribution
    if 'y_ht_draw' in df.columns:
        target_counts = df['y_ht_draw'].value_counts()
        metadata["target_distribution"] = {
            "ht_draws": int(target_counts.get(1, 0)),
            "ht_non_draws": int(target_counts.get(0, 0)),
            "ht_draw_rate": round(target_counts.get(1, 0) / len(df) * 100, 2)
        }

    # Save metadata
    metadata_path = output_dir / "mega_metadata.json"
    print(f"📋 Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return dataset_path, metadata_path, metadata

def print_summary(metadata: Dict):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("🏈 MEGA DATASET SUMMARY")
    print("="*80)

    print(f"📊 Total Matches: {metadata['total_matches']:,}")
    print(f"📁 Files Processed: {metadata['total_files_processed']:,} / {metadata['total_files_found']:,}")
    print(f"📅 Date Range: {metadata['date_range']['min'][:10]} to {metadata['date_range']['max'][:10]}")

    if metadata["target_distribution"]:
        print(f"🎯 Half-Time Draws: {metadata['target_distribution']['ht_draws']:,} ({metadata['target_distribution']['ht_draw_rate']}%)")

    print(f"\n🌍 Countries ({len(metadata['countries'])}):")
    for country, count in sorted(metadata['match_counts_by_country'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {country}: {count:,} matches")

    print(f"\n🏆 Leagues ({len(metadata['leagues'])}):")
    for league, count in sorted(metadata['match_counts_by_league'].items(), key=lambda x: x[1], reverse=True):
        country = next((m['country'] for m in metadata['league_metadata'].values() if league in metadata['league_metadata']), 'Unknown')
        print(f"  {league} ({country}): {count:,} matches")

    print(f"\n📈 Feature Completeness (top features):")
    sorted_features = sorted(metadata['feature_completeness'].items(), key=lambda x: x[1], reverse=True)
    for feature, completeness in sorted_features[:15]:
        print(f"  {feature}: {completeness}%")

    if len(metadata.get('errors', [])) > 0:
        print(f"\n⚠️  Errors ({len(metadata['errors'])}):")
        for error in metadata['errors'][:5]:
            print(f"  {error}")

    print("="*80)

def main():
    """Main execution function"""
    print("🚀 Building MEGA dataset from ALL football-data.co.uk CSV files")
    print("This may take a while for large datasets...\n")

    # Load and process all data
    combined_df, stats, encoders = load_and_process_all_data()

    if combined_df is None:
        print("❌ Failed to load any data. Check the raw_all directory.")
        return

    # Save the mega dataset
    dataset_path, metadata_path, metadata = save_mega_dataset(combined_df, stats, encoders)

    # Print comprehensive summary
    print_summary(metadata)

    print(f"\n🎉 Mega dataset successfully created!")
    print(f"💾 Dataset: {dataset_path}")
    print(f"📋 Metadata: {metadata_path}")
    print(f"🏈 Ready for half-time draw prediction with {len(combined_df):,} matches!")

if __name__ == "__main__":
    main()