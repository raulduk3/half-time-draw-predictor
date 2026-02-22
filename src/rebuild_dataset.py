"""
Rebuild dataset with latest data including the new 2025-26 season data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our feature engineering functions
from features import compute_rolling_form, add_rest_days, transform_odds, create_target
from utils import load_raw_data, normalize_columns

def rebuild_dataset():
    """Rebuild the dataset with latest data."""
    print("🔄 Rebuilding Dataset with Latest Data")
    print("=" * 50)

    # Load raw data
    print("📊 Loading raw CSV files...")
    df_raw = load_raw_data('data/raw')

    print(f"Raw data shape: {df_raw.shape}")
    print(f"Date range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")

    # Normalize columns
    print("\n🔧 Normalizing columns...")
    df = normalize_columns(df_raw)

    # Clean data - remove rows with missing team names
    print("🧹 Cleaning data...")
    initial_rows = len(df)
    df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
    print(f"  Removed {initial_rows - len(df)} rows with missing team names")

    # Create target variable
    print("🎯 Creating target variable...")
    df = create_target(df)

    # Transform odds
    print("📈 Transforming betting odds...")
    df = transform_odds(df)

    # Add rest days
    print("⏰ Adding rest days...")
    df = add_rest_days(df)

    # Compute rolling form
    print("📊 Computing rolling form (this may take a few minutes)...")
    df = compute_rolling_form(df)

    # Clean and prepare final dataset
    print("\n🧹 Final data cleaning...")

    # Keep only matches with required columns
    required_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'y_ht_draw',
        'home_gf_r5', 'home_ga_r5', 'home_gd_r5',
        'away_gf_r5', 'away_ga_r5', 'away_gd_r5',
        'log_home_win_odds', 'log_draw_odds', 'log_away_win_odds',
        'home_days_since_last', 'away_days_since_last', 'month'
    ]

    df_final = df[required_cols].copy()

    # Sort by date
    df_final = df_final.sort_values('Date').reset_index(drop=True)

    # Calculate statistics
    total_matches = len(df_final)
    missing_values = df_final.isnull().sum().sum()
    draw_rate = df_final['y_ht_draw'].mean()
    total_draws = df_final['y_ht_draw'].sum()

    print(f"\n📈 Dataset Statistics:")
    print(f"  Total matches: {total_matches:,}")
    print(f"  Date range: {df_final['Date'].min().date()} to {df_final['Date'].max().date()}")
    print(f"  Draw rate: {draw_rate:.1%}")
    print(f"  Total draws: {total_draws:,}")
    print(f"  Missing values: {missing_values:,}")

    # Save dataset
    print("\n💾 Saving updated dataset...")
    df_final.to_parquet('data/processed/dataset.parquet', index=False)

    # Create metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'num_matches': int(total_matches),
        'date_range': {
            'start': df_final['Date'].min().isoformat(),
            'end': df_final['Date'].max().isoformat()
        },
        'features': [
            'home_gf_r5', 'home_ga_r5', 'home_gd_r5',
            'away_gf_r5', 'away_ga_r5', 'away_gd_r5',
            'log_home_win_odds', 'log_draw_odds', 'log_away_win_odds',
            'home_days_since_last', 'away_days_since_last', 'month'
        ],
        'target': 'y_ht_draw',
        'target_distribution': {
            'draw_rate': float(draw_rate),
            'total_draws': int(total_draws)
        },
        'data_quality': {
            'missing_values': int(missing_values),
            'duplicates': int(df_final.duplicated().sum())
        }
    }

    with open('data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Dataset rebuilt successfully!")
    print(f"  📁 Saved: data/processed/dataset.parquet")
    print(f"  📁 Saved: data/processed/metadata.json")

    return df_final

if __name__ == "__main__":
    rebuild_dataset()