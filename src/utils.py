"""
General utility functions for the soccer ML project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any

def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and concatenate raw CSV files from the data directory.
    
    Args:
        data_dir: Path to directory containing raw CSV files
    
    Returns:
        Combined DataFrame with all matches
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for file in csv_files:
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    print(f"✅ Loaded {file.name} with {encoding} encoding ({len(df)} rows)")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"⚠️  Skipping {file.name} - encoding issues")
                continue
                
            # Add season column based on filename if needed
            season = file.stem
            df['season'] = season
            
            # Only keep files with the required columns
            required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HTHG', 'HTAG']
            if all(col in df.columns for col in required_cols):
                dfs.append(df)
                print(f"✅ Added {file.name} to dataset")
            else:
                print(f"⚠️  Skipping {file.name} - missing required columns")
                
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid CSV files could be loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and ensure consistent data types.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    # Convert date column
    if 'Date' in df_copy.columns:
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    # Add month column for seasonality
    if 'Date' in df_copy.columns:
        df_copy['month'] = df_copy['Date'].dt.month
    
    return df_copy

def train_val_test_split(df: pd.DataFrame, 
                        train_frac: float = 0.7, 
                        val_frac: float = 0.15) -> tuple:
    """
    Split data chronologically into train, validation, and test sets.
    
    Args:
        df: DataFrame sorted by date
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def save_metadata(metadata: Dict[str, Any], filepath: str):
    """
    Save model metadata to JSON file.
    
    Args:
        metadata: Dictionary containing metadata
        filepath: Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load model metadata from JSON file.
    
    Args:
        filepath: Path to JSON metadata file
    
    Returns:
        Dictionary containing metadata
    """
    with open(filepath, 'r') as f:
        return json.load(f)