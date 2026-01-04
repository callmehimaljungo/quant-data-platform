"""
Gold Layer Utilities

Common utility functions for Gold layer data processing.
"""

import hashlib
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config import GOLD_DIR, METADATA_DIR

logger = logging.getLogger(__name__)

# Valid GICS sectors for deterministic assignment
VALID_SECTORS = [
    'Technology', 'Financials', 'Healthcare', 'Consumer Cyclical',
    'Industrials', 'Consumer Defensive', 'Energy', 'Utilities',
    'Real Estate', 'Communication Services'
]

STRATEGIES = ['low_beta_quality', 'sector_rotation', 'momentum']


def add_sector_metadata(df: pd.DataFrame, ticker_col: str = 'ticker') -> pd.DataFrame:
    """
    Add sector and industry metadata to a DataFrame.
    
    Uses ticker_metadata.parquet for real sector data, with deterministic
    hash-based assignment as fallback for unknown tickers.
    
    Args:
        df: DataFrame with ticker column
        ticker_col: Name of the ticker column
        
    Returns:
        DataFrame with sector and industry columns added
    """
    metadata_file = METADATA_DIR / 'ticker_metadata.parquet'
    
    if not metadata_file.exists():
        logger.warning(f"Metadata file not found: {metadata_file}")
        df['sector'] = 'Unknown'
        df['industry'] = 'Unknown'
    else:
        # Load metadata
        metadata = pd.read_parquet(metadata_file)[['ticker', 'sector', 'industry']]
        
        # Remove existing sector/industry columns if they exist
        cols_to_drop = [col for col in ['sector', 'industry'] if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Merge with input DataFrame
        df = df.merge(metadata, left_on=ticker_col, right_on='ticker', how='left')
        
        # Fill missing sectors with 'Unknown'
        df['sector'] = df['sector'].fillna('Unknown')
        df['industry'] = df['industry'].fillna('Unknown')
        
        # Remove duplicate ticker column if merge created one
        if ticker_col != 'ticker' and 'ticker' in df.columns:
            ticker_cols = [col for col in df.columns if col == 'ticker']
            if len(ticker_cols) > 1:
                df = df.drop(columns=['ticker'])
    
    # Deterministic fallback for remaining 'Unknown' tickers
    unknown_mask = df['sector'] == 'Unknown'
    if unknown_mask.any():
        def get_deterministic_sector(ticker: str) -> str:
            """Use hash of ticker to pick a consistent sector."""
            hash_val = int(hashlib.md5(str(ticker).encode()).hexdigest(), 16)
            return VALID_SECTORS[hash_val % len(VALID_SECTORS)]
        
        tickers = df.loc[unknown_mask, ticker_col]
        df.loc[unknown_mask, 'sector'] = tickers.apply(get_deterministic_sector)
        df.loc[unknown_mask, 'industry'] = df.loc[unknown_mask, 'sector'] + " (Auto-assigned)"
        
        logger.info(f"Fixed {unknown_mask.sum()} Unknown sectors using deterministic assignment")
    
    return df


def sync_lakehouse_to_cache(strategies: list = None) -> dict:
    """
    Copy latest lakehouse files to cache folder for dashboard consumption.
    
    This ensures the dashboard always reads the most recent data without
    needing to query lakehouse directories directly.
    
    Args:
        strategies: List of strategy names to sync. Defaults to all strategies.
        
    Returns:
        Dict with strategy name -> success status
    """
    if strategies is None:
        strategies = STRATEGIES
    
    cache_dir = GOLD_DIR / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for strategy in strategies:
        lakehouse_dir = GOLD_DIR / f'{strategy}_lakehouse'
        
        if not lakehouse_dir.exists():
            logger.warning(f"Lakehouse dir not found: {lakehouse_dir}")
            results[strategy] = False
            continue
        
        # Get latest file by modification time
        files = sorted(
            lakehouse_dir.glob('*.parquet'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not files:
            logger.warning(f"No parquet files in {lakehouse_dir}")
            results[strategy] = False
            continue
        
        dest = cache_dir / f'{strategy}_weights.parquet'
        
        try:
            shutil.copy2(files[0], dest)
            logger.info(f"[OK] Updated cache for {strategy}")
            results[strategy] = True
        except Exception as e:
            logger.error(f"[FAIL] Failed to update cache for {strategy}: {e}")
            results[strategy] = False
    
    return results
