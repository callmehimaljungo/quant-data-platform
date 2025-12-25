"""
Silver Layer: Join All Schemas
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Join price data with metadata (sector, industry, market_cap)
- Join with benchmark data (SPY returns for beta calculation)
- Join with economic indicators (market regime context)
- Filter to intersection of tickers (only tickers with complete data)

Business Context:
- Creates unified dataset with all enrichment
- Enables comprehensive analysis across data dimensions
- Implements the "intersection filtering" strategy

Schema Join Strategy:
- prices LEFT JOIN metadata ON ticker
- prices LEFT JOIN benchmarks ON date
- prices LEFT JOIN economic ON date
- Filter to tickers appearing in metadata (intersection)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Input paths
PRICES_SILVER = SILVER_DIR / 'enriched_stocks.parquet'
METADATA_SILVER = SILVER_DIR / 'metadata_lakehouse'
NEWS_SILVER = SILVER_DIR / 'news_lakehouse'
ECONOMIC_SILVER = SILVER_DIR / 'economic_lakehouse'
BENCHMARKS_BRONZE = BRONZE_DIR / 'benchmarks_lakehouse'

# Output path
OUTPUT_DIR = SILVER_DIR / 'unified_lakehouse'


# =============================================================================
# LOADER FUNCTIONS
# =============================================================================

def load_prices() -> pd.DataFrame:
    """Load enriched price data"""
    if not PRICES_SILVER.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_SILVER}")
    
    df = pd.read_parquet(PRICES_SILVER)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"[OK] Loaded prices: {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    return df


def load_metadata() -> Optional[pd.DataFrame]:
    """Load processed metadata"""
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    if not is_lakehouse_table(METADATA_SILVER):
        logger.warning(f"[WARN] Metadata not found: {METADATA_SILVER}")
        return None
    
    df = lakehouse_to_pandas(METADATA_SILVER)
    
    # Select relevant columns
    cols = ['ticker', 'company_name', 'sector', 'industry', 'market_cap', 
            'market_cap_category', 'exchange', 'country']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()
    
    # Remove duplicates (keep latest)
    if 'ticker' in df.columns:
        df = df.drop_duplicates(subset=['ticker'], keep='last')
    
    logger.info(f"[OK] Loaded metadata: {len(df):,} tickers")
    return df


def load_benchmarks() -> Optional[pd.DataFrame]:
    """Load benchmark data (SPY for beta calculation)"""
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    if not is_lakehouse_table(BENCHMARKS_BRONZE):
        logger.warning(f"[WARN] Benchmarks not found: {BENCHMARKS_BRONZE}")
        return None
    
    df = lakehouse_to_pandas(BENCHMARKS_BRONZE)
    
    # Filter to SPY only (main benchmark)
    spy_df = df[df['ticker'] == 'SPY'].copy()
    
    if len(spy_df) == 0:
        logger.warning("[WARN] SPY data not found in benchmarks")
        return None
    
    # Calculate SPY daily return
    spy_df = spy_df.sort_values('date')
    spy_df['spy_return'] = spy_df['close'].pct_change()
    spy_df['spy_close'] = spy_df['close']
    
    # Select relevant columns
    spy_df = spy_df[['date', 'spy_return', 'spy_close']].copy()
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    
    logger.info(f"[OK] Loaded SPY benchmark: {len(spy_df):,} days")
    return spy_df


def load_economic() -> Optional[pd.DataFrame]:
    """Load processed economic indicators"""
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    if not is_lakehouse_table(ECONOMIC_SILVER):
        logger.warning(f"[WARN] Economic data not found: {ECONOMIC_SILVER}")
        return None
    
    df = lakehouse_to_pandas(ECONOMIC_SILVER)
    
    # Select relevant columns
    cols = ['date', 'vix', 'fed_funds_rate', 'treasury_10y', 'market_regime',
            'yield_curve_slope', 'unemployment_rate']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()
    
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"[OK] Loaded economic: {len(df):,} days")
    return df


def load_news_aggregated() -> Optional[pd.DataFrame]:
    """Load aggregated news sentiment"""
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    if not is_lakehouse_table(NEWS_SILVER):
        logger.warning(f"[WARN] News data not found: {NEWS_SILVER}")
        return None
    
    df = lakehouse_to_pandas(NEWS_SILVER)
    
    # Aggregate by ticker-date
    agg_df = df.groupby(['date', 'ticker']).agg({
        'news_count': 'sum',
        'avg_sentiment': 'mean',
        'positive_count': 'sum',
        'negative_count': 'sum'
    }).reset_index()
    
    agg_df['date'] = pd.to_datetime(agg_df['date'])
    
    # Rename columns to avoid conflicts
    agg_df = agg_df.rename(columns={
        'news_count': 'daily_news_count',
        'avg_sentiment': 'daily_sentiment'
    })
    
    logger.info(f"[OK] Loaded news: {len(agg_df):,} ticker-dates")
    return agg_df


# =============================================================================
# JOIN FUNCTIONS
# =============================================================================

def join_with_metadata(
    prices_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Join prices with metadata on ticker
    
    Updates sector, industry if better data available in metadata
    """
    if metadata_df is None:
        logger.info("Skipping metadata join (no data)")
        return prices_df
    
    logger.info("Joining with metadata...")
    
    initial_rows = len(prices_df)
    
    # Identify columns to update/add
    meta_cols = [c for c in metadata_df.columns if c != 'ticker']
    
    # For columns already in prices, suffix them
    for col in meta_cols:
        if col in prices_df.columns:
            # Keep metadata version if it's better
            metadata_df = metadata_df.rename(columns={col: f'{col}_meta'})
    
    # Left join
    merged = prices_df.merge(
        metadata_df,
        on='ticker',
        how='left'
    )
    
    # Update sector from metadata if available
    if 'sector_meta' in merged.columns:
        merged['sector'] = merged['sector_meta'].fillna(merged.get('sector', 'Unknown'))
        merged = merged.drop(columns=['sector_meta'])
    
    if 'industry_meta' in merged.columns:
        merged['industry'] = merged['industry_meta'].fillna(merged.get('industry', 'Unknown'))
        merged = merged.drop(columns=['industry_meta'])
    
    logger.info(f"  Rows: {initial_rows:,} → {len(merged):,}")
    
    return merged


def join_with_benchmarks(
    prices_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Join prices with SPY benchmark on date
    
    Adds spy_return column for beta calculation
    """
    if benchmark_df is None:
        logger.info("Skipping benchmark join (no data)")
        prices_df['spy_return'] = np.nan
        return prices_df
    
    logger.info("Joining with benchmarks (SPY)...")
    
    initial_rows = len(prices_df)
    
    # Left join on date
    merged = prices_df.merge(
        benchmark_df,
        on='date',
        how='left'
    )
    
    # Count how many got spy_return
    spy_coverage = merged['spy_return'].notna().sum()
    coverage_pct = spy_coverage / len(merged) * 100
    
    logger.info(f"  SPY coverage: {spy_coverage:,}/{len(merged):,} ({coverage_pct:.1f}%)")
    
    return merged


def join_with_economic(
    prices_df: pd.DataFrame,
    economic_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Join prices with economic indicators on date
    
    Adds VIX, rates, market regime
    """
    if economic_df is None:
        logger.info("Skipping economic join (no data)")
        prices_df['market_regime'] = 'unknown'
        return prices_df
    
    logger.info("Joining with economic indicators...")
    
    initial_rows = len(prices_df)
    
    # Left join on date
    merged = prices_df.merge(
        economic_df,
        on='date',
        how='left'
    )
    
    # Forward fill economic data (it doesn't change every day)
    econ_cols = ['vix', 'fed_funds_rate', 'treasury_10y', 'market_regime', 'yield_curve_slope']
    for col in econ_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    
    # Fill remaining nulls
    if 'market_regime' in merged.columns:
        merged['market_regime'] = merged['market_regime'].fillna('unknown')
    
    coverage = merged['vix'].notna().sum() if 'vix' in merged.columns else 0
    coverage_pct = coverage / len(merged) * 100 if len(merged) > 0 else 0
    
    logger.info(f"  Economic coverage: {coverage:,}/{len(merged):,} ({coverage_pct:.1f}%)")
    
    return merged


def join_with_news(
    prices_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Join prices with news sentiment on ticker+date
    """
    if news_df is None:
        logger.info("Skipping news join (no data)")
        prices_df['daily_sentiment'] = np.nan
        prices_df['daily_news_count'] = 0
        return prices_df
    
    logger.info("Joining with news sentiment...")
    
    initial_rows = len(prices_df)
    
    # Left join on ticker and date
    merged = prices_df.merge(
        news_df,
        on=['date', 'ticker'],
        how='left'
    )
    
    # Fill missing news counts with 0
    merged['daily_news_count'] = merged['daily_news_count'].fillna(0).astype(int)
    
    # Count tickers with news
    tickers_with_news = merged[merged['daily_news_count'] > 0]['ticker'].nunique()
    
    logger.info(f"  Tickers with news: {tickers_with_news:,}")
    
    return merged


# =============================================================================
# TICKER FILTERING
# =============================================================================

def filter_to_intersection(
    df: pd.DataFrame,
    min_sources: int = 2
) -> pd.DataFrame:
    """
    Filter to tickers that appear in multiple data sources
    
    Args:
        df: Unified DataFrame
        min_sources: Minimum number of data sources required
        
    Returns:
        Filtered DataFrame
    """
    from utils.ticker_universe import get_universe
    
    logger.info(f"Filtering to tickers with >= {min_sources} data sources...")
    
    universe = get_universe()
    summary = universe.get_summary()
    
    if summary['total_sources'] < min_sources:
        logger.warning(f"Only {summary['total_sources']} sources available")
        return df
    
    # Get tickers appearing in multiple sources
    ticker_counts = {}
    for source, tickers in universe.sources.items():
        for t in tickers:
            ticker_counts[t] = ticker_counts.get(t, 0) + 1
    
    valid_tickers = {t for t, c in ticker_counts.items() if c >= min_sources}
    
    # Filter
    initial_rows = len(df)
    initial_tickers = df['ticker'].nunique()
    
    df = df[df['ticker'].isin(valid_tickers)].copy()
    
    final_rows = len(df)
    final_tickers = df['ticker'].nunique()
    
    logger.info(f"  Tickers: {initial_tickers:,} → {final_tickers:,}")
    logger.info(f"  Rows: {initial_rows:,} → {final_rows:,}")
    
    return df


# =============================================================================
# MAIN JOIN FUNCTION
# =============================================================================

def join_all_schemas(
    filter_intersection: bool = True,
    min_sources: int = 2
) -> pd.DataFrame:
    """
    Main function to join all schemas together
    
    Args:
        filter_intersection: Whether to filter to intersection of sources
        min_sources: Minimum sources for intersection filtering
        
    Returns:
        Unified DataFrame with all joins applied
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("SILVER LAYER: SCHEMA JOINING")
    logger.info("=" * 70)
    
    # Step 1: Load all data
    logger.info("\n Loading Data Sources...")
    prices_df = load_prices()
    metadata_df = load_metadata()
    benchmark_df = load_benchmarks()
    economic_df = load_economic()
    news_df = load_news_aggregated()
    
    # Step 2: Join all schemas
    logger.info("\n🔗 Joining Schemas...")
    df = prices_df
    df = join_with_metadata(df, metadata_df)
    df = join_with_benchmarks(df, benchmark_df)
    df = join_with_economic(df, economic_df)
    df = join_with_news(df, news_df)
    
    # Step 3: Filter to intersection (optional)
    if filter_intersection:
        logger.info("\n🎯 Filtering to Intersection...")
        df = filter_to_intersection(df, min_sources=min_sources)
    
    # Step 4: Add metadata
    df['unified_at'] = datetime.now()
    df['data_version'] = 'unified_v1'
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 70)
    logger.info("SCHEMA JOINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Output rows: {len(df):,}")
    logger.info(f"  Output columns: {len(df.columns)}")
    logger.info(f"  Unique tickers: {df['ticker'].nunique():,}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Duration: {duration:.2f} seconds")
    
    # Column summary
    logger.info("\nColumns added from joins:")
    join_cols = ['spy_return', 'vix', 'market_regime', 'daily_sentiment', 
                 'company_name', 'market_cap_category']
    for col in join_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            logger.info(f"  {col}: {non_null:,} values ({pct:.1f}%)")
    
    return df


def save_to_lakehouse(df: pd.DataFrame) -> str:
    """Save unified data to Silver Lakehouse"""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"[OK] Saved to {path}")
    return path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(filter_intersection: bool = True, min_sources: int = 1):
    """CLI entry point."""
    logger.info("")
    logger.info(" SILVER LAYER: SCHEMA JOIN")
    logger.info("")
    
    try:
        # Join all schemas
        df = join_all_schemas(
            filter_intersection=filter_intersection,
            min_sources=min_sources
        )
        
        # Save to Lakehouse
        save_to_lakehouse(df)
        
        logger.info("")
        logger.info("✅ Schema joining completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info("")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"[ERR] Required data not found: {e}")
        logger.error("[ERR] Run silver/clean.py first")
        return 1
        
    except Exception as e:
        logger.error("")
        logger.error(f"[ERR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("")
        return 1


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    no_filter = '--no-filter' in sys.argv
    min_sources = 1
    
    for arg in sys.argv[1:]:
        if arg.startswith('--min-sources='):
            min_sources = int(arg.split('=')[1])
    
    exit(main(
        filter_intersection=not no_filter,
        min_sources=min_sources
    ))
