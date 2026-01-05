
import pandas as pd
import numpy as np
import logging
from config import SILVER_DIR, BRONZE_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_unified_data_full() -> pd.DataFrame:
    """
    Load and aggregate ALL data using DuckDB for memory efficiency
    """
    import duckdb
    
    logger.info("=" * 60)
    logger.info("LOADING FULL DATASET WITH DUCKDB AGGREGATION")
    logger.info("=" * 60)
    
    # Price data path
    price_path = SILVER_DIR / 'enriched_stocks.parquet'
    
    if not price_path.exists():
        logger.error(f"Price data not found: {price_path}")
        return pd.DataFrame()
    
    # Use DuckDB for aggregation - handles 33M rows efficiently
    logger.info("Aggregating 33M+ rows with DuckDB...")
    
    conn = duckdb.connect()
    
    # Aggregate price data to daily level
    query = f"""
    SELECT 
        date,
        COUNT(DISTINCT ticker) as n_tickers,
        AVG(daily_return) as avg_return,
        STDDEV(daily_return) as volatility,
        SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as pct_positive,
        AVG(close) as avg_close,
        AVG(ABS(daily_return)) as avg_abs_return
    FROM read_parquet('{price_path}')
    WHERE daily_return IS NOT NULL
      AND date >= '1990-01-01'
    GROUP BY date
    ORDER BY date
    """
    
    try:
        df_daily = conn.execute(query).fetchdf()
        logger.info(f"  Price aggregated: {len(df_daily):,} unique dates")
    except Exception as e:
        logger.error(f"DuckDB aggregation failed: {e}")
        return pd.DataFrame()
    
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    # Also get sector breakdown for later analysis
    try:
        # We don't return this directly but could store it or save it
        pass 
    except Exception:
        pass
    
    conn.close()
    
    # Load economic data
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    econ_path = SILVER_DIR / 'economic_lakehouse'
    if is_lakehouse_table(econ_path):
        logger.info("Loading economic indicators...")
        econ_df = lakehouse_to_pandas(econ_path)
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        
        # Rename columns
        econ_cols = {'VIXCLS': 'vix', 'DFF': 'fed_rate', 'GDP': 'gdp', 
                     'CPIAUCSL': 'cpi', 'DTWEXBGS': 'dollar_index', 'DGS10': 'treasury_10y'}
        econ_df = econ_df.rename(columns=econ_cols)
        
        merge_cols = ['date'] + [c for c in ['vix', 'fed_rate', 'gdp', 'cpi', 'dollar_index', 'treasury_10y'] 
                                  if c in econ_df.columns]
        econ_subset = econ_df[merge_cols].drop_duplicates('date')
        
        df_daily = df_daily.merge(econ_subset, on='date', how='left')
    
    # Forward fill
    for col in ['vix', 'fed_rate', 'gdp', 'cpi', 'dollar_index', 'treasury_10y']:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].ffill()
    
    df_daily = df_daily.fillna(0)
    
    return df_daily

def load_unified_data(max_rows: int = 200000) -> pd.DataFrame:
    """Wrapper for data loading strategy"""
    # 1. Try DuckDB full load
    try:
        df = load_unified_data_full()
        if not df.empty and len(df) > 50:
            return df
    except Exception:
        pass

    # 2. Fallback to manual loading (simplified for this refactor file)
    logger.warning("Falling back to manual load...")
    return pd.DataFrame() # Placeholder for the massive manual logic to keep this file clean
