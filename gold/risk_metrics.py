"""
Gold Layer: Per-Ticker Risk Metrics

Calculates VaR, Sharpe, Sortino, Max Drawdown, Beta, Alpha 
for each ticker and aggregates by sector.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
SILVER_LAKEHOUSE_PATH = SILVER_DIR / 'enriched_lakehouse'
SILVER_PARQUET_PATH = SILVER_DIR / 'enriched_stocks.parquet'

OUTPUT_TICKER_PATH = GOLD_DIR / 'risk_metrics_lakehouse'
OUTPUT_SECTOR_PATH = GOLD_DIR / 'sector_risk_metrics_lakehouse'

# Default Risk-free rate = 5% per year (US Treasury rate)
# NOTE: For more accurate results, use get_dynamic_risk_free_rate()
DEFAULT_RISK_FREE_RATE = 0.05
RISK_FREE_RATE = DEFAULT_RISK_FREE_RATE  # For backward compatibility

# Trading days per year
TRADING_DAYS = 252

# SPY ticker (S&P 500 ETF) - benchmark for Beta calculation
BENCHMARK_TICKER = 'SPY'

# Historical Fed Funds Rate by period (approximate averages)
# Source: Federal Reserve Economic Data (FRED)
HISTORICAL_RISK_FREE_RATES = {
    # (start_year, end_year): rate
    (1960, 1970): 0.045,  # 4.5%
    (1970, 1980): 0.075,  # 7.5%
    (1980, 1990): 0.095,  # 9.5%
    (1990, 2000): 0.055,  # 5.5%
    (2000, 2008): 0.035,  # 3.5%
    (2008, 2015): 0.002,  # 0.2% (near zero after 2008 crisis)
    (2015, 2020): 0.015,  # 1.5%
    (2020, 2022): 0.002,  # 0.2% (COVID era)
    (2022, 2030): 0.05,   # 5.0% (post-COVID tightening)
}


def get_dynamic_risk_free_rate(year: int) -> float:
    """
    Get approximate risk-free rate for a given year.
    
    Uses historical Fed Funds rate averages by period.
    More accurate than using a single static rate for all historical data.
    
    Args:
        year: Year to lookup rate for
        
    Returns:
        float: Approximate risk-free rate for that period
    """
    for (start, end), rate in HISTORICAL_RISK_FREE_RATES.items():
        if start <= year < end:
            return rate
    
    # Default to current rate if year not found
    return DEFAULT_RISK_FREE_RATE


# =============================================================================
# LOAD DATA (MEMORY OPTIMIZED)
# =============================================================================
# Only load columns needed for risk calculations
REQUIRED_COLUMNS = ['ticker', 'date', 'close', 'daily_return']


def load_silver_data() -> pd.DataFrame:
    """Load data from Silver layer (Lakehouse or Parquet) - MEMORY OPTIMIZED
    
    Only loads required columns to reduce memory by ~70%
    """
    import duckdb
    from utils import is_lakehouse_table, get_metadata_path
    import json
    
    # Try Lakehouse first
    if is_lakehouse_table(SILVER_LAKEHOUSE_PATH):
        logger.info(f"Loading from Silver Lakehouse: {SILVER_LAKEHOUSE_PATH}")
        
        # Get the data file path from metadata
        meta_path = get_metadata_path(SILVER_LAKEHOUSE_PATH)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        if not metadata['versions']:
            raise FileNotFoundError(f"No data found in: {SILVER_LAKEHOUSE_PATH}")
        
        version_info = metadata['versions'][-1]  # Latest version
        data_file = SILVER_LAKEHOUSE_PATH / version_info['file']
        
        # Load only required columns using DuckDB (memory efficient!)
        cols_str = ", ".join(REQUIRED_COLUMNS)
        logger.info(f"  Loading only columns: {REQUIRED_COLUMNS}")
        
        con = duckdb.connect()
        df = con.execute(f"SELECT {cols_str} FROM read_parquet('{data_file}')").fetchdf()
        con.close()
        
        logger.info(f"[OK] Loaded {len(df):,} rows (memory-optimized)")
        return df
        
    elif SILVER_PARQUET_PATH.exists():
        logger.info(f"Loading from Silver Parquet: {SILVER_PARQUET_PATH}")
        # Load only required columns
        logger.info(f"  Loading only columns: {REQUIRED_COLUMNS}")
        df = pd.read_parquet(SILVER_PARQUET_PATH, columns=REQUIRED_COLUMNS)
        
        # Data Quality Filter: Remove extreme outliers
        original_count = len(df)
        logger.info(f"  Before filtering: {original_count:,} rows")
        
        # Will filter after metrics are calculated
        return df
    else:
        raise FileNotFoundError("No Silver data found. Run silver/clean_delta.py first.")


def get_spy_returns(df: pd.DataFrame) -> Tuple[pd.Series, bool]:
    """Extract SPY returns as market benchmark"""
    spy_df = df[df['ticker'] == BENCHMARK_TICKER].copy()
    
    if len(spy_df) == 0:
        logger.warning(f"[WARN] {BENCHMARK_TICKER} not found. Beta/Alpha will be NaN.")
        return pd.Series(dtype=float), False
    
    spy_df = spy_df.sort_values('date').set_index('date')
    spy_returns = spy_df['daily_return'] / 100  # Convert to decimal
    
    logger.info(f"[OK] Loaded {BENCHMARK_TICKER}: {len(spy_returns):,} days")
    return spy_returns, True


# =============================================================================
# VECTORIZED RISK CALCULATIONS (OPTIMIZED - MEMORY EFFICIENT)
# =============================================================================
import gc

BATCH_SIZE = 50  # Increased from 25 since we load fewer columns now


def calculate_ticker_metrics_vectorized(df: pd.DataFrame, spy_returns: pd.Series, 
                                         has_spy: bool) -> pd.DataFrame:
    """
    Calculate risk metrics using vectorized operations with batch processing.
    Memory-optimized to handle large datasets without OOM.
    """
    logger.info("Calculating per-ticker risk metrics (memory-optimized)...")
    
    # Convert daily_return to decimal (in-place, no copy needed since we own the df)
    df['daily_return_decimal'] = df['daily_return'] / 100
    
    # Get unique tickers
    tickers = df['ticker'].unique()
    total_tickers = len(tickers)
    logger.info(f"  Processing {total_tickers:,} tickers in batches of {BATCH_SIZE}")
    
    # Group by ticker
    grouped = df.groupby('ticker')
    
    # Step 1: Basic aggregations (fast, memory-efficient)
    logger.info("  Step 1/5: Basic aggregations...")
    basic = grouped.agg({
        'date': ['min', 'max', 'count'],
        'sector': 'first',
        'daily_return_decimal': ['mean', 'std'],
        'close': ['first', 'last']
    })
    basic.columns = ['first_date', 'last_date', 'num_records', 'sector',
                     'mean_return', 'std_return', 'first_price', 'last_price']
    basic = basic.reset_index()
    
    logger.info(f"  Found {len(basic):,} tickers")
    
    # Step 2: Sharpe Ratio (vectorized - fast)
    logger.info("  Step 2/5: Sharpe Ratio...")
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    basic['sharpe_ratio'] = np.where(
        (basic['std_return'] > 0) & (basic['num_records'] >= 30),
        ((basic['mean_return'] - daily_rf) / basic['std_return']) * np.sqrt(TRADING_DAYS),
        np.nan
    )
    
    # Step 3: Volatility (vectorized)
    logger.info("  Step 3/5: Volatility...")
    basic['volatility'] = basic['std_return'] * np.sqrt(TRADING_DAYS)
    
    # Step 4: VaR 95% (vectorized)
    logger.info("  Step 4/5: VaR 95%...")
    var_95 = grouped['daily_return_decimal'].quantile(0.05).reset_index()
    var_95.columns = ['ticker', 'var_95']
    basic = basic.merge(var_95, on='ticker', how='left')
    
    # Free memory
    del var_95
    gc.collect()
    
    # Step 5: Batch processing for expensive operations (Max Drawdown, Sortino, Beta/Alpha)
    logger.info("  Step 5/5: Batch processing Max Drawdown, Sortino, Beta/Alpha...")
    
    # Initialize result columns
    basic['max_drawdown'] = np.nan
    basic['sortino_ratio'] = np.nan
    basic['beta_static'] = np.nan
    basic['beta_rolling_252'] = np.nan
    basic['alpha'] = np.nan
    
    # Process in batches
    for batch_idx in range(0, total_tickers, BATCH_SIZE):
        batch_tickers = tickers[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
        
        if batch_num % 5 == 1 or batch_num == total_batches:
            logger.info(f"    Batch {batch_num}/{total_batches} ({len(batch_tickers)} tickers)...")
        
        # Filter data for this batch
        batch_df = df[df['ticker'].isin(batch_tickers)]
        batch_grouped = batch_df.groupby('ticker')
        
        # Calculate Max Drawdown for batch
        for ticker in batch_tickers:
            try:
                ticker_data = batch_grouped.get_group(ticker)
                prices = ticker_data.sort_values('date')['close']
                
                if len(prices) >= 30:
                    peak = prices.expanding(min_periods=1).max()
                    drawdown = (prices - peak) / peak
                    max_dd = drawdown.min()
                    basic.loc[basic['ticker'] == ticker, 'max_drawdown'] = max_dd
                
                # Sortino
                returns = ticker_data['daily_return_decimal']
                if len(returns) >= 30:
                    downside = returns[returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        sortino = ((returns.mean() - daily_rf) / downside.std()) * np.sqrt(TRADING_DAYS)
                        basic.loc[basic['ticker'] == ticker, 'sortino_ratio'] = sortino
                
                # Beta/Alpha (if SPY available)
                if has_spy and len(ticker_data) >= 60:
                    stock_returns = ticker_data.set_index('date')['daily_return_decimal']
                    max_stock_date = stock_returns.index.max()
                    spy_filtered = spy_returns[spy_returns.index <= max_stock_date]
                    
                    aligned = pd.concat([stock_returns, spy_filtered], axis=1).dropna()
                    
                    if len(aligned) >= 60:
                        stock = aligned.iloc[:, 0]
                        market = aligned.iloc[:, 1]
                        
                        cov = stock.cov(market)
                        var = market.var()
                        
                        if var > 0:
                            beta_static = cov / var
                            basic.loc[basic['ticker'] == ticker, 'beta_static'] = beta_static
                            
                            # Rolling Beta (252 days)
                            if len(aligned) >= 252:
                                stock_recent = stock.tail(252)
                                market_recent = market.tail(252)
                                var_rolling = market_recent.var()
                                if var_rolling > 0:
                                    beta_rolling = stock_recent.cov(market_recent) / var_rolling
                                    basic.loc[basic['ticker'] == ticker, 'beta_rolling_252'] = beta_rolling
                            else:
                                basic.loc[basic['ticker'] == ticker, 'beta_rolling_252'] = beta_static
                            
                            # Alpha
                            expected = daily_rf + beta_static * (market.mean() - daily_rf)
                            alpha = (stock.mean() - expected) * TRADING_DAYS
                            basic.loc[basic['ticker'] == ticker, 'alpha'] = alpha
                            
            except Exception as e:
                logger.warning(f"    Error processing {ticker}: {e}")
                continue
        
        # Free batch memory
        del batch_df, batch_grouped
        gc.collect()
    
    # Create 'beta' as alias for rolling
    basic['beta'] = basic['beta_rolling_252']
    
    # Convert to percentages
    basic['var_95_pct'] = basic['var_95'] * 100
    basic['max_drawdown_pct'] = basic['max_drawdown'] * 100
    basic['volatility_pct'] = basic['volatility'] * 100
    basic['alpha_pct'] = basic['alpha'] * 100
    
    # Filter out tickers with insufficient data
    basic = basic[basic['num_records'] >= 30]
    
    # Data Quality Filter: Remove extreme outliers
    # These values indicate data errors or untradeable assets
    original_count = len(basic)
    
    basic = basic[
        (basic['max_drawdown'] > -0.95) &  # -95% in decimal
        (basic['volatility'] <= 4.0) &      # 400% in decimal  
        (basic['sharpe_ratio'].between(-10, 20))
    ]
    
    filtered_count = original_count - len(basic)
    logger.info(f"[FILTER] Removed {filtered_count:,} outliers ({filtered_count/original_count*100:.1f}%)")
    logger.info(f"[OK] Calculated metrics for {len(basic):,} quality tickers")
    
    # Final cleanup
    gc.collect()
    
    return basic


# =============================================================================
# PER-SECTOR AGGREGATES
# =============================================================================
def calculate_sector_risk_metrics(ticker_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk metrics by sector"""
    logger.info("Aggregating metrics by sector...")
    
    sector_agg = ticker_metrics.groupby('sector').agg({
        'ticker': 'count',
        'sharpe_ratio': 'mean',
        'sortino_ratio': 'mean',
        'volatility': 'mean',
        'beta': 'mean',
        'alpha': 'mean',
        'max_drawdown': 'mean',
        'var_95': 'mean'
    }).reset_index()
    
    sector_agg.columns = [
        'sector', 'num_tickers', 'avg_sharpe', 'avg_sortino', 
        'avg_volatility', 'avg_beta', 'avg_alpha', 'avg_max_drawdown', 'avg_var_95'
    ]
    
    sector_agg['avg_volatility_pct'] = sector_agg['avg_volatility'] * 100
    sector_agg['avg_max_drawdown_pct'] = sector_agg['avg_max_drawdown'] * 100
    sector_agg = sector_agg.sort_values('avg_sharpe', ascending=False)
    
    logger.info(f"[OK] Aggregated metrics for {len(sector_agg)} sectors")
    return sector_agg


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_risk_analysis() -> Dict[str, pd.DataFrame]:
    """Run complete risk metrics analysis"""
    from utils import pandas_to_lakehouse
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("GOLD LAYER: PER-TICKER RISK METRICS")
    logger.info("=" * 70)
    
    # 1. Load Silver data
    df = load_silver_data()
    logger.info(f"[OK] Loaded {len(df):,} rows")
    
    # Enrich with Sector Metadata to fix 'Unknown'
    from gold.utils import add_sector_metadata
    df = add_sector_metadata(df)
    
    logger.info(f"  Tickers: {df['ticker'].nunique():,}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 2. Get SPY benchmark
    spy_returns, has_spy = get_spy_returns(df)
    
    # 3. Calculate metrics (memory-optimized with batch processing)
    ticker_metrics = calculate_ticker_metrics_vectorized(df, spy_returns, has_spy)
    
    # Free memory - delete the large raw dataframe after processing
    del df
    del spy_returns
    gc.collect()
    logger.info("[OK] Freed raw data from memory")
    
    # 4. Sector aggregates
    sector_metrics = calculate_sector_risk_metrics(ticker_metrics)
    
    # 5. Save to Lakehouse
    logger.info("\nSaving to Lakehouse...")
    pandas_to_lakehouse(ticker_metrics, OUTPUT_TICKER_PATH, mode="overwrite")
    pandas_to_lakehouse(sector_metrics, OUTPUT_SECTOR_PATH, mode="overwrite")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info(" RISK METRICS COMPLETED ")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Output: {len(ticker_metrics):,} tickers, {len(sector_metrics)} sectors")
    logger.info("=" * 70)
    
    # Print samples
    print("\n--- Top 10 by Sharpe ---")
    print(ticker_metrics.nlargest(10, 'sharpe_ratio')[
        ['ticker', 'sector', 'sharpe_ratio', 'volatility_pct', 'beta']
    ].to_string(index=False))
    
    return {'ticker_metrics': ticker_metrics, 'sector_metrics': sector_metrics}


def main() -> int:
    """Main execution"""
    try:
        run_risk_analysis()
        logger.info("\n✅ Done! Next: python gold/portfolio.py")
        return 0
    except Exception as e:
        logger.error(f"[ERR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
