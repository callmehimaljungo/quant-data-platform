"""
Gold Layer: Sector Analysis & Risk Metrics
Aggregated analytics for quantitative trading

Features:
- Sector performance metrics
- Risk metrics (Sharpe ratio, volatility, max drawdown)
- Portfolio statistics
- Time-series aggregations

Output: Aggregated datasets ready for visualization and backtesting

Usage:
    python gold/sector_analysis.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import gc

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
SILVER_LAKEHOUSE_PATH = SILVER_DIR / 'enriched_lakehouse'
SILVER_PARQUET_PATH = SILVER_DIR / 'enriched_stocks.parquet'

GOLD_LAKEHOUSE_PATH = GOLD_DIR / 'analytics_lakehouse'
GOLD_METRICS_PATH = GOLD_DIR / 'metrics_lakehouse'

# Memory optimization: columns required for sector analysis
REQUIRED_COLUMNS = ['ticker', 'date', 'close', 'daily_return', 'sector', 'volume']


# =============================================================================
# LOAD DATA
# =============================================================================
def load_silver_data() -> pd.DataFrame:
    """Load data from Silver layer - MEMORY OPTIMIZED"""
    import duckdb
    from utils import is_lakehouse_table, get_metadata_path
    import json
    
    logger.info("Loading Silver data (MEMORY OPTIMIZED)...")
    logger.info(f"  Loading only columns: {REQUIRED_COLUMNS}")
    
    if is_lakehouse_table(SILVER_LAKEHOUSE_PATH):
        logger.info(f"Loading from Silver Lakehouse: {SILVER_LAKEHOUSE_PATH}")
        meta_path = get_metadata_path(SILVER_LAKEHOUSE_PATH)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata['versions']:
            version_info = metadata['versions'][-1]
            data_file = SILVER_LAKEHOUSE_PATH / version_info['file']
            
            cols_str = ", ".join(REQUIRED_COLUMNS)
            con = duckdb.connect()
            df = con.execute(f"SELECT {cols_str} FROM read_parquet('{data_file}')").fetchdf()
            con.close()
            
            logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            return df
    
    if SILVER_PARQUET_PATH.exists():
        logger.info(f"Loading from Silver Parquet: {SILVER_PARQUET_PATH}")
        df = pd.read_parquet(SILVER_PARQUET_PATH, columns=REQUIRED_COLUMNS)
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return df
    
    raise FileNotFoundError("No Silver data found")


# =============================================================================
# RISK METRICS CALCULATIONS
# =============================================================================
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe Ratio
    
    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    """
    if returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean() * 252  # Annualized
    std_return = returns.std() * np.sqrt(252)  # Annualized
    
    return (mean_return - risk_free_rate) / std_return


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate Maximum Drawdown
    
    Max Drawdown = (Trough - Peak) / Peak
    """
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility
    """
    return returns.std() * np.sqrt(252)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio (only considers downside volatility)
    """
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean() * 252
    downside_std = downside_returns.std() * np.sqrt(252)
    
    return (mean_return - risk_free_rate) / downside_std


# =============================================================================
# SECTOR METRICS
# =============================================================================
def calculate_sector_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metrics by sector
    """
    logger.info("Calculating sector metrics...")
    
    # If sector is all 'Unknown', create mock sectors based on first letter of ticker
    if df['sector'].nunique() == 1 and df['sector'].iloc[0] == 'Unknown':
        logger.info("No sector data, creating mock sectors based on ticker patterns")
        df['sector'] = df['ticker'].apply(lambda x: f"Sector_{x[0].upper()}" if x else "Unknown")
    
    sector_metrics = []
    
    for sector in df['sector'].unique():
        sector_df = df[df['sector'] == sector]
        
        # Get daily returns
        daily_returns = sector_df.groupby('date')['daily_return'].mean() / 100  # Convert to decimal
        
        # Get average close price series for max drawdown
        avg_prices = sector_df.groupby('date')['close'].mean()
        
        metrics = {
            'sector': sector,
            'num_tickers': sector_df['ticker'].nunique(),
            'total_records': len(sector_df),
            'avg_daily_return': daily_returns.mean() * 100,  # As percentage
            'total_return': ((1 + daily_returns).prod() - 1) * 100,  # As percentage
            'volatility': calculate_volatility(daily_returns) * 100,  # As percentage
            'sharpe_ratio': calculate_sharpe_ratio(daily_returns),
            'sortino_ratio': calculate_sortino_ratio(daily_returns),
            'max_drawdown': calculate_max_drawdown(avg_prices) * 100,  # As percentage
            'min_date': sector_df['date'].min(),
            'max_date': sector_df['date'].max()
        }
        
        sector_metrics.append(metrics)
    
    result = pd.DataFrame(sector_metrics)
    logger.info(f"[OK] Calculated metrics for {len(result)} sectors")
    
    return result


# =============================================================================
# TICKER METRICS
# =============================================================================
def calculate_ticker_metrics(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Calculate metrics by ticker (top N by data count)
    """
    logger.info(f"Calculating ticker metrics (top {top_n})...")
    
    # Get top N tickers by data count
    top_tickers = df['ticker'].value_counts().head(top_n).index.tolist()
    df_top = df[df['ticker'].isin(top_tickers)]
    
    ticker_metrics = []
    
    for ticker in top_tickers:
        ticker_df = df_top[df_top['ticker'] == ticker].sort_values('date')
        
        daily_returns = ticker_df['daily_return'] / 100
        prices = ticker_df['close']
        
        metrics = {
            'ticker': ticker,
            'sector': ticker_df['sector'].iloc[0] if 'sector' in ticker_df.columns else 'Unknown',
            'num_records': len(ticker_df),
            'first_date': ticker_df['date'].min(),
            'last_date': ticker_df['date'].max(),
            'first_price': prices.iloc[0],
            'last_price': prices.iloc[-1],
            'price_change_pct': ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100 if prices.iloc[0] > 0 else 0,
            'avg_daily_return': daily_returns.mean() * 100,
            'volatility': calculate_volatility(daily_returns) * 100,
            'sharpe_ratio': calculate_sharpe_ratio(daily_returns),
            'max_drawdown': calculate_max_drawdown(prices) * 100,
            'avg_volume': ticker_df['volume'].mean()
        }
        
        ticker_metrics.append(metrics)
    
    result = pd.DataFrame(ticker_metrics)
    result = result.sort_values('sharpe_ratio', ascending=False)
    logger.info(f"[OK] Calculated metrics for {len(result)} tickers")
    
    return result


# =============================================================================
# TIME SERIES AGGREGATIONS
# =============================================================================
def calculate_monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly aggregated performance
    """
    logger.info("Calculating monthly performance...")
    
    df['year_month'] = df['date'].dt.to_period('M')
    
    monthly = df.groupby('year_month').agg({
        'daily_return': ['mean', 'std', 'count'],
        'close': 'mean',
        'volume': 'sum',
        'ticker': 'nunique'
    }).reset_index()
    
    monthly.columns = ['year_month', 'avg_return', 'return_std', 'trading_days', 
                       'avg_price', 'total_volume', 'active_tickers']
    
    monthly['year_month'] = monthly['year_month'].astype(str)
    
    logger.info(f"[OK] Calculated {len(monthly)} months of performance")
    
    return monthly


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_sector_analysis() -> Dict[str, pd.DataFrame]:
    """
    Run complete Gold Layer analysis
    """
    from utils import pandas_to_lakehouse, show_history
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("GOLD LAYER: SECTOR ANALYSIS & RISK METRICS")
    logger.info("=" * 70)
    
    # Load Silver data
    df = load_silver_data()
    logger.info(f"[OK] Loaded {len(df):,} rows from Silver layer")
    
    # Enrich with Sector Metadata
    from gold.utils import add_sector_metadata
    df = add_sector_metadata(df)
    logger.info(f"Enriched with sector metadata. Unknown count: {len(df[df['sector']=='Unknown'])}")
    
    # Calculate metrics
    results = {}
    
    # 1. Sector metrics
    sector_metrics = calculate_sector_metrics(df)
    results['sector_metrics'] = sector_metrics
    
    # 2. Ticker metrics (top 100)
    ticker_metrics = calculate_ticker_metrics(df, top_n=100)
    results['ticker_metrics'] = ticker_metrics
    
    # 3. Monthly performance
    monthly_perf = calculate_monthly_performance(df)
    results['monthly_performance'] = monthly_perf
    
    # Save results to Lakehouse
    logger.info("\nSaving results to Gold Lakehouse...")
    
    # Save sector metrics
    sector_path = GOLD_DIR / 'sector_metrics_lakehouse'
    pandas_to_lakehouse(sector_metrics, sector_path, mode="overwrite")
    
    # Save ticker metrics
    ticker_path = GOLD_DIR / 'ticker_metrics_lakehouse'
    pandas_to_lakehouse(ticker_metrics, ticker_path, mode="overwrite")
    
    # Save monthly performance
    monthly_path = GOLD_DIR / 'monthly_performance_lakehouse'
    pandas_to_lakehouse(monthly_perf, monthly_path, mode="overwrite")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info(" GOLD LAYER COMPLETED ")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Outputs:")
    logger.info(f"  - Sector metrics: {len(sector_metrics)} sectors")
    logger.info(f"  - Ticker metrics: {len(ticker_metrics)} tickers")
    logger.info(f"  - Monthly performance: {len(monthly_perf)} months")
    logger.info("=" * 70)
    
    # Print summary tables
    logger.info("\n--- Top 10 Sectors by Sharpe Ratio ---")
    print(sector_metrics.nlargest(10, 'sharpe_ratio')[
        ['sector', 'num_tickers', 'sharpe_ratio', 'volatility', 'total_return']
    ].to_string(index=False))
    
    logger.info("\n--- Top 10 Tickers by Sharpe Ratio ---")
    print(ticker_metrics.head(10)[
        ['ticker', 'sector', 'sharpe_ratio', 'volatility', 'price_change_pct']
    ].to_string(index=False))
    
    return results


def main():
    """Main execution"""
    logger.info("")
    logger.info(" GOLD LAYER - ANALYTICS")
    logger.info("")
    
    try:
        results = run_sector_analysis()
        
        logger.info("")
        logger.info("✅ Gold Layer completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  - View results in data/gold/")
        logger.info("  - Run backtesting with: python backtest/run.py")
        logger.info("  - Launch dashboard with: streamlit run dashboard/app.py")
        return 0
        
    except Exception as e:
        logger.error(f"[ERR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
