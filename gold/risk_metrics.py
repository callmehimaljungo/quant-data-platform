"""
Gold Layer: Per-Ticker Risk Metrics (Optimized Version)
Calculate individual stock risk metrics for portfolio construction

Purpose:
- Calculate VaR, Sharpe, Sortino, Max Drawdown, Volatility for each ticker
- Calculate Beta and Alpha vs SPY benchmark
- Aggregate metrics by sector for sector-level analysis

Output: data/gold/risk_metrics_lakehouse/

Usage:
    python gold/risk_metrics.py
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

# Risk-free rate = 5% per year (US Treasury rate)
RISK_FREE_RATE = 0.05
# Trading days per year
TRADING_DAYS = 252

# SPY ticker (S&P 500 ETF) - benchmark for Beta calculation
BENCHMARK_TICKER = 'SPY'


# =============================================================================
# LOAD DATA
# =============================================================================
def load_silver_data() -> pd.DataFrame:
    """Load data from Silver layer (Lakehouse or Parquet)"""
    from utils import is_lakehouse_table, lakehouse_to_pandas
    
    if is_lakehouse_table(SILVER_LAKEHOUSE_PATH):
        logger.info(f"Loading from Silver Lakehouse: {SILVER_LAKEHOUSE_PATH}")
        return lakehouse_to_pandas(SILVER_LAKEHOUSE_PATH)
    elif SILVER_PARQUET_PATH.exists():
        logger.info(f"Loading from Silver Parquet: {SILVER_PARQUET_PATH}")
        return pd.read_parquet(SILVER_PARQUET_PATH)
    else:
        raise FileNotFoundError("No Silver data found. Run silver/clean_delta.py first.")


def get_spy_returns(df: pd.DataFrame) -> Tuple[pd.Series, bool]:
    """Extract SPY returns as market benchmark"""
    spy_df = df[df['ticker'] == BENCHMARK_TICKER].copy()
    
    if len(spy_df) == 0:
        logger.warning(f"⚠️ {BENCHMARK_TICKER} not found. Beta/Alpha will be NaN.")
        return pd.Series(dtype=float), False
    
    spy_df = spy_df.sort_values('date').set_index('date')
    spy_returns = spy_df['daily_return'] / 100  # Convert to decimal
    
    logger.info(f"✓ Loaded {BENCHMARK_TICKER}: {len(spy_returns):,} days")
    return spy_returns, True


# =============================================================================
# VECTORIZED RISK CALCULATIONS (OPTIMIZED)
# =============================================================================
def calculate_ticker_metrics_vectorized(df: pd.DataFrame, spy_returns: pd.Series, 
                                         has_spy: bool) -> pd.DataFrame:
    """
    Calculate risk metrics using vectorized operations (FAST)
    """
    logger.info("Calculating per-ticker risk metrics (vectorized)...")
    
    # Convert daily_return to decimal
    df = df.copy()
    df['daily_return_decimal'] = df['daily_return'] / 100
    
    # Group by ticker
    grouped = df.groupby('ticker')
    
    # Basic aggregations (fast)
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
    
    # Calculate Sharpe Ratio
    logger.info("  Step 2/5: Sharpe Ratio...")
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    basic['sharpe_ratio'] = np.where(
        (basic['std_return'] > 0) & (basic['num_records'] >= 30),
        ((basic['mean_return'] - daily_rf) / basic['std_return']) * np.sqrt(TRADING_DAYS),
        np.nan
    )
    
    # Volatility (annualized)
    logger.info("  Step 3/5: Volatility...")
    basic['volatility'] = basic['std_return'] * np.sqrt(TRADING_DAYS)
    
    # VaR 95% - need to calculate per ticker
    logger.info("  Step 4/5: VaR 95%...")
    var_95 = grouped['daily_return_decimal'].quantile(0.05).reset_index()
    var_95.columns = ['ticker', 'var_95']
    basic = basic.merge(var_95, on='ticker', how='left')
    
    # Max Drawdown - calculate per ticker
    logger.info("  Step 5/5: Max Drawdown...")
    
    def calc_max_drawdown(group):
        prices = group.sort_values('date')['close']
        if len(prices) < 30:
            return np.nan
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    max_dd = grouped.apply(calc_max_drawdown).reset_index()
    max_dd.columns = ['ticker', 'max_drawdown']
    basic = basic.merge(max_dd, on='ticker', how='left')
    
    # Sortino Ratio
    logger.info("  Calculating Sortino...")
    
    def calc_sortino(group):
        returns = group['daily_return_decimal']
        if len(returns) < 30:
            return np.nan
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return np.nan
        daily_rf = RISK_FREE_RATE / TRADING_DAYS
        return ((returns.mean() - daily_rf) / downside.std()) * np.sqrt(TRADING_DAYS)
    
    sortino = grouped.apply(calc_sortino).reset_index()
    sortino.columns = ['ticker', 'sortino_ratio']
    basic = basic.merge(sortino, on='ticker', how='left')
    
    # Beta and Alpha (if SPY available)
    if has_spy:
        logger.info("  Calculating Beta & Alpha...")
        
        def calc_beta_alpha(group):
            returns = group.set_index('date')['daily_return_decimal']
            aligned = pd.concat([returns, spy_returns], axis=1).dropna()
            
            if len(aligned) < 60:
                return pd.Series({'beta': np.nan, 'alpha': np.nan})
            
            stock = aligned.iloc[:, 0]
            market = aligned.iloc[:, 1]
            
            cov = stock.cov(market)
            var = market.var()
            
            if var == 0:
                return pd.Series({'beta': np.nan, 'alpha': np.nan})
            
            beta = cov / var
            daily_rf = RISK_FREE_RATE / TRADING_DAYS
            expected = daily_rf + beta * (market.mean() - daily_rf)
            alpha = (stock.mean() - expected) * TRADING_DAYS
            
            return pd.Series({'beta': beta, 'alpha': alpha})
        
        beta_alpha = grouped.apply(calc_beta_alpha).reset_index()
        basic = basic.merge(beta_alpha, on='ticker', how='left')
    else:
        basic['beta'] = np.nan
        basic['alpha'] = np.nan
    
    # Convert to percentages
    basic['var_95_pct'] = basic['var_95'] * 100
    basic['max_drawdown_pct'] = basic['max_drawdown'] * 100
    basic['volatility_pct'] = basic['volatility'] * 100
    basic['alpha_pct'] = basic['alpha'] * 100
    
    # Filter out tickers with insufficient data
    basic = basic[basic['num_records'] >= 30]
    
    logger.info(f"✓ Calculated metrics for {len(basic):,} tickers")
    
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
    
    logger.info(f"✓ Aggregated metrics for {len(sector_agg)} sectors")
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
    logger.info(f"✓ Loaded {len(df):,} rows")
    logger.info(f"  Tickers: {df['ticker'].nunique():,}")
    
    # 2. Get SPY benchmark
    spy_returns, has_spy = get_spy_returns(df)
    
    # 3. Calculate metrics (vectorized - fast!)
    ticker_metrics = calculate_ticker_metrics_vectorized(df, spy_returns, has_spy)
    
    # 4. Sector aggregates
    sector_metrics = calculate_sector_risk_metrics(ticker_metrics)
    
    # 5. Save to Lakehouse
    logger.info("\nSaving to Lakehouse...")
    pandas_to_lakehouse(ticker_metrics, OUTPUT_TICKER_PATH, mode="overwrite")
    pandas_to_lakehouse(sector_metrics, OUTPUT_SECTOR_PATH, mode="overwrite")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("✓✓✓ RISK METRICS COMPLETED ✓✓✓")
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
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
