"""
Gold Layer: Strategy 1 - Low-Beta Quality Portfolio
Phân bổ danh mục dựa trên cổ phiếu có Beta thấp và chất lượng cao

Ý nghĩa kinh tế:
- Beta thấp = ít bị ảnh hưởng bởi biến động thị trường chung
- Low-Beta Anomaly: Nghịch lý được Fisher Black phát hiện năm 1972
  "Cổ phiếu rủi ro thấp thường cho lợi nhuận cao hơn dự đoán"
- Quality filter: Công ty lợi nhuận ổn định thường có kết quả tốt hơn

Data sử dụng:
- enriched_stocks.parquet: Giá, daily_return, sector
- benchmarks (SPY): Để tính Beta
- economic_lakehouse: VIX để điều chỉnh tỷ trọng

Output: data/gold/low_beta_quality_lakehouse/
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime
from typing import Dict, Tuple
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
SILVER_PARQUET = SILVER_DIR / 'enriched_stocks.parquet'
SILVER_LAKEHOUSE_PATH = SILVER_DIR / 'enriched_lakehouse'
ECONOMIC_PATH = SILVER_DIR / 'economic_lakehouse'
OUTPUT_PATH = GOLD_DIR / 'low_beta_quality_lakehouse'

# Memory optimization: Only load required columns
REQUIRED_COLUMNS = ['ticker', 'date', 'close', 'daily_return', 'sector']

# Thông số chiến lược
BETA_THRESHOLD = 1.0        # Chỉ chọn cổ phiếu có Beta < 1
MIN_HISTORY_DAYS = 252      # Cần ít nhất 1 năm data
TOP_N_STOCKS = 30           # Số cổ phiếu trong danh mục
RISK_FREE_RATE = 0.05       # 5% năm (US Treasury)
BATCH_SIZE = 100            # Process tickers in batches


# =============================================================================
# LOAD DATA
# =============================================================================
def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load tất cả data cần thiết từ Silver layer - MEMORY OPTIMIZED
    
    Only loads required columns to reduce memory by ~70%
    
    Returns:
        - df_prices: DataFrame chứa price data
        - df_economic: DataFrame chứa economic indicators
        - spy_returns: Series chứa daily returns của SPY (benchmark)
    """
    import duckdb
    from utils import is_lakehouse_table, lakehouse_to_pandas, get_metadata_path
    import json
    
    logger.info("Loading data from Silver layer (MEMORY OPTIMIZED)...")
    logger.info(f"  Loading only columns: {REQUIRED_COLUMNS}")
    
    # 1. Load price data - prefer Lakehouse for efficiency
    df_prices = None
    
    if is_lakehouse_table(SILVER_LAKEHOUSE_PATH):
        logger.info(f"  Loading from Silver Lakehouse: {SILVER_LAKEHOUSE_PATH}")
        meta_path = get_metadata_path(SILVER_LAKEHOUSE_PATH)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata['versions']:
            version_info = metadata['versions'][-1]
            data_file = SILVER_LAKEHOUSE_PATH / version_info['file']
            
            # Use DuckDB for memory-efficient loading
            cols_str = ", ".join(REQUIRED_COLUMNS)
            con = duckdb.connect()
            df_prices = con.execute(f"SELECT {cols_str} FROM read_parquet('{data_file}')").fetchdf()
            con.close()
    
    if df_prices is None and SILVER_PARQUET.exists():
        logger.info(f"  Loading from Silver Parquet: {SILVER_PARQUET}")
        df_prices = pd.read_parquet(SILVER_PARQUET, columns=REQUIRED_COLUMNS)
    
    if df_prices is None:
        raise FileNotFoundError("No Silver data found")
    
    logger.info(f"  [OK] Prices: {len(df_prices):,} rows, {df_prices['ticker'].nunique():,} tickers")
    logger.info(f"  Memory usage: {df_prices.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 2. Load economic data
    if is_lakehouse_table(ECONOMIC_PATH):
        df_economic = lakehouse_to_pandas(ECONOMIC_PATH)
        logger.info(f"  [OK] Economic: {len(df_economic):,} rows")
    else:
        logger.warning("  [WARN] Economic data not found, will use defaults")
        df_economic = pd.DataFrame()
    
    # 3. Extract SPY returns (benchmark)
    spy_df = df_prices[df_prices['ticker'] == 'SPY'].copy()
    if len(spy_df) > 0:
        spy_df = spy_df.sort_values('date').set_index('date')
        spy_returns = spy_df['daily_return'] / 100  # Convert to decimal
        logger.info(f"  [OK] SPY benchmark: {len(spy_returns):,} days")
    else:
        logger.warning("  [WARN] SPY not found, Beta calculation will be limited")
        spy_returns = pd.Series(dtype=float)
    
    return df_prices, df_economic, spy_returns


# =============================================================================
# CALCULATE BETA
# =============================================================================
def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Tính Beta của một cổ phiếu so với thị trường (SPY)
    
    Beta = Covariance(Stock, Market) / Variance(Market)
    
    Ý nghĩa kinh tế:
    - Beta = 1: Cổ phiếu di chuyển cùng mức với thị trường
    - Beta < 1: Cổ phiếu ít biến động hơn thị trường (defensive)
    - Beta > 1: Cổ phiếu biến động nhiều hơn thị trường (aggressive)
    
    Ví dụ:
    - Utilities thường có Beta = 0.3-0.5 (an toàn)
    - Technology thường có Beta = 1.2-1.5 (rủi ro cao)
    """
    # Align dates
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    
    if len(aligned) < 60:  # Cần ít nhất 60 ngày
        return np.nan
    
    stock = aligned.iloc[:, 0]
    market = aligned.iloc[:, 1]
    
    covariance = stock.cov(market)
    variance = market.var()
    
    if variance == 0:
        return np.nan
    
    return covariance / variance


def calculate_return_stability(returns: pd.Series) -> float:
    """
    Đo lường độ ổn định của lợi nhuận
    
    Công thức: Sharpe Ratio đơn giản = Mean / StdDev
    
    Ý nghĩa kinh tế:
    - Tỷ số cao = lợi nhuận ổn định, ít biến động
    - Công ty chất lượng cao thường có return stability cao
    """
    if len(returns) < 30 or returns.std() == 0:
        return 0.0
    
    # Sharpe-like ratio (không annualized để đơn giản)
    return returns.mean() / returns.std()


# =============================================================================
# BUILD LOW-BETA QUALITY PORTFOLIO
# =============================================================================
def build_low_beta_quality_portfolio(df_prices: pd.DataFrame, 
                                      spy_returns: pd.Series,
                                      df_economic: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng danh mục Low-Beta Quality
    
    Logic kinh tế:
    1. Tính Beta cho mỗi cổ phiếu
    2. Lọc những cổ phiếu có Beta < 1 (ít rủi ro hệ thống)
    3. Xếp hạng theo Return Stability (chất lượng)
    4. Chọn top N cổ phiếu
    5. Phân bổ tỷ trọng theo inverse volatility
    """
    logger.info("Building Low-Beta Quality Portfolio (MEMORY OPTIMIZED)...")
    
    # Bước 1: Tính metrics cho mỗi ticker với batch processing
    logger.info("  Step 1: Calculating Beta and Quality metrics...")
    
    ticker_metrics = []
    tickers = df_prices['ticker'].unique()
    total_tickers = len(tickers)
    total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info(f"  Processing {total_tickers:,} tickers in {total_batches} batches of {BATCH_SIZE}")
    
    # Group by ticker for faster lookup
    grouped = df_prices.groupby('ticker')
    
    for batch_idx in range(0, total_tickers, BATCH_SIZE):
        batch_tickers = tickers[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        if batch_num % 5 == 1 or batch_num == total_batches:
            logger.info(f"    Batch {batch_num}/{total_batches} ({len(batch_tickers)} tickers)...")
        
        for ticker in batch_tickers:
            try:
                ticker_df = grouped.get_group(ticker).sort_values('date')
                
                # Bỏ qua nếu không đủ data
                if len(ticker_df) < MIN_HISTORY_DAYS:
                    continue
                
                returns = ticker_df['daily_return'].values / 100  # Convert to decimal
                returns_series = ticker_df.set_index('date')['daily_return'] / 100
                
                # Tính Beta
                beta = calculate_beta(returns_series, spy_returns)
                if pd.isna(beta):
                    continue
                
                # Tính Return Stability (proxy cho Quality)
                stability = calculate_return_stability(pd.Series(returns))
                
                # Tính Volatility
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                # Lấy sector
                sector = ticker_df['sector'].iloc[0] if 'sector' in ticker_df.columns else 'Unknown'
                
                ticker_metrics.append({
                    'ticker': ticker,
                    'sector': sector,
                    'beta': beta,
                    'return_stability': stability,
                    'volatility': volatility,
                    'num_days': len(ticker_df),
                    'avg_return': np.mean(returns) * 252,  # Annualized
                })
            except Exception as e:
                continue
        
        # Free memory after each batch
        gc.collect()
    
    df_metrics = pd.DataFrame(ticker_metrics)
    logger.info(f"  [OK] Calculated metrics for {len(df_metrics):,} tickers")
    
    # Bước 2: Lọc Low-Beta
    logger.info(f"  Step 2: Filtering Low-Beta stocks (Beta < {BETA_THRESHOLD})...")
    df_low_beta = df_metrics[df_metrics['beta'] < BETA_THRESHOLD].copy()
    logger.info(f"  [OK] Found {len(df_low_beta):,} low-beta stocks")
    
    if len(df_low_beta) == 0:
        logger.warning("  [WARN] No low-beta stocks found!")
        return pd.DataFrame()
    
    # Bước 3: Xếp hạng theo Quality (Return Stability)
    logger.info("  Step 3: Ranking by Quality (Return Stability)...")
    df_low_beta = df_low_beta.sort_values('return_stability', ascending=False)
    
    # Bước 4: Chọn top N
    logger.info(f"  Step 4: Selecting top {TOP_N_STOCKS} stocks...")
    df_selected = df_low_beta.head(TOP_N_STOCKS).copy()
    
    # Bước 5: Tính tỷ trọng (Inverse Volatility)
    # Ý nghĩa kinh tế: Cổ phiếu ít biến động → tỷ trọng cao hơn
    logger.info("  Step 5: Calculating weights (Inverse Volatility)...")
    
    df_selected['inv_vol'] = 1 / df_selected['volatility']
    total_inv_vol = df_selected['inv_vol'].sum()
    df_selected['weight'] = df_selected['inv_vol'] / total_inv_vol
    
    # Điều chỉnh theo VIX (nếu có economic data)
    if len(df_economic) > 0 and 'vix' in df_economic.columns:
        latest_vix = df_economic.sort_values('date')['vix'].iloc[-1]
        
        # Khi VIX cao (thị trường sợ hãi): Tăng tỷ trọng Low-Beta stocks
        # Khi VIX thấp: Giữ nguyên
        if latest_vix > 25:
            vix_adjustment = 1.1  # Tăng 10% allocation
            logger.info(f"  [INFO] VIX={latest_vix:.1f} > 25: Increasing low-beta weights by 10%")
        elif latest_vix > 30:
            vix_adjustment = 1.2  # Tăng 20%
            logger.info(f"  [INFO] VIX={latest_vix:.1f} > 30: Increasing low-beta weights by 20%")
        else:
            vix_adjustment = 1.0
        
        # Apply adjustment (không thay đổi tổng = 100%)
        # Adjustment chỉ là signal, không thực sự thay đổi weights ở đây
        df_selected['vix_signal'] = vix_adjustment
    else:
        df_selected['vix_signal'] = 1.0
    
    # Add metadata
    df_selected['strategy'] = 'low_beta_quality'
    df_selected['created_at'] = datetime.now()
    
    # Select output columns
    output_cols = ['ticker', 'sector', 'beta', 'return_stability', 'volatility',
                   'avg_return', 'weight', 'vix_signal', 'strategy', 'created_at']
    df_output = df_selected[output_cols].reset_index(drop=True)
    
    logger.info(f"  [OK] Portfolio built with {len(df_output)} stocks")
    logger.info(f"  [OK] Total weight: {df_output['weight'].sum():.4f}")
    
    return df_output


# =============================================================================
# MAIN
# =============================================================================
def run_low_beta_quality() -> pd.DataFrame:
    """Run Low-Beta Quality strategy"""
    from utils import pandas_to_lakehouse
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("GOLD LAYER: LOW-BETA QUALITY PORTFOLIO")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Economic Rationale:")
    logger.info("  - Low-Beta stocks are less sensitive to market swings")
    logger.info("  - Quality filter ensures stable, profitable companies")
    logger.info("  - Academic backing: Low-Beta Anomaly (Black, 1972)")
    logger.info("")
    
    # Load data
    df_prices, df_economic, spy_returns = load_all_data()
    
    # Build portfolio
    df_portfolio = build_low_beta_quality_portfolio(df_prices, spy_returns, df_economic)
    
    # Free memory after processing
    del df_prices
    del spy_returns
    gc.collect()
    logger.info("[OK] Freed raw data from memory")
    
    if len(df_portfolio) == 0:
        logger.error("[ERROR] Failed to build portfolio")
        return pd.DataFrame()
    
    # Save to Lakehouse
    logger.info(f"\nSaving to: {OUTPUT_PATH}")
    pandas_to_lakehouse(df_portfolio, OUTPUT_PATH, mode="overwrite")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[OK] LOW-BETA QUALITY PORTFOLIO COMPLETED")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 70)
    
    # Print portfolio
    print("\n--- LOW-BETA QUALITY PORTFOLIO ---")
    print(df_portfolio[['ticker', 'sector', 'beta', 'return_stability', 'weight']].to_string(index=False))
    print(f"\nTotal stocks: {len(df_portfolio)}")
    print(f"Average Beta: {df_portfolio['beta'].mean():.3f}")
    print(f"Average Volatility: {df_portfolio['volatility'].mean()*100:.1f}%")
    
    return df_portfolio


def main() -> int:
    try:
        run_low_beta_quality()
        logger.info("\n[OK] Done! Next: python gold/sector_rotation.py")
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
