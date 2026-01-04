"""
Gold Layer: Strategy 2 - Sector Rotation theo Chu kỳ Kinh tế
Phân bổ danh mục dựa trên giai đoạn kinh tế hiện tại

Ý nghĩa kinh tế:
- Mỗi ngành hoạt động tốt nhất trong giai đoạn kinh tế khác nhau
- Ví dụ: Suy thoái → Ngành Y tế, Tiện ích thắng (người ta vẫn cần)
- Nghiên cứu của NBER xác nhận pattern này lặp lại qua nhiều thập kỷ

Giai đoạn kinh tế và ngành phù hợp:
- RECOVERY (Phục hồi): Technology, Financials, Consumer Discretionary
- EXPANSION (Tăng trưởng): Technology, Industrials, Consumer Discretionary
- PEAK (Đỉnh cao): Energy, Materials, Industrials
- RECESSION (Suy thoái): Healthcare, Utilities, Consumer Staples

Data sử dụng:
- economic_lakehouse: VIX, fed_funds_rate, unemployment, market_regime
- enriched_stocks.parquet: Sector performance
- metadata_lakehouse: Sector classification

Output: data/gold/sector_rotation_lakehouse/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import gc

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
SILVER_PARQUET = SILVER_DIR / 'enriched_stocks.parquet'
SILVER_LAKEHOUSE_PATH = SILVER_DIR / 'enriched_lakehouse'
ECONOMIC_PATH = SILVER_DIR / 'economic_lakehouse'
OUTPUT_PATH = GOLD_DIR / 'sector_rotation_lakehouse'

# Memory optimization: Only load required columns
REQUIRED_COLUMNS = ['ticker', 'date', 'close', 'daily_return', 'sector']

# Sector allocation theo Economic Regime
# Dựa trên nghiên cứu của Fidelity và NBER về Business Cycle Investing
# Note: Sector names match yfinance format
REGIME_SECTOR_ALLOCATION = {
    'recovery': {
        # Kinh tế vừa qua suy thoái, bắt đầu hồi phục
        # → Ngành nhạy cảm với lãi suất và tiêu dùng
        'Technology': 0.25,
        'Financial Services': 0.20,  # yfinance: Financial Services (not Financials)
        'Consumer Cyclical': 0.20,   # yfinance: Consumer Cyclical (not Consumer Discretionary)
        'Industrials': 0.15,
        'Communication Services': 0.10,
        'Others': 0.10,
    },
    'expansion': {
        # Kinh tế đang tăng trưởng mạnh
        # → Ngành tăng trưởng và chu kỳ
        'Technology': 0.30,
        'Industrials': 0.20,
        'Consumer Cyclical': 0.15,   # yfinance format
        'Financial Services': 0.15,  # yfinance format
        'Basic Materials': 0.10,     # yfinance: Basic Materials (not Materials)
        'Others': 0.10,
    },
    'peak': {
        # Kinh tế đạt đỉnh, lạm phát tăng
        # → Ngành hàng hóa và vật liệu
        'Energy': 0.25,
        'Basic Materials': 0.20,     # yfinance format
        'Industrials': 0.15,
        'Financial Services': 0.15,  # yfinance format
        'Real Estate': 0.10,
        'Others': 0.15,
    },
    'recession': {
        # Kinh tế suy thoái
        # → Ngành phòng thủ, cần thiết
        'Healthcare': 0.25,
        'Utilities': 0.25,
        'Consumer Defensive': 0.25,  # yfinance: Consumer Defensive (not Consumer Staples)
        'Communication Services': 0.10,
        'Real Estate': 0.10,
        'Others': 0.05,
    }
}

# VIX thresholds để xác định regime
VIX_THRESHOLDS = {
    'low': 15,      # VIX < 15: Expansion/Peak
    'normal': 20,   # VIX 15-20: Normal
    'high': 25,     # VIX 20-25: Uncertainty
    'crisis': 30,   # VIX > 30: Recession/Crisis
}


# =============================================================================
# DETERMINE ECONOMIC REGIME
# =============================================================================
def determine_economic_regime(df_economic: pd.DataFrame) -> str:
    """
    Xác định giai đoạn kinh tế hiện tại dựa trên VIX và indicators
    
    Logic kinh tế đơn giản:
    - VIX > 30: Thị trường hoảng loạn → RECESSION signal
    - VIX < 15: Thị trường tự tin → EXPANSION/PEAK
    - Kết hợp với lãi suất và market_regime (nếu có)
    """
    if len(df_economic) == 0:
        logger.warning("No economic data, defaulting to 'expansion'")
        return 'expansion'
    
    # Lấy data gần nhất
    latest = df_economic.sort_values('date').iloc[-1]
    
    # VIX-based classification
    vix = latest.get('vix', 20)
    
    if vix > VIX_THRESHOLDS['crisis']:
        regime = 'recession'
        logger.info(f"  VIX = {vix:.1f} > 30 → RECESSION (Crisis mode)")
    elif vix > VIX_THRESHOLDS['high']:
        regime = 'recession'
        logger.info(f"  VIX = {vix:.1f} > 25 → RECESSION (High uncertainty)")
    elif vix < VIX_THRESHOLDS['low']:
        # Khi VIX thấp, cần xem thêm indicator khác
        # Nếu lãi suất đang tăng → Peak, ngược lại → Expansion
        fed_rate = latest.get('fed_funds_rate', 5)
        if fed_rate > 5:
            regime = 'peak'
            logger.info(f"  VIX = {vix:.1f} < 15, Fed Rate = {fed_rate:.1f}% > 5% → PEAK")
        else:
            regime = 'expansion'
            logger.info(f"  VIX = {vix:.1f} < 15, Fed Rate = {fed_rate:.1f}% → EXPANSION")
    else:
        # VIX normal (15-25)
        # Dùng market_regime nếu có
        if 'market_regime' in latest and pd.notna(latest['market_regime']):
            stored_regime = str(latest['market_regime']).lower()
            if stored_regime in REGIME_SECTOR_ALLOCATION:
                regime = stored_regime
            else:
                regime = 'expansion'
        else:
            regime = 'expansion'
        logger.info(f"  VIX = {vix:.1f} (normal range) → {regime.upper()}")
    
    return regime


# =============================================================================
# CALCULATE SECTOR PERFORMANCE
# =============================================================================
def calculate_sector_performance(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Tính hiệu suất của mỗi sector
    
    Metrics:
    - Average return
    - Volatility
    - Number of stocks
    """
    logger.info("Calculating sector performance...")
    
    # Lọc data gần đây (1 năm)
    latest_date = df_prices['date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    df_recent = df_prices[df_prices['date'] >= one_year_ago]
    
    sector_stats = []
    
    for sector in df_recent['sector'].unique():
        sector_df = df_recent[df_recent['sector'] == sector]
        
        returns = sector_df['daily_return'] / 100
        
        stats = {
            'sector': sector,
            'num_stocks': sector_df['ticker'].nunique(),
            'avg_return': returns.mean() * 252,  # Annualized
            'volatility': returns.std() * np.sqrt(252),
            'total_records': len(sector_df),
        }
        sector_stats.append(stats)
    
    df_sectors = pd.DataFrame(sector_stats)
    df_sectors = df_sectors.sort_values('avg_return', ascending=False)
    
    logger.info(f"  [OK] Calculated performance for {len(df_sectors)} sectors")
    
    return df_sectors


# =============================================================================
# BUILD SECTOR ROTATION PORTFOLIO
# =============================================================================
def build_sector_rotation_portfolio(df_prices: pd.DataFrame,
                                     df_economic: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng danh mục Sector Rotation
    
    Logic kinh tế:
    1. Xác định giai đoạn kinh tế hiện tại (từ VIX + indicators)
    2. Lấy sector allocation tương ứng với giai đoạn đó
    3. Chọn top stocks trong mỗi sector
    4. Tính weight cuối cùng
    """
    logger.info("Building Sector Rotation Portfolio...")
    
    # Bước 1: Xác định regime
    logger.info("  Step 1: Determining economic regime...")
    current_regime = determine_economic_regime(df_economic)
    
    # Bước 2: Lấy sector allocation
    logger.info(f"  Step 2: Getting sector allocation for '{current_regime}'...")
    target_allocation = REGIME_SECTOR_ALLOCATION[current_regime]
    
    # Bước 3: Tính sector performance
    logger.info("  Step 3: Calculating sector performance...")
    df_sector_perf = calculate_sector_performance(df_prices)
    
    # Bước 4: Map sectors và chọn stocks
    logger.info("  Step 4: Selecting stocks per sector...")
    
    portfolio_rows = []
    
    for sector, target_weight in target_allocation.items():
        if sector == 'Others':
            continue
        
        # Tìm stocks trong sector này
        sector_stocks = df_prices[df_prices['sector'] == sector]['ticker'].unique()
        
        if len(sector_stocks) == 0:
            # Fallback: Tìm sector tương tự
            logger.warning(f"    [WARN] No stocks in '{sector}', skipping")
            continue
        
        # Chọn top 5 stocks trong sector (dựa trên data count)
        top_n = min(5, len(sector_stocks))
        stock_counts = df_prices[df_prices['sector'] == sector]['ticker'].value_counts()
        selected_stocks = stock_counts.head(top_n).index.tolist()
        
        # Phân bổ đều trong sector
        stock_weight = target_weight / len(selected_stocks)
        
        for ticker in selected_stocks:
            portfolio_rows.append({
                'ticker': ticker,
                'sector': sector,
                'regime': current_regime,
                'sector_target_weight': target_weight,
                'weight': stock_weight,
            })
    
    df_portfolio = pd.DataFrame(portfolio_rows)
    
    if len(df_portfolio) == 0:
        logger.error("[ERROR] No stocks selected for portfolio")
        return pd.DataFrame()
    
    # Normalize weights
    total_weight = df_portfolio['weight'].sum()
    df_portfolio['weight'] = df_portfolio['weight'] / total_weight
    
    # Add metadata
    df_portfolio['strategy'] = 'sector_rotation'
    df_portfolio['created_at'] = datetime.now()
    
    logger.info(f"  [OK] Portfolio built with {len(df_portfolio)} stocks across {df_portfolio['sector'].nunique()} sectors")
    
    return df_portfolio


# =============================================================================
# MAIN
# =============================================================================
def run_sector_rotation() -> pd.DataFrame:
    """Run Sector Rotation strategy"""
    from utils import is_lakehouse_table, lakehouse_to_pandas, pandas_to_lakehouse
    from gold.utils import add_sector_metadata
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("GOLD LAYER: SECTOR ROTATION PORTFOLIO")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Economic Rationale:")
    logger.info("  - Different sectors outperform in different economic phases")
    logger.info("  - Recovery: Tech, Financials (interest rate sensitive)")
    logger.info("  - Expansion: Tech, Industrials (growth-oriented)")
    logger.info("  - Peak: Energy, Materials (inflation hedge)")
    logger.info("  - Recession: Healthcare, Utilities (defensive)")
    logger.info("  - Academic backing: NBER Business Cycle Research")
    logger.info("")
    
    # Load data
    logger.info("Loading data from Silver layer...")
    
    # Load data - MEMORY OPTIMIZED
    # Prefer parquet (frequently updated) over lakehouse
    logger.info("Loading data from Silver layer (MEMORY OPTIMIZED)...")
    logger.info(f"  Loading only columns: {REQUIRED_COLUMNS}")
    
    import duckdb
    
    df_prices = None
    
    # Option 1: Load from parquet (has latest sector data)
    if SILVER_PARQUET.exists():
        logger.info(f"  Loading from Silver Parquet: {SILVER_PARQUET}")
        cols_str = ", ".join(REQUIRED_COLUMNS)
        con = duckdb.connect()
        df_prices = con.execute(f"SELECT {cols_str} FROM read_parquet('{SILVER_PARQUET}')").fetchdf()
        con.close()
    
    # Option 2: Fallback to lakehouse
    if df_prices is None and is_lakehouse_table(SILVER_LAKEHOUSE_PATH):
        from utils import get_metadata_path
        import json
        
        logger.info(f"  Loading from Silver Lakehouse: {SILVER_LAKEHOUSE_PATH}")
        meta_path = get_metadata_path(SILVER_LAKEHOUSE_PATH)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata['versions']:
            version_info = metadata['versions'][-1]
            data_file = SILVER_LAKEHOUSE_PATH / version_info['file']
            
            cols_str = ", ".join(REQUIRED_COLUMNS)
            con = duckdb.connect()
            df_prices = con.execute(f"SELECT {cols_str} FROM read_parquet('{data_file}')").fetchdf()
            con.close()
    
    if df_prices is None:
        raise FileNotFoundError(f"Price data not found")
    
    logger.info(f"  [OK] Prices: {len(df_prices):,} rows")
    logger.info(f"  Memory usage: {df_prices.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Fix sectors in prices data before building portfolio
    logger.info("Updating sector metadata in price data...")
    df_prices = add_sector_metadata(df_prices, ticker_col='ticker')
    
    if is_lakehouse_table(ECONOMIC_PATH):
        df_economic = lakehouse_to_pandas(ECONOMIC_PATH)
        logger.info(f"  [OK] Economic: {len(df_economic):,} rows")
    else:
        df_economic = pd.DataFrame()
        logger.warning("  [WARN] Economic data not found")
    
    # Build portfolio
    df_portfolio = build_sector_rotation_portfolio(df_prices, df_economic)
    
    if len(df_portfolio) == 0:
        logger.error("[ERROR] Failed to build portfolio")
        return pd.DataFrame()
    
    # Update metadata in final portfolio
    df_portfolio = add_sector_metadata(df_portfolio, ticker_col='ticker')
    
    # Save to Lakehouse
    logger.info(f"\nSaving to: {OUTPUT_PATH}")
    pandas_to_lakehouse(df_portfolio, OUTPUT_PATH, mode="overwrite")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[OK] SECTOR ROTATION PORTFOLIO COMPLETED")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 70)
    
    # Print portfolio
    print("\n--- SECTOR ROTATION PORTFOLIO ---")
    print(f"Current Regime: {df_portfolio['regime'].iloc[0].upper()}")
    print("")
    
    # Group by sector for display
    sector_summary = df_portfolio.groupby('sector').agg({
        'ticker': 'count',
        'weight': 'sum'
    }).reset_index()
    sector_summary.columns = ['sector', 'num_stocks', 'total_weight']
    sector_summary = sector_summary.sort_values('total_weight', ascending=False)
    
    print(sector_summary.to_string(index=False))
    print(f"\nTotal stocks: {len(df_portfolio)}")
    print(f"Total weight: {df_portfolio['weight'].sum():.4f}")
    
    return df_portfolio


def main() -> int:
    try:
        run_sector_rotation()
        logger.info("\n[OK] Done! Next: python gold/sentiment_allocation.py")
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
