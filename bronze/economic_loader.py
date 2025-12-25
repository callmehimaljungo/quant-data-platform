"""
Bronze Layer: Economic Indicators Loader
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Load economic indicators from FRED (Federal Reserve Economic Data)
- Save as CSV (original format) and Parquet (Lakehouse)
- Provide sample data fallback if FRED API unavailable

Data Format:
- INPUT: FRED API (CSV format) or sample data
- OUTPUT: 
  - Raw CSV backup
  - Parquet in Lakehouse format

Economic Indicators:
- GDP: Gross Domestic Product
- CPI: Consumer Price Index
- UNRATE: Unemployment Rate
- DFF: Federal Funds Rate
- DGS10: 10-Year Treasury Rate
- VIXCLS: VIX Volatility Index

Business Context:
- Macro factors affect all stocks
- Enable regime detection (bull/bear/recovery)
- Risk-off vs risk-on environment classification
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import pandas_datareader for FRED access
try:
    import pandas_datareader as pdr
    from pandas_datareader import data as web
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False
    logger.warning("[WARN] pandas_datareader not installed. Install with: pip install pandas-datareader")


# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUT_DIR = BRONZE_DIR / 'economic_lakehouse'
RAW_CSV_DIR = BRONZE_DIR / 'economic_raw'

# Publication Lag (days) - Time between reference period end and public availability
# CRITICAL: These lags prevent look-ahead bias in backtests
# Source: FRED release schedules
PUBLICATION_LAGS = {
    'GDP': 30,        # GDP released ~30 days after quarter ends
    'CPIAUCSL': 15,   # CPI released ~15 days after month ends
    'UNRATE': 7,      # Unemployment released first Friday of month
    'DFF': 1,         # Fed rate effective next day
    'DGS10': 1,       # Treasury yields available next day
    'VIXCLS': 0,      # VIX is real-time
    'DTWEXBGS': 1,    # Dollar index next day
}

# FRED Series definitions
FRED_SERIES = {
    'GDP': {
        'id': 'GDP',
        'name': 'Gross Domestic Product',
        'unit': 'Billions of Dollars',
        'frequency': 'quarterly',
        'description': 'Total value of goods and services produced'
    },
    'CPIAUCSL': {
        'id': 'CPIAUCSL',
        'name': 'Consumer Price Index',
        'unit': 'Index 1982-84=100',
        'frequency': 'monthly',
        'description': 'Measure of inflation/deflation'
    },
    'UNRATE': {
        'id': 'UNRATE',
        'name': 'Unemployment Rate',
        'unit': 'Percent',
        'frequency': 'monthly',
        'description': 'Percentage of labor force unemployed'
    },
    'DFF': {
        'id': 'DFF',
        'name': 'Federal Funds Rate',
        'unit': 'Percent',
        'frequency': 'daily',
        'description': 'Interest rate at which banks lend to each other overnight'
    },
    'DGS10': {
        'id': 'DGS10',
        'name': '10-Year Treasury Rate',
        'unit': 'Percent',
        'frequency': 'daily',
        'description': 'Yield on 10-year US Treasury bonds'
    },
    'VIXCLS': {
        'id': 'VIXCLS',
        'name': 'CBOE Volatility Index',
        'unit': 'Index',
        'frequency': 'daily',
        'description': 'Market expectation of near-term volatility (Fear Index)'
    },
    'DXY': {
        'id': 'DTWEXBGS',
        'name': 'US Dollar Index',
        'unit': 'Index',
        'frequency': 'daily',
        'description': 'Trade-weighted US dollar index'
    }
}


# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================

def generate_sample_economic_data(
    start_date: str = '2020-01-01',
    end_date: str = None
) -> pd.DataFrame:
    """
    Generate realistic sample economic data for testing
    
    Uses actual historical patterns to create realistic data
    
    Args:
        start_date: Start date for data
        end_date: End date (default: today)
        
    Returns:
        DataFrame with economic indicators
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Seed for reproducibility
    np.random.seed(42)
    
    data = []
    
    for date in date_range:
        # Skip weekends for some indicators
        is_trading_day = date.weekday() < 5
        
        # Time-based factors (simulate economic cycles)
        days_since_start = (date - date_range[0]).days
        cycle_phase = np.sin(days_since_start / 365 * 2 * np.pi)  # Annual cycle
        
        # GDP (quarterly) - Only first day of quarter
        if date.month in [1, 4, 7, 10] and date.day == 1:
            data.append({
                'date': date,
                'indicator_id': 'GDP',
                'indicator_name': FRED_SERIES['GDP']['name'],
                'value': round(22000 + cycle_phase * 1000 + np.random.normal(0, 100), 1),
                'unit': FRED_SERIES['GDP']['unit'],
                'frequency': 'quarterly'
            })
        
        # CPI (monthly) - First day of month
        if date.day == 1:
            data.append({
                'date': date,
                'indicator_id': 'CPIAUCSL',
                'indicator_name': FRED_SERIES['CPIAUCSL']['name'],
                'value': round(280 + days_since_start * 0.015 + np.random.normal(0, 0.5), 2),
                'unit': FRED_SERIES['CPIAUCSL']['unit'],
                'frequency': 'monthly'
            })
            
            # Unemployment (monthly)
            base_unemp = 4.0 - cycle_phase * 1.5  # Counter-cyclical
            data.append({
                'date': date,
                'indicator_id': 'UNRATE',
                'indicator_name': FRED_SERIES['UNRATE']['name'],
                'value': round(max(3.0, min(8.0, base_unemp + np.random.normal(0, 0.2))), 1),
                'unit': FRED_SERIES['UNRATE']['unit'],
                'frequency': 'monthly'
            })
        
        # Daily indicators (trading days only)
        if is_trading_day:
            # Fed Funds Rate
            ffr_trend = 0.25 + days_since_start * 0.005  # Gradual increase
            ffr_trend = min(5.5, max(0.25, ffr_trend))  # Bound to realistic range
            data.append({
                'date': date,
                'indicator_id': 'DFF',
                'indicator_name': FRED_SERIES['DFF']['name'],
                'value': round(ffr_trend + np.random.normal(0, 0.02), 2),
                'unit': FRED_SERIES['DFF']['unit'],
                'frequency': 'daily'
            })
            
            # 10-Year Treasury
            t10y = ffr_trend + 0.5 + np.random.normal(0, 0.05)
            data.append({
                'date': date,
                'indicator_id': 'DGS10',
                'indicator_name': FRED_SERIES['DGS10']['name'],
                'value': round(t10y, 2),
                'unit': FRED_SERIES['DGS10']['unit'],
                'frequency': 'daily'
            })
            
            # VIX (inversely related to stock performance)
            vix_base = 20 - cycle_phase * 8  # Higher in down markets
            vix = max(10, min(40, vix_base + np.random.exponential(2)))
            data.append({
                'date': date,
                'indicator_id': 'VIXCLS',
                'indicator_name': FRED_SERIES['VIXCLS']['name'],
                'value': round(vix, 2),
                'unit': FRED_SERIES['VIXCLS']['unit'],
                'frequency': 'daily'
            })
            
            # Dollar Index
            dxy = 100 + cycle_phase * 5 + np.random.normal(0, 0.5)
            data.append({
                'date': date,
                'indicator_id': 'DTWEXBGS',
                'indicator_name': FRED_SERIES['DXY']['name'],
                'value': round(dxy, 2),
                'unit': FRED_SERIES['DXY']['unit'],
                'frequency': 'daily'
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df):,} sample economic data points")
    
    return df


# =============================================================================
# FRED DATA FETCHER
# =============================================================================

def fetch_from_fred(
    series_ids: List[str] = None,
    start_date: str = '2020-01-01',
    end_date: str = None
) -> Optional[pd.DataFrame]:
    """
    Fetch economic data from FRED
    
    Args:
        series_ids: List of FRED series IDs to fetch
        start_date: Start date
        end_date: End date (default: today)
        
    Returns:
        DataFrame with economic data or None if failed
    """
    if not PDR_AVAILABLE:
        logger.warning("pandas_datareader not available")
        return None
    
    if series_ids is None:
        series_ids = list(FRED_SERIES.keys())
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    
    for series_id in series_ids:
        try:
            logger.info(f"Fetching {series_id} from FRED...")
            
            # Get actual FRED series ID (some have different IDs)
            fred_id = FRED_SERIES.get(series_id, {}).get('id', series_id)
            
            df = web.DataReader(fred_id, 'fred', start_date, end_date)
            
            # Convert to long format
            for date, row in df.iterrows():
                all_data.append({
                    'date': date,
                    'indicator_id': series_id,
                    'indicator_name': FRED_SERIES.get(series_id, {}).get('name', series_id),
                    'value': row[fred_id],
                    'unit': FRED_SERIES.get(series_id, {}).get('unit', ''),
                    'frequency': FRED_SERIES.get(series_id, {}).get('frequency', 'unknown')
                })
            
            logger.info(f"[OK] Fetched {len(df)} data points for {series_id}")
            
        except Exception as e:
            logger.warning(f"Failed to fetch {series_id}: {e}")
    
    if all_data:
        return pd.DataFrame(all_data)
    return None


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_economic_data(
    source: str = 'auto',
    start_date: str = '2020-01-01',
    end_date: str = None
) -> pd.DataFrame:
    """
    Main function to load economic indicators
    
    Args:
        source: Data source ('fred', 'sample', 'auto')
        start_date: Start date for data
        end_date: End date (default: today)
        
    Returns:
        DataFrame with economic indicators
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BRONZE LAYER: ECONOMIC INDICATORS INGESTION")
    logger.info("=" * 70)
    
    df = None
    
    # Try FRED first if available
    if source in ['auto', 'fred'] and PDR_AVAILABLE:
        logger.info("Attempting to fetch from FRED...")
        df = fetch_from_fred(start_date=start_date, end_date=end_date)
    
    # Fall back to sample data
    if df is None or len(df) == 0:
        logger.info("Using sample economic data")
        df = generate_sample_economic_data(start_date=start_date, end_date=end_date)
    
    # Add metadata
    df['source_file'] = 'fred_api' if source == 'fred' else 'sample_data'
    df['fetched_at'] = datetime.now()
    
    # Log summary
    logger.info(f"\n[OK] Loaded {len(df):,} data points")
    logger.info(f"[OK] Indicators: {df['indicator_id'].nunique()}")
    logger.info(f"[OK] Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Indicator distribution
    logger.info("\nIndicator Distribution:")
    for indicator, count in df['indicator_id'].value_counts().items():
        logger.info(f"  {indicator}: {count:,}")
    
    # Save raw CSV
    RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save each indicator as separate CSV
    for indicator_id in df['indicator_id'].unique():
        indicator_df = df[df['indicator_id'] == indicator_id]
        csv_file = RAW_CSV_DIR / f'{indicator_id}_{datetime.now().strftime("%Y%m%d")}.csv'
        indicator_df.to_csv(csv_file, index=False)
        logger.info(f"[OK] Saved CSV: {csv_file}")
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info(f"ECONOMIC DATA INGESTION COMPLETED")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("=" * 70)
    
    return df


def save_to_lakehouse(df: pd.DataFrame) -> str:
    """Save economic data to Lakehouse format"""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"[OK] Saved to {path}")
    return path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(start_date: str = '2020-01-01', test: bool = False):
    """CLI entry point."""
    logger.info("")
    logger.info(" BRONZE LAYER: ECONOMIC INDICATORS LOADER")
    logger.info("")
    
    try:
        if test:
            start_date = '2024-01-01'
            logger.info("Running in TEST mode (2024 data only)")
        
        # Load economic data
        df = load_economic_data(start_date=start_date)
        
        # Save to Lakehouse
        save_to_lakehouse(df)
        
        logger.info("")
        logger.info("✅ Economic data loading completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info(f"✅ CSV files: {RAW_CSV_DIR}")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"[ERR] Economic data loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("")
        return 1


if __name__ == "__main__":
    import sys
    
    test_mode = '--test' in sys.argv
    start_date = '2020-01-01'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--start='):
            start_date = arg.split('=')[1]
    
    exit(main(start_date=start_date, test=test_mode))
