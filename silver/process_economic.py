"""
Silver Layer: Process Economic Indicators
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Process raw CSV/Parquet economic data from Bronze layer
- Parse and normalize dates across different frequencies
- Interpolate missing values (forward fill)
- Pivot to wide format (one column per indicator)
- Create regime classification based on economic conditions

Processing Steps:
1. Load raw CSV/Parquet from Bronze
2. Parse dates (handle different formats)
3. Aggregate to daily frequency
4. Pivot to wide format
5. Forward fill missing values
6. Calculate derived indicators
7. Classify market regime
8. Save to Silver Lakehouse
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, SILVER_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

INPUT_DIR = BRONZE_DIR / 'economic_lakehouse'
RAW_CSV_DIR = BRONZE_DIR / 'economic_raw'
OUTPUT_DIR = SILVER_DIR / 'economic_lakehouse'

# Indicator column mapping
INDICATOR_COLUMNS = {
    'GDP': 'gdp',
    'CPIAUCSL': 'cpi',
    'UNRATE': 'unemployment_rate',
    'DFF': 'fed_funds_rate',
    'DGS10': 'treasury_10y',
    'VIXCLS': 'vix',
    'DTWEXBGS': 'dollar_index',
}


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def load_bronze_economic() -> pd.DataFrame:
    """Load raw economic data from Bronze layer"""
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    dfs = []
    
    # Try Lakehouse first
    if is_lakehouse_table(INPUT_DIR):
        logger.info(f"Loading from Lakehouse: {INPUT_DIR}")
        df = lakehouse_to_pandas(INPUT_DIR)
        dfs.append(df)
    
    # Also try raw CSV files
    if RAW_CSV_DIR.exists():
        csv_files = list(RAW_CSV_DIR.glob('*.csv'))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                logger.info(f"Loaded CSV: {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
    
    if not dfs:
        raise FileNotFoundError("No Bronze economic data found")
    
    # Combine all data
    combined = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    combined = combined.drop_duplicates(subset=['date', 'indicator_id'], keep='last')
    
    logger.info(f"✓ Loaded {len(combined):,} economic data points from Bronze")
    return combined


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and normalize date column"""
    logger.info("Parsing dates...")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removed {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['date'])
    
    return df


def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from long format to wide format
    
    Long: date, indicator_id, value
    Wide: date, gdp, cpi, unemployment_rate, ...
    """
    logger.info("Pivoting to wide format...")
    
    # Ensure we have required columns
    if 'indicator_id' not in df.columns or 'value' not in df.columns:
        raise ValueError("DataFrame must have 'indicator_id' and 'value' columns")
    
    # Map indicator IDs to column names
    df['column_name'] = df['indicator_id'].map(INDICATOR_COLUMNS)
    df = df.dropna(subset=['column_name'])
    
    # Pivot
    wide_df = df.pivot_table(
        index='date',
        columns='column_name',
        values='value',
        aggfunc='last'  # Take last value if duplicates
    ).reset_index()
    
    # Sort by date
    wide_df = wide_df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"✓ Pivoted to {len(wide_df):,} rows × {len(wide_df.columns)} columns")
    
    return wide_df


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample all indicators to daily frequency
    Forward fill quarterly/monthly data to daily
    """
    logger.info("Resampling to daily frequency...")
    
    # Set date as index
    df = df.set_index('date')
    
    # Create daily date range
    date_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='D'
    )
    
    # Reindex to daily
    df = df.reindex(date_range)
    df.index.name = 'date'
    
    logger.info(f"✓ Expanded to {len(df):,} daily rows")
    
    return df.reset_index()


def forward_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fill missing values
    
    Appropriate for economic data where values persist until updated
    """
    logger.info("Forward filling missing values...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        missing_before = df[col].isna().sum()
        df[col] = df[col].ffill()
        missing_after = df[col].isna().sum()
        
        if missing_before > missing_after:
            logger.info(f"  {col}: filled {missing_before - missing_after:,} values")
    
    return df


def calculate_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived economic indicators
    """
    logger.info("Calculating derived indicators...")
    
    # Yield curve slope (10Y - Fed Funds)
    if 'treasury_10y' in df.columns and 'fed_funds_rate' in df.columns:
        df['yield_curve_slope'] = df['treasury_10y'] - df['fed_funds_rate']
        logger.info("  ✓ yield_curve_slope = 10Y Treasury - Fed Funds")
    
    # Real rate (10Y - CPI)
    if 'treasury_10y' in df.columns and 'cpi' in df.columns:
        # CPI needs to be converted to YoY % change first
        df['cpi_yoy'] = df['cpi'].pct_change(252) * 100  # Approximate annual
        df['real_rate'] = df['treasury_10y'] - df['cpi_yoy']
        logger.info("  ✓ real_rate = 10Y Treasury - CPI YoY")
    
    # VIX regime
    if 'vix' in df.columns:
        df['vix_regime'] = pd.cut(
            df['vix'],
            bins=[0, 15, 20, 30, 100],
            labels=['low_vol', 'normal', 'elevated', 'high_vol']
        )
        logger.info("  ✓ vix_regime: low_vol/normal/elevated/high_vol")
    
    return df


def classify_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market regime based on economic conditions
    
    Regimes:
    - expansion: Low unemployment, positive GDP growth, low VIX
    - contraction: Rising unemployment, negative GDP growth
    - recovery: Improving conditions from contraction
    - uncertainty: High VIX, mixed signals
    """
    logger.info("Classifying market regimes...")
    
    conditions = []
    
    for _, row in df.iterrows():
        regime = 'unknown'
        
        vix = row.get('vix', 20)
        unemployment = row.get('unemployment_rate', 5)
        yield_slope = row.get('yield_curve_slope', 1)
        
        if pd.isna(vix):
            vix = 20
        if pd.isna(unemployment):
            unemployment = 5
        if pd.isna(yield_slope):
            yield_slope = 1
        
        # Simple regime classification
        if vix > 30:
            regime = 'crisis'
        elif vix > 25:
            regime = 'uncertainty'
        elif yield_slope < 0:
            regime = 'inversion'  # Yield curve inverted - recession signal
        elif unemployment > 6:
            regime = 'contraction'
        elif unemployment < 4 and vix < 20:
            regime = 'expansion'
        else:
            regime = 'normal'
        
        conditions.append(regime)
    
    df['market_regime'] = conditions
    
    # Log regime distribution
    regime_counts = df['market_regime'].value_counts()
    logger.info("Market Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {regime}: {count:,} days ({pct:.1f}%)")
    
    return df


def add_processing_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add processing metadata"""
    df['processed_at'] = datetime.now()
    df['data_version'] = 'silver_v1'
    return df


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_economic() -> pd.DataFrame:
    """
    Main function to process economic data from Bronze to Silver
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("SILVER LAYER: ECONOMIC INDICATORS PROCESSING")
    logger.info("=" * 70)
    
    # Step 1: Load from Bronze
    df = load_bronze_economic()
    
    # Step 2: Parse dates
    df = parse_dates(df)
    
    # Step 3: Pivot to wide format
    df = pivot_to_wide_format(df)
    
    # Step 4: Resample to daily
    df = resample_to_daily(df)
    
    # Step 5: Forward fill missing values
    df = forward_fill_missing(df)
    
    # Step 6: Calculate derived indicators
    df = calculate_derived_indicators(df)
    
    # Step 7: Classify market regime
    df = classify_market_regime(df)
    
    # Step 8: Add metadata
    df = add_processing_metadata(df)
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info("ECONOMIC PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Output rows: {len(df):,}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Duration: {duration:.2f} seconds")
    
    return df


def save_to_lakehouse(df: pd.DataFrame) -> str:
    """Save processed economic data to Silver Lakehouse"""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"✓ Saved to {path}")
    return path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    logger.info("")
    logger.info("🚀 SILVER LAYER: ECONOMIC PROCESSOR")
    logger.info("")
    
    try:
        # Process economic data
        df = process_economic()
        
        # Save to Lakehouse
        save_to_lakehouse(df)
        
        logger.info("")
        logger.info("✅ Economic processing completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info("")
        return 0
        
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Bronze economic data not found: {e}")
        logger.warning("⚠️ Run bronze/economic_loader.py first")
        return 1
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("")
        return 1


if __name__ == "__main__":
    exit(main())
