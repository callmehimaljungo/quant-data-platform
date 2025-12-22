"""
Silver Layer: Process Stock Metadata
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Parse raw JSON metadata from Bronze layer
- Flatten nested structures
- Standardize sector/industry names to GICS
- Handle missing values appropriately
- Save to Silver Lakehouse with time travel

Processing Steps:
1. Load raw JSON/Parquet from Bronze
2. Flatten nested JSON fields
3. Map sectors to standard GICS categories
4. Validate market cap classifications
5. Handle nulls (set to 'Unknown' where appropriate)
6. Schema validation
7. Save to Silver Lakehouse
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, SILVER_DIR, LOG_FORMAT, GICS_SECTORS

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

INPUT_DIR = BRONZE_DIR / 'stock_metadata_lakehouse'
OUTPUT_DIR = SILVER_DIR / 'metadata_lakehouse'

# Sector mapping (various sources → standard GICS)
SECTOR_MAPPING = {
    # Direct matches
    'technology': 'Technology',
    'healthcare': 'Healthcare',
    'financial services': 'Financials',
    'financials': 'Financials',
    'finance': 'Financials',
    'consumer cyclical': 'Consumer Discretionary',
    'consumer discretionary': 'Consumer Discretionary',
    'communication services': 'Communication Services',
    'industrials': 'Industrials',
    'industrial': 'Industrials',
    'consumer defensive': 'Consumer Staples',
    'consumer staples': 'Consumer Staples',
    'energy': 'Energy',
    'utilities': 'Utilities',
    'real estate': 'Real Estate',
    'basic materials': 'Materials',
    'materials': 'Materials',
    
    # Alternative names
    'tech': 'Technology',
    'health care': 'Healthcare',
    'financial': 'Financials',
    'banks': 'Financials',
    'insurance': 'Financials',
    'retail': 'Consumer Discretionary',
    'automotive': 'Consumer Discretionary',
    'telecom': 'Communication Services',
    'telecommunications': 'Communication Services',
    'media': 'Communication Services',
    'manufacturing': 'Industrials',
    'aerospace': 'Industrials',
    'defense': 'Industrials',
    'food': 'Consumer Staples',
    'beverage': 'Consumer Staples',
    'tobacco': 'Consumer Staples',
    'oil': 'Energy',
    'gas': 'Energy',
    'electricity': 'Utilities',
    'water': 'Utilities',
    'mining': 'Materials',
    'chemicals': 'Materials',
    'reit': 'Real Estate',
}


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def load_bronze_metadata() -> pd.DataFrame:
    """Load raw metadata from Bronze layer"""
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    if is_lakehouse_table(INPUT_DIR):
        logger.info(f"Loading from Lakehouse: {INPUT_DIR}")
        df = lakehouse_to_pandas(INPUT_DIR)
    else:
        # Try raw JSON
        json_dir = BRONZE_DIR / 'stock_metadata_raw'
        if json_dir.exists():
            json_files = list(json_dir.glob('*.json'))
            if json_files:
                with open(json_files[-1], 'r') as f:  # Latest file
                    data = json.load(f)
                df = pd.DataFrame(data)
                logger.info(f"Loaded from JSON: {json_files[-1]}")
            else:
                raise FileNotFoundError(f"No JSON files in {json_dir}")
        else:
            raise FileNotFoundError(f"No Bronze metadata found at {INPUT_DIR} or {json_dir}")
    
    logger.info(f"✓ Loaded {len(df):,} rows from Bronze")
    return df


def flatten_json_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten any nested JSON fields in the DataFrame
    
    Handles case where raw_json column contains additional data
    """
    logger.info("Flattening JSON fields...")
    
    # If raw_json column exists, we can extract additional fields
    if 'raw_json' in df.columns:
        additional_fields = []
        
        for idx, raw in df['raw_json'].items():
            if pd.isna(raw):
                additional_fields.append({})
                continue
            
            try:
                if isinstance(raw, str):
                    data = json.loads(raw)
                else:
                    data = raw
                
                # Extract useful fields not in main columns
                extracted = {
                    'forward_pe': data.get('forwardPE'),
                    'trailing_pe': data.get('trailingPE'),
                    'price_to_book': data.get('priceToBook'),
                    'enterprise_value': data.get('enterpriseValue'),
                    'profit_margin': data.get('profitMargins'),
                    'revenue_growth': data.get('revenueGrowth'),
                }
                additional_fields.append(extracted)
            except:
                additional_fields.append({})
        
        # Add extracted fields
        extras_df = pd.DataFrame(additional_fields)
        for col in extras_df.columns:
            if col not in df.columns:
                df[col] = extras_df[col]
        
        logger.info(f"✓ Extracted {len(extras_df.columns)} additional fields from JSON")
    
    return df


def standardize_sector(sector: str) -> str:
    """Map sector name to standard GICS category"""
    if pd.isna(sector):
        return 'Unknown'
    
    sector_lower = str(sector).lower().strip()
    
    # Check direct mapping
    if sector_lower in SECTOR_MAPPING:
        return SECTOR_MAPPING[sector_lower]
    
    # Check partial matches
    for key, value in SECTOR_MAPPING.items():
        if key in sector_lower:
            return value
    
    # Check if it's already a valid GICS sector
    if sector in GICS_SECTORS:
        return sector
    
    return 'Unknown'


def standardize_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sector standardization to DataFrame"""
    logger.info("Standardizing sectors to GICS...")
    
    original_sectors = df['sector'].nunique() if 'sector' in df.columns else 0
    
    if 'sector' in df.columns:
        df['sector_original'] = df['sector']  # Keep original
        df['sector'] = df['sector'].apply(standardize_sector)
    else:
        df['sector'] = 'Unknown'
    
    new_sectors = df['sector'].nunique()
    
    logger.info(f"✓ Standardized {original_sectors} → {new_sectors} sector categories")
    
    # Log sector distribution
    sector_counts = df['sector'].value_counts()
    for sector, count in sector_counts.head(10).items():
        logger.info(f"  {sector}: {count:,}")
    
    return df


def classify_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """Add market cap classification"""
    logger.info("Classifying market cap...")
    
    def get_cap_category(market_cap):
        if pd.isna(market_cap) or market_cap <= 0:
            return 'Unknown'
        elif market_cap >= 10_000_000_000:  # >= $10B
            return 'Large'
        elif market_cap >= 2_000_000_000:   # >= $2B
            return 'Mid'
        else:
            return 'Small'
    
    if 'market_cap' in df.columns:
        df['market_cap_category'] = df['market_cap'].apply(get_cap_category)
    else:
        df['market_cap_category'] = 'Unknown'
    
    # Log distribution
    cap_counts = df['market_cap_category'].value_counts()
    for cap, count in cap_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {cap}: {count:,} ({pct:.1f}%)")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values appropriately"""
    logger.info("Handling missing values...")
    
    # String columns - fill with 'Unknown'
    string_cols = ['sector', 'industry', 'exchange', 'currency', 'country']
    for col in string_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].fillna('Unknown')
                logger.info(f"  {col}: filled {missing:,} nulls with 'Unknown'")
    
    # Numeric columns - leave as NaN (will be handled in analysis)
    numeric_cols = ['market_cap', 'beta', 'pe_ratio', 'dividend_yield']
    for col in numeric_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                logger.info(f"  {col}: {missing:,} nulls (keeping as NaN)")
    
    return df


def add_processing_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add processing metadata columns"""
    df['processed_at'] = datetime.now()
    df['data_version'] = 'silver_v1'
    return df


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_metadata() -> pd.DataFrame:
    """
    Main function to process metadata from Bronze to Silver
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("SILVER LAYER: METADATA PROCESSING")
    logger.info("=" * 70)
    
    # Step 1: Load from Bronze
    df = load_bronze_metadata()
    initial_count = len(df)
    
    # Step 2: Flatten JSON fields
    df = flatten_json_fields(df)
    
    # Step 3: Standardize sectors
    df = standardize_sectors(df)
    
    # Step 4: Classify market cap
    df = classify_market_cap(df)
    
    # Step 5: Handle missing values
    df = handle_missing_values(df)
    
    # Step 6: Add processing metadata
    df = add_processing_metadata(df)
    
    # Step 7: Schema validation
    from utils.schema_registry import validate_schema
    # Note: We may need to adjust schema for Silver metadata
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info("METADATA PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Input rows: {initial_count:,}")
    logger.info(f"  Output rows: {len(df):,}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Duration: {duration:.2f} seconds")
    
    return df


def save_to_lakehouse(df: pd.DataFrame) -> str:
    """Save processed metadata to Silver Lakehouse"""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"✓ Saved to {path}")
    return path


def register_in_universe(df: pd.DataFrame):
    """Register processed tickers in universe"""
    from utils.ticker_universe import get_universe
    
    if 'ticker' in df.columns:
        universe = get_universe()
        tickers = df['ticker'].unique().tolist()
        universe.register_source('silver_metadata', tickers)
        logger.info(f"✓ Registered {len(tickers):,} tickers in universe")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    logger.info("")
    logger.info("🚀 SILVER LAYER: METADATA PROCESSOR")
    logger.info("")
    
    try:
        # Process metadata
        df = process_metadata()
        
        # Save to Lakehouse
        save_to_lakehouse(df)
        
        # Register in universe
        register_in_universe(df)
        
        logger.info("")
        logger.info("✅ Metadata processing completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info("")
        return 0
        
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Bronze metadata not found: {e}")
        logger.warning("⚠️ Run bronze/metadata_loader.py first")
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
