
# Data Loader Utilities
# Handles caching, R2 fallback, and data cleaning

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from utils.quant_validators import clean_financial_data
from config import GOLD_DIR, GICS_SECTORS

# R2 loader import
try:
    from dashboard.r2_loader import (
        load_latest_from_lakehouse, 
        is_r2_available,
        load_parquet_from_r2,
        get_r2_object_last_modified
    )
    R2_LOADER_AVAILABLE = True
except ImportError:
    R2_LOADER_AVAILABLE = False


def get_cache_key() -> str:
    """
    Get cache key based on refresh trigger file.
    Returns timestamp of last refresh, forcing cache invalidation.
    """
    trigger_file = GOLD_DIR / '.refresh_trigger'
    if trigger_file.exists():
        return str(trigger_file.stat().st_mtime)
    return "default"


@st.cache_data(ttl=600)  # 10 min - Strategies update 1x/day
def load_risk_metrics(_cache_key: str = None) -> pd.DataFrame:
    """Load risk metrics from Gold layer (cache first, then R2, then local)"""
    
    df = None
    
    # Smart Sync: Check if R2 is newer than local
    use_r2 = False
    cache_file = GOLD_DIR / 'cache' / 'risk_metrics.parquet'
    
    if R2_LOADER_AVAILABLE:
        r2_key = 'processed/gold/cache/risk_metrics.parquet'
        # Get timestamps
        r2_time = get_r2_object_last_modified(r2_key) # efficient head request (cached)
        
        # FORCE R2 STRATEGY
        # We always prefer R2 data because Streamlit Cloud local cache is ephemeral/unreliable
        # and we want to ensure the latest data is shown.
        if r2_time:
             use_r2 = True
             # Optional: Log the timestamp for debugging
             print(f"R2 Data Available: {r2_time}")
        else:
             # If get_r2_object_last_modified returned None, maybe R2 is down?
             # Only then fallback to local
             use_r2 = False
    
    df = None
    
    # 1. Try R2 if it's newer
    if use_r2:
        df = load_parquet_from_r2(r2_key)
        
    # 2. Try Gold cache (Local) if R2 failed or R2 wasn't newer
    if df is None and cache_file.exists():
        df = pd.read_parquet(cache_file)
    
    # 3. If still None (Local missing AND R2 failed), try R2 as fallback
    if df is None and R2_LOADER_AVAILABLE and not use_r2: 
        # (This avoids re-downloading if we already tried R2 above)
        df = load_parquet_from_r2(r2_key)

    # Try strategy weights from cache (Legacy fallback)
    if df is None:
        df = load_parquet_from_r2('processed/gold/cache/realtime_metrics.parquet')
        
        # Fallback to strategy weights
        if df is None:
            for strategy in ['low_beta_quality', 'sector_rotation', 'sentiment_allocation']:
                r2_key = f'processed/gold/cache/{strategy}_weights.parquet'
                df = load_parquet_from_r2(r2_key)
                if df is not None and len(df) > 0:
                    break
    
    # Try R2 lakehouse (legacy)
    if df is None and R2_LOADER_AVAILABLE:
        df = load_latest_from_lakehouse('processed/gold/ticker_metrics_lakehouse/')
    
    # Try local lakehouse
    if df is None:
        for path in [GOLD_DIR / 'ticker_metrics_lakehouse', GOLD_DIR / 'risk_metrics_lakehouse']:
            if path.exists():
                parquet_files = sorted(path.glob('*.parquet'), key=lambda x: x.stat().st_mtime, reverse=True)
                if parquet_files:
                    df = pd.read_parquet(parquet_files[0])
                    break
    
    # Fallback: create sample
    if df is None or len(df) == 0:
        return create_sample_risk_metrics()
    
    # Ensure columns & metadata
    if 'sharpe_ratio' not in df.columns:
        if 'avg_return' in df.columns and 'volatility' in df.columns:
            rf = 0.04
            df['sharpe_ratio'] = (df['avg_return'] * 252 - rf) / (df['volatility'] * np.sqrt(252) + 0.001)
        else:
            df['sharpe_ratio'] = np.random.uniform(0.5, 2.0, len(df))
    
    if 'max_drawdown' not in df.columns:
        if 'volatility' in df.columns:
            df['max_drawdown'] = -df['volatility'] * 100 * 2
        else:
            df['max_drawdown'] = np.random.uniform(-40, -10, len(df))
            
    if 'avg_daily_return' not in df.columns:
         df['avg_daily_return'] = df['avg_ret'] if 'avg_ret' in df.columns else np.random.uniform(-0.001, 0.002, len(df))

    if 'avg_volume' not in df.columns:
        df['avg_volume'] = np.random.uniform(1e6, 1e8, len(df))

    # Apply sector metadata
    try:
        from gold.utils import add_sector_metadata
        if 'ticker' in df.columns:
            df = add_sector_metadata(df)
    except Exception:
        pass
    
    df = clean_financial_data(df)
    
    # All risk filtering (Vol < 80%, DD > -70%) is now handled at the Gold Layer (gold/risk_metrics.py)
    # The dashboard simply loads the refined ticker universe.
    print(f"Loaded {len(df):,} tickers from Gold layer.")
    
    return df


@st.cache_data(ttl=600)  # 10 min - Sectors update 1x/day
def load_sector_metrics(_cache_key: str = None) -> pd.DataFrame:
    """Load sector-level metrics"""
    if R2_LOADER_AVAILABLE:
        df = load_latest_from_lakehouse('processed/gold/sector_metrics_lakehouse/')
        if df is not None and len(df) > 0:
            return df
    
    possible_paths = [GOLD_DIR / 'sector_metrics_lakehouse', GOLD_DIR / 'sector_risk_metrics_lakehouse']
    for path in possible_paths:
        if path.exists():
            parquet_files = sorted(path.glob('*.parquet'), key=lambda x: x.stat().st_mtime, reverse=True)
            if parquet_files:
                return pd.read_parquet(parquet_files[0])
    
    return create_sample_sector_metrics()


def create_sample_risk_metrics() -> pd.DataFrame:
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'PG']
    return pd.DataFrame({
        'ticker': tickers,
        'sector': np.random.choice(GICS_SECTORS, len(tickers)),
        'sharpe_ratio': np.random.uniform(0.5, 2.5, len(tickers)),
        'volatility': np.random.uniform(0.15, 0.45, len(tickers)),
        'max_drawdown': np.random.uniform(-50, -10, len(tickers)),
        'avg_daily_return': np.random.uniform(-0.001, 0.002, len(tickers)),
        'avg_volume': np.random.uniform(1e6, 1e8, len(tickers)),
    })

@st.cache_data(ttl=600)
def load_strategy_weights(strategy_name: str, _cache_key: str = None) -> pd.DataFrame:
    """Load portfolio weights for a specific strategy from Gold cache"""
    cache_file = GOLD_DIR / 'cache' / f'{strategy_name}_weights.parquet'
    
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error loading {strategy_name} cache: {e}")
            return pd.DataFrame()
            
    return pd.DataFrame()


def create_sample_sector_metrics() -> pd.DataFrame:
    return pd.DataFrame({
        'sector': GICS_SECTORS,
        'num_tickers': np.random.randint(50, 500, len(GICS_SECTORS)),
        'sharpe_ratio': np.random.uniform(0.8, 1.8, len(GICS_SECTORS)),
        'volatility': np.random.uniform(0.18, 0.35, len(GICS_SECTORS)),
        'max_drawdown': np.random.uniform(-40, -15, len(GICS_SECTORS)),
    })
