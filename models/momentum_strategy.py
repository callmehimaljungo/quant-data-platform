"""
ML-enhanced momentum strategy
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import numpy as np

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

OUTPUT_PATH = GOLD_DIR / 'momentum_portfolio_lakehouse'


class MLMomentumStrategy:
    """
    Momentum strategy with optional ML filter
    12-1 momentum: rank by 12-month return, skip most recent month
    """
    
    def __init__(self, 
                 lookback_months: int = 12,
                 skip_months: int = 1,
                 top_n: int = 30,
                 ml_threshold: float = 0.55):
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.top_n = top_n
        self.ml_threshold = ml_threshold
    
    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum score for each stock"""
        df = df.copy()
        
        lookback_days = self.lookback_months * 21
        skip_days = self.skip_months * 21
        
        def calc_momentum(group):
            if len(group) < lookback_days:
                return np.nan
            
            sorted_group = group.sort_values('date')
            
            if len(sorted_group) < lookback_days + skip_days:
                return np.nan
            
            end_price = sorted_group['close'].iloc[-skip_days - 1] if skip_days > 0 else sorted_group['close'].iloc[-1]
            start_price = sorted_group['close'].iloc[-lookback_days - skip_days]
            
            if start_price <= 0:
                return np.nan
            
            momentum = (end_price / start_price) - 1
            return momentum
        
        momentum_scores = df.groupby('ticker').apply(calc_momentum)
        momentum_df = momentum_scores.reset_index()
        momentum_df.columns = ['ticker', 'momentum']
        
        return momentum_df
    
    def build_portfolio(self, df: pd.DataFrame, 
                         ml_predictions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build momentum portfolio with optional ML filter
        """
        logger.info("=" * 50)
        logger.info("ML-ENHANCED MOMENTUM STRATEGY")
        logger.info("=" * 50)
        logger.info("Calculating momentum scores...")
        
        momentum_df = self.calculate_momentum(df)
        momentum_df = momentum_df.dropna(subset=['momentum'])
        
        latest_date = df['date'].max()
        logger.info(f"Date: {latest_date}")
        logger.info(f"Tickers with momentum: {len(momentum_df):,}")
        
        # Merge with sector info
        sector_info = df.groupby('ticker')['sector'].first().reset_index()
        momentum_df = momentum_df.merge(sector_info, on='ticker', how='left')
        momentum_df['sector'] = momentum_df['sector'].fillna('Unknown')
        
        # Apply ML filter if available
        if ml_predictions is not None and len(ml_predictions) > 0:
            logger.info("Applying ML filter...")
            
            before_count = len(momentum_df)
            
            ml_subset = ml_predictions[['ticker', 'xgb_probability']].copy()
            momentum_df = momentum_df.merge(ml_subset, on='ticker', how='left')
            
            # Filter: keep stocks where ML predicts UP with high confidence
            # OR keep if no ML prediction available
            momentum_df = momentum_df[
                (momentum_df['xgb_probability'].isna()) | 
                (momentum_df['xgb_probability'] >= self.ml_threshold) |
                (momentum_df['xgb_probability'] <= 1 - self.ml_threshold)
            ]
            
            after_count = len(momentum_df)
            logger.info(f"ML filter: {before_count} -> {after_count} tickers")
        
        # Rank by momentum
        momentum_df = momentum_df.sort_values('momentum', ascending=False)
        
        # Select top N
        df_portfolio = momentum_df.head(self.top_n).copy()
        
        # Equal weight
        df_portfolio['weight'] = 1.0 / len(df_portfolio)
        
        logger.info(f"[OK] Portfolio: {len(df_portfolio)} stocks")
        logger.info(f"Average momentum: {df_portfolio['momentum'].mean():.4f}")
        
        return df_portfolio[['ticker', 'sector', 'momentum', 'weight']]


def run_momentum_strategy(test_mode: bool = False) -> Dict:
    """Run momentum strategy pipeline"""
    from utils import pandas_to_lakehouse, lakehouse_to_pandas
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("ML-ENHANCED MOMENTUM STRATEGY")
    logger.info("=" * 70)
    
    parquet_path = SILVER_DIR / 'enriched_stocks.parquet'
    
    if not parquet_path.exists():
        logger.error(f"Data not found: {parquet_path}")
        return {}
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"[OK] Loaded {len(df):,} rows")
    
    # Load ML predictions if available
    ml_predictions = None
    ml_path = GOLD_DIR / 'ml_predictions_lakehouse'
    
    try:
        ml_predictions = lakehouse_to_pandas(ml_path)
        logger.info(f"[OK] ML predictions loaded: {len(ml_predictions)} rows")
    except Exception as e:
        logger.warning(f"ML predictions not found, running without filter")
    
    strategy = MLMomentumStrategy(
        lookback_months=12,
        skip_months=1,
        top_n=30,
        ml_threshold=0.55
    )
    
    portfolio = strategy.build_portfolio(df, ml_predictions)
    
    # Save portfolio
    pandas_to_lakehouse(portfolio, OUTPUT_PATH, mode='overwrite')
    logger.info(f"[OK] Saved to {OUTPUT_PATH}")
    
    # Print portfolio
    print("\n--- MOMENTUM PORTFOLIO ---")
    print(portfolio.to_string(index=False))
    
    return {
        'portfolio_size': len(portfolio),
        'avg_momentum': portfolio['momentum'].mean()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    run_momentum_strategy(test_mode=args.test)


if __name__ == "__main__":
    main()
