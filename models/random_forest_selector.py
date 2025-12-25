"""
Random Forest feature importance analyzer
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from config import GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

OUTPUT_PATH = GOLD_DIR / 'feature_importance_lakehouse'


class RandomForestFeatureAnalyzer:
    """Analyze feature importance using Random Forest classifier"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.feature_importance = None
    
    def analyze(self, df: pd.DataFrame, feature_cols: List[str], 
                target_col: str = 'target_direction') -> pd.DataFrame:
        """
        Analyze feature importance
        Returns DataFrame with feature rankings
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.error("sklearn not installed")
            return pd.DataFrame()
        
        logger.info("=" * 50)
        logger.info("RANDOM FOREST FEATURE ANALYSIS")
        logger.info("=" * 50)
        
        df = df.dropna(subset=feature_cols + [target_col])
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"Data: {len(X):,} rows, {len(feature_cols)} features")
        logger.info("Training Random Forest...")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['rank'] = range(1, len(self.feature_importance) + 1)
        
        logger.info("")
        logger.info("--- FEATURE IMPORTANCE RANKING ---")
        for _, row in self.feature_importance.head(10).iterrows():
            logger.info(f"  {row['rank']:2d}. {row['feature']:20s}: {row['importance']:.4f}")
        
        return self.feature_importance
    
    def get_top_features(self, n: int = 10) -> List[str]:
        """Get list of top N feature names"""
        if self.feature_importance is None:
            return []
        return self.feature_importance.head(n)['feature'].tolist()


def run_feature_analysis(test_mode: bool = False) -> Dict:
    """Run feature importance analysis"""
    from models.feature_engineering import engineer_features, get_feature_columns
    from utils import pandas_to_lakehouse, lakehouse_to_pandas
    from config import SILVER_DIR
    
    logger.info("=" * 70)
    logger.info("RANDOM FOREST FEATURE ANALYSIS")
    logger.info("=" * 70)
    
    parquet_path = SILVER_DIR / 'enriched_stocks.parquet'
    
    if not parquet_path.exists():
        logger.error(f"Data not found: {parquet_path}")
        return {}
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"[OK] Loaded {len(df):,} rows")
    
    if test_mode:
        top_tickers = df['ticker'].value_counts().head(20).index.tolist()
        df = df[df['ticker'].isin(top_tickers)]
    
    df = engineer_features(df, include_target=True)
    
    feature_cols = get_feature_columns()
    available_features = [c for c in feature_cols if c in df.columns]
    
    analyzer = RandomForestFeatureAnalyzer(n_estimators=100, max_depth=10)
    importance = analyzer.analyze(df, available_features)
    
    if len(importance) > 0:
        pandas_to_lakehouse(importance, OUTPUT_PATH, mode='overwrite')
        logger.info(f"[OK] Results saved to {OUTPUT_PATH}")
    
    return {
        'top_features': analyzer.get_top_features(10),
        'importance': importance
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    run_feature_analysis(test_mode=args.test)


if __name__ == "__main__":
    main()
