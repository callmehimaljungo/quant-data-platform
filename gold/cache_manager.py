"""
Gold Layer Cache Manager
Pre-compute and cache heavy calculations for lightweight streaming
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

from config import GOLD_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = GOLD_DIR / 'cache'


class CacheManager:
    """Manage pre-computed cache for streaming mode"""
    
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.metadata_file = CACHE_DIR / 'cache_metadata.json'
    
    def save_portfolio_weights(self, weights: pd.DataFrame, strategy_name: str):
        """Save portfolio weights after batch computation"""
        output_file = CACHE_DIR / f'{strategy_name}_weights.parquet'
        weights.to_parquet(output_file, index=False)
        logger.info(f"[CACHE] Saved {strategy_name} weights to {output_file}")
        self._update_metadata(strategy_name, 'weights')
        return output_file
    
    def load_portfolio_weights(self, strategy_name: str) -> pd.DataFrame:
        """Load cached portfolio weights for streaming"""
        cache_file = CACHE_DIR / f'{strategy_name}_weights.parquet'
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        else:
            logger.warning(f"[CACHE] No cached weights for {strategy_name}")
            return pd.DataFrame()
    
    def save_risk_metrics(self, metrics: pd.DataFrame):
        """Save pre-computed risk metrics"""
        output_file = CACHE_DIR / 'risk_metrics.parquet'
        metrics.to_parquet(output_file, index=False)
        logger.info(f"[CACHE] Saved risk metrics to {output_file}")
        self._update_metadata('risk_metrics', 'metrics')
        return output_file
    
    def load_risk_metrics(self) -> pd.DataFrame:
        """Load cached risk metrics"""
        cache_file = CACHE_DIR / 'risk_metrics.parquet'
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return pd.DataFrame()
    
    def save_dashboard_data(self, data: pd.DataFrame):
        """Save pre-aggregated dashboard data"""
        output_file = CACHE_DIR / 'dashboard_daily.parquet'
        data.to_parquet(output_file, index=False)
        logger.info(f"[CACHE] Saved dashboard data to {output_file}")
        self._update_metadata('dashboard', 'data')
        return output_file
    
    def load_dashboard_data(self) -> pd.DataFrame:
        """Load cached dashboard data"""
        cache_file = CACHE_DIR / 'dashboard_daily.parquet'
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return pd.DataFrame()
    
    def append_daily_performance(self, date: str, pnl_data: dict):
        """Append today's performance to history (streaming mode)"""
        history_file = CACHE_DIR / 'performance_history.parquet'
        
        new_row = pd.DataFrame([{
            'date': pd.to_datetime(date),
            **pnl_data,
            'updated_at': datetime.now()
        }])
        
        if history_file.exists():
            history = pd.read_parquet(history_file)
            # Remove duplicate date if exists
            history = history[history['date'] != pd.to_datetime(date)]
            history = pd.concat([history, new_row], ignore_index=True)
        else:
            history = new_row
        
        history.to_parquet(history_file, index=False)
        logger.info(f"[CACHE] Appended {date} performance")
        return history_file
    
    def _update_metadata(self, component: str, cache_type: str):
        """Update cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata[component] = {
            'type': cache_type,
            'updated_at': datetime.now().isoformat(),
            'batch_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_last_batch_date(self) -> str:
        """Get when cache was last computed"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            if metadata:
                dates = [v.get('batch_date') for v in metadata.values() if v.get('batch_date')]
                if dates:
                    return max(dates)
        return None
    
    def is_cache_valid(self, max_age_days: int = 1) -> bool:
        """Check if cache is still valid"""
        last_batch = self.get_last_batch_date()
        if not last_batch:
            return False
        
        batch_date = datetime.strptime(last_batch, '%Y-%m-%d')
        age = (datetime.now() - batch_date).days
        return age <= max_age_days
    
    def clear_cache(self):
        """Clear all cached data"""
        for file in CACHE_DIR.glob('*.parquet'):
            file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("[CACHE] Cleared all cache")


# Singleton instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
