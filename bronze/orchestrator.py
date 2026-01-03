"""
Data Orchestrator
Smart fallback logic for multi-source data collection
"""

from typing import List, Dict, Type
import pandas as pd
import logging

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bronze.collectors.base_collector import BaseCollector, RateLimitError, DataCollectionError

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """Orchestrates data collection from multiple sources with automatic fallback"""
    
    def __init__(self):
        self.collectors: Dict[str, List[BaseCollector]] = {
            'prices': [],
            'news': [],
            'economic': []
        }
        self._init_collectors()
    
    def _init_collectors(self):
        """Initialize all available collectors"""
        # Price collectors
        try:
            from bronze.collectors.prices.polygon_collector import PolygonCollector
            self.collectors['prices'].append(PolygonCollector())
        except Exception as e:
            logger.debug(f"Polygon collector not available: {e}")
        
        try:
            from bronze.collectors.prices.yfinance_collector import YFinanceCollector
            self.collectors['prices'].append(YFinanceCollector())
        except Exception as e:
            logger.debug(f"YFinance collector not available: {e}")
        
        # Sort by priority
        for data_type in self.collectors:
            self.collectors[data_type].sort(key=lambda x: x.get_priority())
    
    def collect_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect price data with automatic fallback.
        
        Tries collectors in priority order until one succeeds.
        """
        return self._collect_with_fallback('prices', tickers, start_date, end_date)
    
    def _collect_with_fallback(
        self, 
        data_type: str, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Try collectors in priority order until success"""
        
        if data_type not in self.collectors:
            raise ValueError(f"Unknown data type: {data_type}")
        
        available_collectors = [c for c in self.collectors[data_type] if c.is_available()]
        
        if not available_collectors:
            raise DataCollectionError(f"No available collectors for {data_type}")
        
        logger.info(f"")
        logger.info(f"=" * 70)
        logger.info(f"DATA ORCHESTRATOR: {data_type.upper()}")
        logger.info(f"=" * 70)
        logger.info(f"  Tickers: {len(tickers)}")
        logger.info(f"  Date Range: {start_date} to {end_date}")
        logger.info(f"  Available Collectors: {[c.get_name() for c in available_collectors]}")
        logger.info(f"")
        
        for i, collector in enumerate(available_collectors, 1):
            try:
                logger.info(f"[{i}/{len(available_collectors)}] Trying {collector.get_name()}...")
                
                df = collector.fetch(tickers, start_date, end_date)
                
                if df.empty:
                    logger.warning(f"  ⚠️ {collector.get_name()} returned no data")
                    continue
                
                logger.info(f"")
                logger.info(f"=" * 70)
                logger.info(f"✅ SUCCESS: {collector.get_name()}")
                logger.info(f"  Rows: {len(df):,}")
                logger.info(f"  Tickers: {df['ticker'].nunique()}")
                logger.info(f"  Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
                logger.info(f"=" * 70)
                logger.info(f"")
                
                return df
                
            except RateLimitError as e:
                logger.warning(f"  ⏸️ {collector.get_name()} rate limited: {e}")
                continue
                
            except Exception as e:
                logger.error(f"  ❌ {collector.get_name()} failed: {e}")
                continue
        
        raise DataCollectionError(f"All {data_type} collectors failed")
    
    def get_available_collectors(self, data_type: str = None) -> Dict[str, List[str]]:
        """Get list of available collectors by type"""
        if data_type:
            return {
                data_type: [c.get_name() for c in self.collectors[data_type] if c.is_available()]
            }
        
        return {
            dt: [c.get_name() for c in collectors if c.is_available()]
            for dt, collectors in self.collectors.items()
        }
