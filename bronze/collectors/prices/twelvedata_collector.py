"""
Twelve Data Collector for Stock Prices
Free tier: 800 requests/day, 8 req/min
"""

import os
from typing import List
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import time

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bronze.collectors.base_collector import BaseCollector, RateLimitError, DataCollectionError

logger = logging.getLogger(__name__)


class TwelveDataCollector(BaseCollector):
    """Collector for Twelve Data API"""
    
    BASE_URL = "https://api.twelvedata.com"
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key or os.getenv('TWELVE_DATA_API_KEY'))
        self.min_request_interval = 7.5  # 8 req/min = 7.5 sec between requests
        
    def get_name(self) -> str:
        return "TwelveData"
    
    def get_priority(self) -> int:
        return 1  # High priority (works!)
    
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    def fetch(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data from Twelve Data"""
        self._validate_date_range(start_date, end_date)
        
        if not self.is_available():
            raise DataCollectionError("Twelve Data API key not configured")
        
        logger.info(f"[TwelveData] Fetching {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_records = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                self._rate_limit_wait()
                
                # Twelve Data endpoint: /time_series
                url = f"{self.BASE_URL}/time_series"
                params = {
                    'symbol': ticker,
                    'interval': '1day',
                    'start_date': start_date,
                    'end_date': end_date,
                    'apikey': self.api_key,
                    'format': 'JSON'
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    raise RateLimitError("Twelve Data rate limit exceeded")
                
                if response.status_code != 200:
                    logger.warning(f"  [{i}/{len(tickers)}] {ticker}: HTTP {response.status_code}")
                    continue
                
                data = response.json()
                
                if 'values' not in data or not data['values']:
                    logger.debug(f"  [{i}/{len(tickers)}] {ticker}: No data")
                    continue
                
                # Parse values
                for value in data['values']:
                    all_records.append({
                        'date': pd.to_datetime(value['datetime']),
                        'ticker': ticker,
                        'open': float(value['open']),
                        'high': float(value['high']),
                        'low': float(value['low']),
                        'close': float(value['close']),
                        'volume': int(value['volume'])
                    })
                
                if i % 50 == 0:
                    logger.info(f"  Progress: {i}/{len(tickers)} tickers")
                    
            except RateLimitError:
                raise
            except Exception as e:
                logger.warning(f"  [{i}/{len(tickers)}] {ticker}: {e}")
                continue
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df = self._normalize_dataframe(df)
        
        logger.info(f"[TwelveData] ✅ Fetched {len(df):,} rows for {df['ticker'].nunique()} tickers")
        return df
