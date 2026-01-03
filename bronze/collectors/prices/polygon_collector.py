"""
Polygon.io Price Collector
Best quality free tier: 5 requests/minute
"""

import os
from typing import List
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bronze.collectors.base_collector import BaseCollector, RateLimitError, DataCollectionError

logger = logging.getLogger(__name__)


class PolygonCollector(BaseCollector):
    """Collector for Polygon.io API"""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key or os.getenv('POLYGON_API_KEY'))
        self.min_request_interval = 12.0  # 5 req/min = 12 sec between requests
        
    def get_name(self) -> str:
        return "Polygon.io"
    
    def get_priority(self) -> int:
        return 1  # Highest priority (best quality)
    
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    def fetch(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data from Polygon.io"""
        self._validate_date_range(start_date, end_date)
        
        if not self.is_available():
            raise DataCollectionError("Polygon API key not configured")
        
        logger.info(f"[Polygon.io] Fetching {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_records = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                self._rate_limit_wait()
                
                # Polygon endpoint: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
                url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
                params = {'apiKey': self.api_key, 'adjusted': 'true', 'sort': 'asc'}
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    raise RateLimitError("Polygon.io rate limit exceeded")
                
                if response.status_code != 200:
                    logger.warning(f"  [{i}/{len(tickers)}] {ticker}: HTTP {response.status_code}")
                    continue
                
                data = response.json()
                
                if data.get('status') != 'OK' or not data.get('results'):
                    logger.debug(f"  [{i}/{len(tickers)}] {ticker}: No data")
                    continue
                
                # Parse results
                for bar in data['results']:
                    all_records.append({
                        'date': pd.to_datetime(bar['t'], unit='ms'),
                        'ticker': ticker,
                        'open': float(bar['o']),
                        'high': float(bar['h']),
                        'low': float(bar['l']),
                        'close': float(bar['c']),
                        'volume': int(bar['v'])
                    })
                
                if i % 10 == 0:
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
        
        logger.info(f"[Polygon.io] ✅ Fetched {len(df):,} rows for {df['ticker'].nunique()} tickers")
        return df
