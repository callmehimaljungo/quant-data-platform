"""
Finnhub Collector for Recent Data
Free tier: 60 requests/minute
"""

import os
from typing import List
import pandas as pd
import requests
from datetime import datetime
import logging

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bronze.collectors.base_collector import BaseCollector, RateLimitError, DataCollectionError

logger = logging.getLogger(__name__)


class FinnhubCollector(BaseCollector):
    """Collector for Finnhub API"""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key or os.getenv('FINNHUB_API_KEY'))
        self.min_request_interval = 1.0  # 60 req/min = 1 sec between requests
        
    def get_name(self) -> str:
        return "Finnhub"
    
    def get_priority(self) -> int:
        return 2  # Second priority (after Polygon)
    
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    def fetch(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data from Finnhub"""
        self._validate_date_range(start_date, end_date)
        
        if not self.is_available():
            raise DataCollectionError("Finnhub API key not configured")
        
        logger.info(f"[Finnhub] Fetching {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Convert dates to timestamps
        from_ts = int(pd.to_datetime(start_date).timestamp())
        to_ts = int(pd.to_datetime(end_date).timestamp())
        
        all_records = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                self._rate_limit_wait()
                
                # Finnhub endpoint: /stock/candle
                url = f"{self.BASE_URL}/stock/candle"
                params = {
                    'symbol': ticker,
                    'resolution': 'D',  # Daily
                    'from': from_ts,
                    'to': to_ts,
                    'token': self.api_key
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    raise RateLimitError("Finnhub rate limit exceeded")
                
                if response.status_code != 200:
                    logger.warning(f"  [{i}/{len(tickers)}] {ticker}: HTTP {response.status_code}")
                    continue
                
                data = response.json()
                
                if data.get('s') != 'ok' or not data.get('c'):
                    logger.debug(f"  [{i}/{len(tickers)}] {ticker}: No data")
                    continue
                
                # Parse candles
                for j in range(len(data['c'])):
                    all_records.append({
                        'date': pd.to_datetime(data['t'][j], unit='s'),
                        'ticker': ticker,
                        'open': float(data['o'][j]),
                        'high': float(data['h'][j]),
                        'low': float(data['l'][j]),
                        'close': float(data['c'][j]),
                        'volume': int(data['v'][j])
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
        
        logger.info(f"[Finnhub] ✅ Fetched {len(df):,} rows for {df['ticker'].nunique()} tickers")
        return df
