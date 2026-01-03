"""
YFinance Collector (Fallback)
Free, no API key needed, but has rate limits
"""

from typing import List
import pandas as pd
import logging

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bronze.collectors.base_collector import BaseCollector, RateLimitError, DataCollectionError

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class YFinanceCollector(BaseCollector):
    """Collector for Yahoo Finance (via yfinance library)"""
    
    def __init__(self):
        super().__init__(api_key=None)  # No API key needed
        self.min_request_interval = 0.5  # Be gentle with Yahoo
        
    def get_name(self) -> str:
        return "Yahoo Finance"
    
    def get_priority(self) -> int:
        return 10  # Lowest priority (use as last resort)
    
    def is_available(self) -> bool:
        return YFINANCE_AVAILABLE
    
    def fetch(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance"""
        self._validate_date_range(start_date, end_date)
        
        if not self.is_available():
            raise DataCollectionError("yfinance library not installed")
        
        logger.info(f"[YFinance] Fetching {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_records = []
        BATCH_SIZE = 50
        total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            try:
                self._rate_limit_wait()
                
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    progress=False,
                    threads=False  # Safer for rate limiting
                )
                
                if data.empty:
                    continue
                
                # Process batch
                if len(batch) == 1:
                    t = batch[0]
                    for ts, row in data.iterrows():
                        all_records.append({
                            'date': ts,
                            'ticker': t,
                            'open': float(row.get('Open', 0)),
                            'high': float(row.get('High', 0)),
                            'low': float(row.get('Low', 0)),
                            'close': float(row.get('Close', 0)),
                            'volume': int(row.get('Volume', 0))
                        })
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        for t in batch:
                            try:
                                df_t = data[t].dropna()
                                for ts, row in df_t.iterrows():
                                    all_records.append({
                                        'date': ts,
                                        'ticker': t,
                                        'open': float(row.get('Open', 0)),
                                        'high': float(row.get('High', 0)),
                                        'low': float(row.get('Low', 0)),
                                        'close': float(row.get('Close', 0)),
                                        'volume': int(row.get('Volume', 0))
                                    })
                            except KeyError:
                                pass
                
                logger.info(f"  Batch {batch_num}/{total_batches} complete")
                
            except Exception as e:
                if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                    raise RateLimitError(f"YFinance rate limit: {e}")
                logger.warning(f"  Batch {batch_num} failed: {e}")
                continue
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df = self._normalize_dataframe(df)
        
        logger.info(f"[YFinance] ✅ Fetched {len(df):,} rows for {df['ticker'].nunique()} tickers")
        return df
