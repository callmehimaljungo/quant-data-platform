"""
Base Collector Abstract Class
All data collectors inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API rate limit is hit"""
    pass


class DataCollectionError(Exception):
    """Raised when data collection fails"""
    pass


class BaseCollector(ABC):
    """Abstract base class for all data collectors"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
        
    @abstractmethod
    def fetch(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for given tickers and date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: date, ticker, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this collector is available (has API key, etc)"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """
        Return priority level (1 = highest priority, 10 = lowest).
        Lower numbers are tried first.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable name of this collector"""
        pass
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limit wait: {wait_time:.2f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _validate_date_range(self, start_date: str, end_date: str):
        """Validate date range format"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start > end:
                raise ValueError("Start date must be before end date")
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard schema.
        Expected columns: date, ticker, open, high, low, close, volume
        """
        if df.empty:
            return df
        
        # Ensure required columns exist
        required = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
        
        # Convert types
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'volume' in df.columns:
            df[col] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        # Sort by date and ticker
        df = df.sort_values(['date', 'ticker'])
        
        return df
