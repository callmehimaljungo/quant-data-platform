"""
Ticker Universe Manager
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Manage the universe of tickers across all data sources
- Compute intersection of tickers that have data from multiple sources
- Filter data to only include tickers with diverse information

Business Context:
- Only analyze tickers that have: price data + metadata + news + economic context
- Ensures comprehensive analysis coverage
- Reduces noise from incomplete data
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional
import pandas as pd
import logging
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, SILVER_DIR, METADATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TICKER UNIVERSE CLASS
# =============================================================================

class TickerUniverse:
    """
    Manages ticker universe across multiple data sources
    
    Computes intersection of tickers available in different data sources
    to ensure only tickers with comprehensive data are analyzed.
    """
    
    def __init__(self):
        self.sources: Dict[str, Set[str]] = {}
        self.metadata_path = METADATA_DIR / 'ticker_universe.json'
        self._load_metadata()
    
    def _load_metadata(self):
        """Load ticker universe metadata from disk"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.sources = {k: set(v) for k, v in data.get('sources', {}).items()}
                logger.info(f"Loaded ticker universe from {self.metadata_path}")
        else:
            self.sources = {}
    
    def _save_metadata(self):
        """Save ticker universe metadata to disk"""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'sources': {k: list(v) for k, v in self.sources.items()},
            'updated_at': datetime.now().isoformat()
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved ticker universe to {self.metadata_path}")
    
    def register_source(self, source_name: str, tickers: List[str]):
        """
        Register tickers available in a data source
        
        Args:
            source_name: Name of data source (e.g., 'prices', 'metadata', 'news')
            tickers: List of ticker symbols
        """
        # Normalize tickers (uppercase, trimmed)
        normalized = {t.strip().upper() for t in tickers if t and isinstance(t, str)}
        self.sources[source_name] = normalized
        logger.info(f"Registered {len(normalized):,} tickers for source '{source_name}'")
        self._save_metadata()
    
    def get_source_tickers(self, source_name: str) -> Set[str]:
        """Get tickers for a specific source"""
        return self.sources.get(source_name, set())
    
    def get_intersection(self, source_names: List[str] = None) -> Set[str]:
        """
        Get intersection of tickers across specified sources
        
        Args:
            source_names: List of source names to intersect (None = all sources)
            
        Returns:
            Set of tickers present in ALL specified sources
        """
        if not self.sources:
            return set()
        
        if source_names is None:
            source_names = list(self.sources.keys())
        
        # Filter to existing sources
        valid_sources = [s for s in source_names if s in self.sources]
        
        if not valid_sources:
            return set()
        
        # Compute intersection
        result = self.sources[valid_sources[0]].copy()
        for source in valid_sources[1:]:
            result &= self.sources[source]
        
        logger.info(f"Intersection of {valid_sources}: {len(result):,} tickers")
        return result
    
    def get_union(self) -> Set[str]:
        """Get union of all tickers across all sources"""
        result = set()
        for tickers in self.sources.values():
            result |= tickers
        return result
    
    def get_coverage_report(self) -> pd.DataFrame:
        """
        Generate coverage report showing ticker availability across sources
        
        Returns:
            DataFrame with coverage statistics
        """
        if not self.sources:
            return pd.DataFrame()
        
        all_tickers = self.get_union()
        
        data = []
        for ticker in sorted(all_tickers):
            row = {'ticker': ticker}
            for source in self.sources:
                row[source] = ticker in self.sources[source]
            row['source_count'] = sum(1 for s in self.sources if ticker in self.sources[s])
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Summary
        summary = {
            'source': list(self.sources.keys()),
            'ticker_count': [len(self.sources[s]) for s in self.sources]
        }
        
        return df
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_sources': len(self.sources),
            'sources': {name: len(tickers) for name, tickers in self.sources.items()},
            'union_count': len(self.get_union()),
            'intersection_count': len(self.get_intersection()),
        }
    
    def filter_dataframe(
        self, 
        df: pd.DataFrame, 
        ticker_column: str = 'ticker',
        min_sources: int = None,
        required_sources: List[str] = None
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include tickers in the universe
        
        Args:
            df: DataFrame to filter
            ticker_column: Name of ticker column
            min_sources: Minimum number of sources a ticker must appear in
            required_sources: List of sources ticker must appear in
            
        Returns:
            Filtered DataFrame
        """
        if ticker_column not in df.columns:
            raise ValueError(f"Column '{ticker_column}' not found in DataFrame")
        
        # Determine valid tickers
        if required_sources:
            valid_tickers = self.get_intersection(required_sources)
        elif min_sources:
            # Get tickers appearing in at least min_sources
            ticker_counts = {}
            for source, tickers in self.sources.items():
                for t in tickers:
                    ticker_counts[t] = ticker_counts.get(t, 0) + 1
            valid_tickers = {t for t, c in ticker_counts.items() if c >= min_sources}
        else:
            valid_tickers = self.get_intersection()  # Default: all sources
        
        # Normalize ticker column
        original_len = len(df)
        df_ticker_upper = df[ticker_column].str.upper().str.strip()
        mask = df_ticker_upper.isin(valid_tickers)
        
        filtered_df = df[mask].copy()
        
        logger.info(f"Filtered {original_len:,} → {len(filtered_df):,} rows "
                   f"({len(valid_tickers):,} valid tickers)")
        
        return filtered_df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance
_universe = None

def get_universe() -> TickerUniverse:
    """Get global ticker universe instance"""
    global _universe
    if _universe is None:
        _universe = TickerUniverse()
    return _universe


def register_tickers_from_bronze():
    """
    Auto-register tickers from all Bronze layer sources
    """
    universe = get_universe()
    
    # 1. Prices
    prices_file = BRONZE_DIR / 'all_stock_data.parquet'
    if prices_file.exists():
        df = pd.read_parquet(prices_file, columns=['Ticker'])
        tickers = df['Ticker'].unique().tolist()
        universe.register_source('prices', tickers)
    
    # 2. Metadata (if exists)
    metadata_dir = BRONZE_DIR / 'stock_metadata_lakehouse'
    if metadata_dir.exists():
        try:
            from utils.lakehouse_helper import lakehouse_to_pandas
            df = lakehouse_to_pandas(metadata_dir)
            tickers = df['ticker'].unique().tolist()
            universe.register_source('metadata', tickers)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
    
    # 3. News (if exists)
    news_dir = BRONZE_DIR / 'market_news_lakehouse'
    if news_dir.exists():
        try:
            from utils.lakehouse_helper import lakehouse_to_pandas
            df = lakehouse_to_pandas(news_dir)
            # News may have tickers as JSON array or comma-separated
            if 'tickers' in df.columns:
                all_tickers = set()
                for t in df['tickers'].dropna():
                    if isinstance(t, list):
                        all_tickers.update(t)
                    elif isinstance(t, str):
                        all_tickers.update([x.strip() for x in t.split(',')])
                universe.register_source('news', list(all_tickers))
        except Exception as e:
            logger.warning(f"Could not load news: {e}")
    
    # 4. Benchmarks (if exists)
    benchmarks_dir = BRONZE_DIR / 'benchmarks_lakehouse'
    if benchmarks_dir.exists():
        try:
            from utils.lakehouse_helper import lakehouse_to_pandas
            df = lakehouse_to_pandas(benchmarks_dir)
            tickers = df['ticker'].unique().tolist()
            universe.register_source('benchmarks', tickers)
        except Exception as e:
            logger.warning(f"Could not load benchmarks: {e}")
    
    return universe.get_summary()


def get_analyzed_tickers(min_sources: int = 2) -> List[str]:
    """
    Get list of tickers that should be analyzed
    (appear in at least min_sources data sources)
    
    Args:
        min_sources: Minimum number of sources required
        
    Returns:
        List of ticker symbols
    """
    universe = get_universe()
    
    if len(universe.sources) < min_sources:
        logger.warning(f"Only {len(universe.sources)} sources available, "
                      f"returning all tickers from intersection")
        return list(universe.get_intersection())
    
    # Get tickers appearing in at least min_sources
    ticker_counts = {}
    for source, tickers in universe.sources.items():
        for t in tickers:
            ticker_counts[t] = ticker_counts.get(t, 0) + 1
    
    valid_tickers = sorted([t for t, c in ticker_counts.items() if c >= min_sources])
    
    logger.info(f"Found {len(valid_tickers):,} tickers appearing in >= {min_sources} sources")
    
    return valid_tickers


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TICKER UNIVERSE MANAGER")
    print("=" * 70)
    
    # Initialize universe
    universe = get_universe()
    
    # Try to register from Bronze
    print("\nRegistering tickers from Bronze layer...")
    summary = register_tickers_from_bronze()
    
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get intersection
    if universe.sources:
        intersection = universe.get_intersection()
        print(f"\nIntersection (all sources): {len(intersection):,} tickers")
        
        if intersection:
            print(f"  Sample: {sorted(list(intersection))[:10]}")
    
    print("\n" + "=" * 70)
