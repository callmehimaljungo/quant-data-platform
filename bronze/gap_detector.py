"""
Gap Detector: Find Missing Dates in Bronze Data
Scans partitioned Bronze data to detect gaps and missing dates

Usage:
    python bronze/gap_detector.py --check
    python bronze/gap_detector.py --fill
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def get_existing_dates(partition_dir: Path) -> List[datetime]:
    """Get list of dates that have data in partitioned Bronze"""
    dates = []
    
    if not partition_dir.exists():
        logger.warning(f"Partition directory not found: {partition_dir}")
        return dates
    
    for partition in partition_dir.iterdir():
        if partition.is_dir() and partition.name.startswith('date='):
            date_str = partition.name.replace('date=', '')
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            except ValueError:
                continue
    
    return sorted(dates)


def detect_gaps(dates: List[datetime], expected_freq: str = 'B') -> List[Tuple[datetime, datetime]]:
    """
    Detect gaps in date sequence.
    
    Args:
        dates: List of existing dates
        expected_freq: Expected frequency ('B' = business days, 'D' = calendar days)
    
    Returns:
        List of (gap_start, gap_end) tuples
    """
    if not dates:
        return []
    
    # Generate expected date range
    start = min(dates)
    end = max(dates)
    
    if expected_freq == 'B':
        # Business days only
        expected = pd.bdate_range(start, end)
    else:
        # Calendar days
        expected = pd.date_range(start, end, freq='D')
    
    # Find missing dates
    existing_set = set(d.date() for d in dates)
    missing = [d for d in expected if d.date() not in existing_set]
    
    if not missing:
        return []
    
    # Group consecutive missing dates into gaps
    gaps = []
    gap_start = missing[0]
    gap_end = missing[0]
    
    for i in range(1, len(missing)):
        if (missing[i] - gap_end).days <= 3:  # Allow small gaps (weekends)
            gap_end = missing[i]
        else:
            gaps.append((gap_start, gap_end))
            gap_start = missing[i]
            gap_end = missing[i]
    
    gaps.append((gap_start, gap_end))
    
    return gaps


def check_gaps():
    """Check for gaps in Bronze data"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("GAP DETECTOR: CHECKING BRONZE DATA")
    logger.info("=" * 70)
    
    # Check prices
    prices_dir = BRONZE_DIR / 'prices_partitioned'
    logger.info(f"\nChecking: {prices_dir}")
    
    dates = get_existing_dates(prices_dir)
    
    if not dates:
        logger.warning("No partitioned data found!")
        return
    
    logger.info(f"  Total partitions: {len(dates)}")
    logger.info(f"  Date range: {min(dates).date()} to {max(dates).date()}")
    
    # Detect gaps
    gaps = detect_gaps(dates, expected_freq='B')
    
    if not gaps:
        logger.info("  ✅ No gaps detected!")
    else:
        logger.warning(f"  ⚠️ Found {len(gaps)} gap(s):")
        total_missing = 0
        for gap_start, gap_end in gaps:
            days = (gap_end - gap_start).days + 1
            total_missing += days
            logger.warning(f"    - {gap_start.date()} to {gap_end.date()} ({days} days)")
        
        logger.info(f"\n  Total missing days: {total_missing}")
    
    # Check news
    news_dir = BRONZE_DIR / 'market_news_lakehouse'
    if news_dir.exists():
        news_files = list(news_dir.glob('*.parquet'))
        logger.info(f"\nNews files: {len(news_files)}")
    
    logger.info("")
    logger.info("=" * 70)
    
    return gaps


def fill_gaps(gaps: List[Tuple[datetime, datetime]]):
    """Fill detected gaps using orchestrator"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("GAP FILLER: FILLING MISSING DATA")
    logger.info("=" * 70)
    
    if not gaps:
        logger.info("No gaps to fill!")
        return
    
    from bronze.orchestrator import DataOrchestrator
    from bronze.backfill import get_existing_tickers, append_to_bronze
    
    orchestrator = DataOrchestrator()
    tickers = get_existing_tickers()
    
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Gaps to fill: {len(gaps)}")
    logger.info("")
    
    for i, (gap_start, gap_end) in enumerate(gaps, 1):
        logger.info(f"[{i}/{len(gaps)}] Filling gap: {gap_start.date()} to {gap_end.date()}")
        
        try:
            # Fetch data for gap
            df = orchestrator.collect_prices(
                tickers=tickers,
                start_date=gap_start.strftime('%Y-%m-%d'),
                end_date=gap_end.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                logger.warning(f"  No data collected for this gap")
                continue
            
            # Append to Bronze
            output_path = append_to_bronze(df)
            logger.info(f"  ✅ Filled {len(df):,} rows")
            
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
            continue
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("GAP FILLING COMPLETE")
    logger.info("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Bronze Gap Detector')
    parser.add_argument('--check', action='store_true',
                        help='Check for gaps in data')
    parser.add_argument('--fill', action='store_true',
                        help='Fill detected gaps')
    args = parser.parse_args()
    
    if args.check or not args.fill:
        gaps = check_gaps()
    
    if args.fill:
        if not args.check:
            # Need to detect gaps first
            gaps = check_gaps()
        fill_gaps(gaps)
    
    return 0


if __name__ == "__main__":
    exit(main())
