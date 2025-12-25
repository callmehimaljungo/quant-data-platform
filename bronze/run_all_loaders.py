"""
Bronze Layer: Run All Loaders
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Run all Bronze layer data loaders in sequence
- Register all tickers in universe for intersection filtering
- Generate summary report

Usage:
    python bronze/run_all_loaders.py           # Full run
    python bronze/run_all_loaders.py --test    # Test mode (limited data)
    python bronze/run_all_loaders.py --skip-prices  # Skip price data (already loaded)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_all_loaders(
    test_mode: bool = False,
    skip_prices: bool = False,
    skip_metadata: bool = False,
    skip_news: bool = False,
    skip_economic: bool = False,
    skip_benchmarks: bool = False
):
    """
    Run all Bronze layer loaders
    
    Args:
        test_mode: If True, run with limited data for testing
        skip_*: Skip individual loaders
    """
    start_time = datetime.now()
    results = {}
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(" BRONZE LAYER: RUNNING ALL DATA LOADERS")
    logger.info("=" * 70)
    logger.info(f"Mode: {'TEST' if test_mode else 'FULL'}")
    logger.info("")
    
    # 1. Stock Metadata (JSON from yfinance)
    if not skip_metadata:
        logger.info("-" * 70)
        logger.info(" STEP 1: Loading Stock Metadata (JSON)")
        logger.info("-" * 70)
        try:
            from bronze.metadata_loader import main as metadata_main
            result = metadata_main(max_tickers=50 if test_mode else None, test=test_mode)
            results['metadata'] = 'SUCCESS' if result == 0 else 'FAILED'
        except Exception as e:
            logger.error(f"Metadata loading failed: {e}")
            results['metadata'] = f'ERROR: {e}'
    else:
        results['metadata'] = 'SKIPPED'
        logger.info("⏭️ Skipping metadata loader")
    
    # 2. Market News (JSON + Text)
    if not skip_news:
        logger.info("-" * 70)
        logger.info("📰 STEP 2: Loading Market News (JSON + Text)")
        logger.info("-" * 70)
        try:
            from bronze.news_loader import main as news_main
            result = news_main(num_articles=100 if test_mode else 500, test=test_mode)
            results['news'] = 'SUCCESS' if result == 0 else 'FAILED'
        except Exception as e:
            logger.error(f"News loading failed: {e}")
            results['news'] = f'ERROR: {e}'
    else:
        results['news'] = 'SKIPPED'
        logger.info("⏭️ Skipping news loader")
    
    # 3. Economic Indicators (CSV from FRED)
    if not skip_economic:
        logger.info("-" * 70)
        logger.info(" STEP 3: Loading Economic Indicators (CSV)")
        logger.info("-" * 70)
        try:
            from bronze.economic_loader import main as economic_main
            result = economic_main(start_date='2024-01-01' if test_mode else '2020-01-01', test=test_mode)
            results['economic'] = 'SUCCESS' if result == 0 else 'FAILED'
        except Exception as e:
            logger.error(f"Economic loading failed: {e}")
            results['economic'] = f'ERROR: {e}'
    else:
        results['economic'] = 'SKIPPED'
        logger.info("⏭️ Skipping economic loader")
    
    # 4. Benchmark Data (Parquet from yfinance)
    if not skip_benchmarks:
        logger.info("-" * 70)
        logger.info(" STEP 4: Loading Benchmark Data (Parquet)")
        logger.info("-" * 70)
        try:
            from bronze.benchmark_loader import main as benchmark_main
            result = benchmark_main(start_date='2024-01-01' if test_mode else '2010-01-01', test=test_mode)
            results['benchmarks'] = 'SUCCESS' if result == 0 else 'FAILED'
        except Exception as e:
            logger.error(f"Benchmark loading failed: {e}")
            results['benchmarks'] = f'ERROR: {e}'
    else:
        results['benchmarks'] = 'SKIPPED'
        logger.info("⏭️ Skipping benchmark loader")
    
    # 5. Register existing price data in universe
    if not skip_prices:
        logger.info("-" * 70)
        logger.info("💰 STEP 5: Registering Price Data in Universe")
        logger.info("-" * 70)
        try:
            prices_file = BRONZE_DIR / 'all_stock_data.parquet'
            if prices_file.exists():
                import pandas as pd
                from utils.ticker_universe import get_universe
                
                df = pd.read_parquet(prices_file, columns=['Ticker'])
                tickers = df['Ticker'].unique().tolist()
                
                universe = get_universe()
                universe.register_source('prices', tickers)
                
                results['prices'] = f'SUCCESS ({len(tickers):,} tickers)'
            else:
                results['prices'] = 'NOT FOUND'
                logger.warning(f"Price data not found: {prices_file}")
        except Exception as e:
            logger.error(f"Price registration failed: {e}")
            results['prices'] = f'ERROR: {e}'
    else:
        results['prices'] = 'SKIPPED'
        logger.info("⏭️ Skipping price registration")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📋 BRONZE LAYER SUMMARY")
    logger.info("=" * 70)
    
    for loader, status in results.items():
        emoji = "✅" if 'SUCCESS' in str(status) else "[ERR]" if 'ERROR' in str(status) or 'FAILED' in str(status) else "⏭️"
        logger.info(f"  {emoji} {loader.upper()}: {status}")
    
    logger.info("")
    logger.info(f"Total duration: {duration:.2f} seconds")
    
    # Generate ticker universe report
    logger.info("")
    logger.info("-" * 70)
    logger.info("🌐 TICKER UNIVERSE SUMMARY")
    logger.info("-" * 70)
    
    try:
        from utils.ticker_universe import get_universe
        universe = get_universe()
        summary = universe.get_summary()
        
        logger.info(f"  Total sources: {summary['total_sources']}")
        for source, count in summary.get('sources', {}).items():
            logger.info(f"    - {source}: {count:,} tickers")
        
        logger.info(f"  Union (all tickers): {summary['union_count']:,}")
        logger.info(f"  Intersection (common tickers): {summary['intersection_count']:,}")
        
    except Exception as e:
        logger.warning(f"Could not generate universe summary: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    
    # Return success if all loaders succeeded or were skipped
    success = all('SUCCESS' in str(s) or 'SKIPPED' in str(s) for s in results.values())
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    test_mode = '--test' in sys.argv
    skip_prices = '--skip-prices' in sys.argv
    skip_metadata = '--skip-metadata' in sys.argv
    skip_news = '--skip-news' in sys.argv
    skip_economic = '--skip-economic' in sys.argv
    skip_benchmarks = '--skip-benchmarks' in sys.argv
    
    exit(run_all_loaders(
        test_mode=test_mode,
        skip_prices=skip_prices,
        skip_metadata=skip_metadata,
        skip_news=skip_news,
        skip_economic=skip_economic,
        skip_benchmarks=skip_benchmarks
    ))
