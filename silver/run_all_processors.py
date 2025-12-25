"""
Silver Layer: Run All Processors
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Run all Silver layer processors in sequence
- Process diverse data formats (JSON, Text, CSV, Parquet)
- Join all schemas together
- Generate unified dataset for Gold layer

Usage:
    python silver/run_all_processors.py           # Full run
    python silver/run_all_processors.py --test    # Test mode
    python silver/run_all_processors.py --skip-clean  # Skip price cleaning
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SILVER_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_all_processors(
    skip_clean: bool = False,
    skip_metadata: bool = False,
    skip_news: bool = False,
    skip_economic: bool = False,
    skip_join: bool = False
):
    """
    Run all Silver layer processors
    
    Args:
        skip_*: Skip individual processors
    """
    start_time = datetime.now()
    results = {}
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(" SILVER LAYER: RUNNING ALL PROCESSORS")
    logger.info("=" * 70)
    logger.info("")
    
    # 1. Clean price data
    if not skip_clean:
        logger.info("-" * 70)
        logger.info(" STEP 1: Cleaning Price Data")
        logger.info("-" * 70)
        try:
            from silver.clean import main as clean_main
            result = clean_main()
            results['clean'] = 'SUCCESS' if result == 0 else 'FAILED'
        except Exception as e:
            logger.error(f"Price cleaning failed: {e}")
            results['clean'] = f'ERROR: {e}'
    else:
        results['clean'] = 'SKIPPED'
        logger.info("⏭️ Skipping price cleaning")
    
    # 2. Process metadata (JSON)
    if not skip_metadata:
        logger.info("-" * 70)
        logger.info("📋 STEP 2: Processing Metadata (JSON)")
        logger.info("-" * 70)
        try:
            from silver.process_metadata import main as metadata_main
            result = metadata_main()
            results['metadata'] = 'SUCCESS' if result == 0 else 'FAILED'
        except FileNotFoundError as e:
            logger.warning(f"[WARN] Metadata not available: {e}")
            results['metadata'] = 'NOT AVAILABLE'
        except Exception as e:
            logger.error(f"Metadata processing failed: {e}")
            results['metadata'] = f'ERROR: {e}'
    else:
        results['metadata'] = 'SKIPPED'
        logger.info("⏭️ Skipping metadata processing")
    
    # 3. Process news (JSON + Text)
    if not skip_news:
        logger.info("-" * 70)
        logger.info("📰 STEP 3: Processing News (JSON + Text)")
        logger.info("-" * 70)
        try:
            from silver.process_news import main as news_main
            result = news_main()
            results['news'] = 'SUCCESS' if result == 0 else 'FAILED'
        except FileNotFoundError as e:
            logger.warning(f"[WARN] News not available: {e}")
            results['news'] = 'NOT AVAILABLE'
        except Exception as e:
            logger.error(f"News processing failed: {e}")
            results['news'] = f'ERROR: {e}'
    else:
        results['news'] = 'SKIPPED'
        logger.info("⏭️ Skipping news processing")
    
    # 4. Process economic (CSV)
    if not skip_economic:
        logger.info("-" * 70)
        logger.info(" STEP 4: Processing Economic Data (CSV)")
        logger.info("-" * 70)
        try:
            from silver.process_economic import main as economic_main
            result = economic_main()
            results['economic'] = 'SUCCESS' if result == 0 else 'FAILED'
        except FileNotFoundError as e:
            logger.warning(f"[WARN] Economic data not available: {e}")
            results['economic'] = 'NOT AVAILABLE'
        except Exception as e:
            logger.error(f"Economic processing failed: {e}")
            results['economic'] = f'ERROR: {e}'
    else:
        results['economic'] = 'SKIPPED'
        logger.info("⏭️ Skipping economic processing")
    
    # 5. Join all schemas
    if not skip_join:
        logger.info("-" * 70)
        logger.info("🔗 STEP 5: Joining All Schemas")
        logger.info("-" * 70)
        try:
            from silver.join_schemas import main as join_main
            result = join_main(filter_intersection=True, min_sources=1)
            results['join'] = 'SUCCESS' if result == 0 else 'FAILED'
        except Exception as e:
            logger.error(f"Schema joining failed: {e}")
            results['join'] = f'ERROR: {e}'
    else:
        results['join'] = 'SKIPPED'
        logger.info("⏭️ Skipping schema joining")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📋 SILVER LAYER SUMMARY")
    logger.info("=" * 70)
    
    for processor, status in results.items():
        emoji = "✅" if 'SUCCESS' in str(status) else "[ERR]" if 'ERROR' in str(status) else "⏭️"
        logger.info(f"  {emoji} {processor.upper()}: {status}")
    
    logger.info("")
    logger.info(f"Total duration: {duration:.2f} seconds")
    
    # List created outputs
    logger.info("")
    logger.info("-" * 70)
    logger.info("📁 CREATED OUTPUTS")
    logger.info("-" * 70)
    
    outputs = [
        ('enriched_stocks.parquet', 'Price data (cleaned)'),
        ('metadata_lakehouse/', 'Metadata (processed)'),
        ('news_lakehouse/', 'News sentiment (aggregated)'),
        ('economic_lakehouse/', 'Economic indicators (pivoted)'),
        ('unified_lakehouse/', 'Unified dataset (all joins)'),
    ]
    
    for path, desc in outputs:
        full_path = SILVER_DIR / path
        exists = full_path.exists()
        emoji = "[OK]" if exists else "[FAIL]"
        logger.info(f"  {emoji} {path}: {desc}")
    
    logger.info("")
    logger.info("=" * 70)
    
    # Return success if all succeeded or skipped
    success = all('SUCCESS' in str(s) or 'SKIPPED' in str(s) or 'NOT AVAILABLE' in str(s) 
                  for s in results.values())
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    skip_clean = '--skip-clean' in sys.argv
    skip_metadata = '--skip-metadata' in sys.argv
    skip_news = '--skip-news' in sys.argv
    skip_economic = '--skip-economic' in sys.argv
    skip_join = '--skip-join' in sys.argv
    
    exit(run_all_processors(
        skip_clean=skip_clean,
        skip_metadata=skip_metadata,
        skip_news=skip_news,
        skip_economic=skip_economic,
        skip_join=skip_join
    ))
