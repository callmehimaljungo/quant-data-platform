"""
Gold Layer: Run All Portfolio Strategies
Orchestrates all 3 portfolio allocation strategies

3 Strategies:
1. Low-Beta Quality: Cổ phiếu ít rủi ro + chất lượng cao
2. Sector Rotation: Xoay vòng ngành theo chu kỳ kinh tế
3. Sentiment-Adjusted: Điều chỉnh theo tin tức + VIX

Usage:
    python gold/run_all_strategies.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime
import pandas as pd

from config import GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_all_strategies(start_date: str = None, end_date: str = None, quick_update: bool = False):
    """
    Run all 3 portfolio strategies and generate summary.
    
    Args:
        start_date: Optional start date for incremental processing (YYYY-MM-DD)
        end_date: Optional end date for incremental processing (YYYY-MM-DD)
        quick_update: If True, skip heavy backtesting (for realtime updates)
    """
    
    start_time = datetime.now()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("GOLD LAYER: RUNNING ALL PORTFOLIO STRATEGIES")
    if start_date or end_date:
        logger.info(f"  Mode: INCREMENTAL ({start_date} to {end_date})")
    if quick_update:
        logger.info("  Mode: QUICK UPDATE (skip backtesting)")
    logger.info("=" * 70)
    logger.info("")
    
    results = {}
    
    # Strategy 1: Low-Beta Quality
    logger.info("-" * 70)
    logger.info("STRATEGY 1: LOW-BETA QUALITY")
    logger.info("-" * 70)
    try:
        from gold.low_beta_quality import run_low_beta_quality
        results['low_beta_quality'] = run_low_beta_quality()
        logger.info("[OK] Strategy 1 completed")
    except Exception as e:
        logger.error(f"[ERROR] Strategy 1 failed: {e}")
        results['low_beta_quality'] = None
    
    logger.info("")
    
    # Strategy 2: Sector Rotation
    logger.info("-" * 70)
    logger.info("STRATEGY 2: SECTOR ROTATION")
    logger.info("-" * 70)
    try:
        from gold.sector_rotation import run_sector_rotation
        results['sector_rotation'] = run_sector_rotation()
        logger.info("[OK] Strategy 2 completed")
    except Exception as e:
        logger.error(f"[ERROR] Strategy 2 failed: {e}")
        results['sector_rotation'] = None
    
    logger.info("")
    
    # Strategy 3: Sentiment Allocation
    logger.info("-" * 70)
    logger.info("STRATEGY 3: SENTIMENT-ADJUSTED ALLOCATION")
    logger.info("-" * 70)
    try:
        from gold.sentiment_allocation import run_sentiment_allocation
        results['sentiment_allocation'] = run_sentiment_allocation()
        logger.info("[OK] Strategy 3 completed")
    except Exception as e:
        logger.error(f"[ERROR] Strategy 3 failed: {e}")
        print("\n--- STRATEGY SUMMARY ---")
        print(df_summary.to_string(index=False))
    
    # Check for overlapping stocks
    all_tickers = set()
    for strategy_name, df in results.items():
        if len(df) > 0:
            tickers = set(df['ticker'].tolist())
            overlap = all_tickers.intersection(tickers)
            if overlap:
                logger.info(f"  Overlap with previous strategies: {len(overlap)} stocks")
            all_tickers.update(tickers)
    
    logger.info(f"\nTotal unique stocks across all strategies: {len(all_tickers)}")

    # -------------------------------------------------------------------------
    # Update Cache for Dashboard/App (Automatic Sync)
    # -------------------------------------------------------------------------
    logger.info("\nUpdating Dashboard Cache from Lakehouse Results...")
    import shutil
    cache_dir = GOLD_DIR / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for strategy in ['low_beta_quality', 'sector_rotation', 'sentiment_allocation']:
        lakehouse_dir = GOLD_DIR / f'{strategy}_lakehouse'
        if lakehouse_dir.exists():
            # Get latest file
            files = sorted(lakehouse_dir.glob('*.parquet'), key=lambda x: x.stat().st_mtime, reverse=True)
            if files:
                dest = cache_dir / f'{strategy}_weights.parquet'
                try:
                    shutil.copy2(files[0], dest)
                    logger.info(f"  [OK] Updated cache for {strategy}")
                except Exception as e:
                    logger.error(f"  [FAIL] Failed to update cache for {strategy}: {e}")
            else:
                 logger.warning(f"  [WARN] No files found in {lakehouse_dir}")
        else:
            logger.warning(f"  [WARN] Lakehouse dir not found: {lakehouse_dir}")

    
    # Duration
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[OK] ALL STRATEGIES COMPLETED")
    logger.info(f"Total duration: {duration:.1f} seconds")
    logger.info("=" * 70)
    
    # Output paths
    logger.info("\nOutput files:")
    logger.info(f"  - {GOLD_DIR / 'low_beta_quality_lakehouse'}")
    logger.info(f"  - {GOLD_DIR / 'sector_rotation_lakehouse'}")
    logger.info(f"  - {GOLD_DIR / 'sentiment_allocation_lakehouse'}")
    
    return results


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description='Gold Layer - Run All Strategies')
    parser.add_argument('--quick-update', '-q', action='store_true',
                        help='Quick update mode: skip heavy backtesting, just update weights')
    parser.add_argument('--start-date', '-s', type=str, default=None,
                        help='Start date for incremental processing (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e', type=str, default=None,
                        help='End date for incremental processing (YYYY-MM-DD)')
    args = parser.parse_args()
    
    try:
        if args.quick_update:
            logger.info("")
            logger.info("=" * 70)
            logger.info("GOLD LAYER: QUICK UPDATE MODE")
            logger.info("=" * 70)
            logger.info("  Skipping heavy backtesting, updating portfolio weights only")
            logger.info("")
        
        run_all_strategies(
            start_date=args.start_date,
            end_date=args.end_date,
            quick_update=args.quick_update
        )
        
        if not args.quick_update:
            logger.info("")
            logger.info("=" * 70)
            logger.info("NEXT STEPS:")
            logger.info("  1. Review portfolios in data/gold/")
            logger.info("  2. Run backtest: python backtest/evaluate.py")
            logger.info("  3. Launch dashboard: streamlit run dashboard/app.py")
            logger.info("=" * 70)
        
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

