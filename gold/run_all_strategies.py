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


def run_all_strategies():
    """Run all 3 portfolio strategies and generate summary"""
    
    start_time = datetime.now()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("GOLD LAYER: RUNNING ALL PORTFOLIO STRATEGIES")
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
        results['low_beta_quality'] = pd.DataFrame()
    
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
        results['sector_rotation'] = pd.DataFrame()
    
    logger.info("")
    
    # Strategy 3: Sentiment-Adjusted
    logger.info("-" * 70)
    logger.info("STRATEGY 3: SENTIMENT-ADJUSTED")
    logger.info("-" * 70)
    try:
        from gold.sentiment_allocation import run_sentiment_allocation
        results['sentiment_adjusted'] = run_sentiment_allocation()
        logger.info("[OK] Strategy 3 completed")
    except Exception as e:
        logger.error(f"[ERROR] Strategy 3 failed: {e}")
        results['sentiment_adjusted'] = pd.DataFrame()
    
    # Generate Combined Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMBINED PORTFOLIO SUMMARY")
    logger.info("=" * 70)
    
    summary_rows = []
    
    for strategy_name, df in results.items():
        if len(df) > 0:
            summary_rows.append({
                'strategy': strategy_name,
                'num_stocks': len(df),
                'sectors': df['sector'].nunique() if 'sector' in df.columns else 0,
                'total_weight': df['weight'].sum() if 'weight' in df.columns else 0,
            })
    
    df_summary = pd.DataFrame(summary_rows)
    
    if len(df_summary) > 0:
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
    try:
        run_all_strategies()
        
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
