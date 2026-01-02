"""
ML pipeline orchestrator - runs all models sequentially
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import argparse
from datetime import datetime

from config import LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_all_models(test_mode: bool = False):
    """Run all ML models in sequence"""
    
    start_time = datetime.now()
    results = {}
    
    logger.info("=" * 70)
    logger.info("ML MODELS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode: {'TEST' if test_mode else 'PRODUCTION'}")
    
    # 1. Feature Analysis (Random Forest)
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 1/3: FEATURE ANALYSIS")
    logger.info("=" * 50)
    
    try:
        from models.random_forest_selector import run_feature_analysis
        result = run_feature_analysis(test_mode=test_mode)
        results['feature_analysis'] = 'SUCCESS'
        logger.info("[OK] Feature analysis completed")
    except Exception as e:
        results['feature_analysis'] = f'FAILED: {str(e)}'
        logger.error(f"[FAIL] Feature analysis: {e}")
    
    # 2. Momentum Strategy
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 2/3: MOMENTUM STRATEGY")
    logger.info("=" * 50)
    
    try:
        from models.momentum_strategy import run_momentum_strategy
        result = run_momentum_strategy(test_mode=test_mode)
        results['momentum'] = f"SUCCESS: {result.get('portfolio_size', 0)} stocks"
        logger.info("[OK] Momentum strategy completed")
    except Exception as e:
        results['momentum'] = f'FAILED: {str(e)}'
        logger.error(f"[FAIL] Momentum: {e}")
    
    # 3. Causal Analysis
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 3/3: CAUSAL ANALYSIS")
    logger.info("=" * 50)
    
    try:
        from models.causal_model import load_unified_data
        df = load_unified_data()
        if len(df) > 0:
            results['causal'] = f"SUCCESS: {len(df):,} days analyzed"
            logger.info(f"[OK] Causal analysis: {len(df):,} days")
        else:
            results['causal'] = 'FAILED: No data'
    except Exception as e:
        results['causal'] = f'FAILED: {str(e)}'
        logger.error(f"[FAIL] Causal: {e}")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("ML PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("")
    logger.info("Results:")
    
    for step, result in results.items():
        status = "✓" if "FAIL" not in str(result) else "✗"
        logger.info(f"  {status} {step}: {result}")
    
    logger.info("")
    logger.info("Output files:")
    logger.info("  - data/gold/feature_importance_lakehouse/")
    logger.info("  - data/gold/momentum_portfolio_lakehouse/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run ML models pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode with subset of data')
    args = parser.parse_args()
    
    run_all_models(test_mode=args.test)


if __name__ == "__main__":
    main()
