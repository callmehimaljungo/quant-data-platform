"""
Master Pipeline Script
Run Bronze → Silver → Gold → Dashboard in one command

Usage:
  python run_pipeline.py --mode batch      # Full recompute (daily)
  python run_pipeline.py --mode stream     # Realtime update only
  python run_pipeline.py --mode batch --skip-bronze  # Skip Bronze if data exists
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd

from config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, DATA_DIR
from utils.r2_sync import (
    is_r2_configured, 
    download_bronze_from_r2, upload_bronze_to_r2,
    download_silver_from_r2, upload_silver_to_r2,
    download_gold_from_r2, upload_gold_to_r2
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_batch_mode(skip_bronze: bool = False, skip_silver: bool = False, sync_r2: bool = True):
    """
    Full pipeline recomputation (daily job)
    - Pulls data from R2 before each step
    - Processes data locally
    - Pushes results to R2 after each step
    """
    start_time = datetime.now()
    
    print("=" * 80)
    print("🚀 BATCH MODE: FULL PIPELINE RECOMPUTATION")
    print("=" * 80)
    print(f"Start time: {start_time}")
    print(f"R2 Sync: {'ENABLED' if sync_r2 and is_r2_configured() else 'DISABLED'}")
    print()
    
    results = {
        'r2_pull': None,
        'bronze': None,
        'silver': None,
        'gold': None,
        'cache': None,
        'r2_push': None
    }
    
    # Check R2 configuration
    r2_enabled = sync_r2 and is_r2_configured()
    
    # Step 0: Pull Bronze from R2 (before processing)
    if r2_enabled and not skip_bronze:
        print("-" * 80)
        print("☁️ Step 0: PULL BRONZE FROM R2")
        print("-" * 80)
        try:
            download_bronze_from_r2(BRONZE_DIR)
            results['r2_pull'] = 'success'
        except Exception as e:
            print(f"   ⚠️ R2 pull failed: {e}")
            results['r2_pull'] = 'failed'
        print()
    
    # Step 1: Bronze (skip if data exists)
    if not skip_bronze:
        print("-" * 80)
        print("📦 Step 1: BRONZE LAYER (Data Ingestion)")
        print("-" * 80)
        try:
            # Check if Bronze data exists
            prices_dir = BRONZE_DIR / 'prices_partitioned'
            econ_file = BRONZE_DIR / 'economic_lakehouse' / 'economic_indicators.parquet'
            
            if prices_dir.exists() and econ_file.exists():
                print("   ✅ Bronze data already exists, skipping ingestion")
                results['bronze'] = 'exists'
            else:
                print("   ⚠️ Bronze data missing, please run data collection first")
                results['bronze'] = 'missing'
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results['bronze'] = 'error'
        
        # Push Bronze to R2 after ingestion
        if r2_enabled and results['bronze'] in ['exists', 'success']:
            print("   ☁️ Pushing Bronze to R2...")
            upload_bronze_to_r2(BRONZE_DIR)
    else:
        print("   ⏭️ Skipping Bronze (--skip-bronze)")
        results['bronze'] = 'skipped'
    
    print()
    
    # Step 2: Silver (Data Cleaning)
    if not skip_silver:
        print("-" * 80)
        print("🥈 Step 2: SILVER LAYER (Data Cleaning)")
        print("-" * 80)
        
        # Pull Silver from R2 first (for incremental updates)
        if r2_enabled:
            print("   ☁️ Pulling existing Silver from R2...")
            download_silver_from_r2(SILVER_DIR)
        
        try:
            from silver.run_all_processors import main as run_silver
            run_silver()
            results['silver'] = 'success'
            print("   ✅ Silver processing complete")
            
            # Push Silver to R2 after processing
            if r2_enabled:
                print("   ☁️ Pushing Silver to R2...")
                upload_silver_to_r2(SILVER_DIR)
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results['silver'] = 'error'
    else:
        print("   ⏭️ Skipping Silver (--skip-silver)")
        results['silver'] = 'skipped'
    
    print()
    
    # Step 3: Gold (Strategy Computation)
    print("-" * 80)
    print("🥇 Step 3: GOLD LAYER (Strategy Computation)")
    print("-" * 80)
    
    # Pull Gold from R2 first (for incremental updates)
    if r2_enabled:
        print("   ☁️ Pulling existing Gold from R2...")
        download_gold_from_r2(GOLD_DIR)
    
    try:
        from gold.run_all_strategies import run_all_strategies
        from gold.cache_manager import get_cache_manager
        
        # Run all strategies
        strategy_results = run_all_strategies()
        
        # Cache results
        cache = get_cache_manager()
        
        for name, result in strategy_results.items():
            if result is not None and isinstance(result, pd.DataFrame):
                cache.save_portfolio_weights(result, name)
        
        results['gold'] = 'success'
        results['cache'] = 'updated'
        print("   ✅ Gold processing complete")
        print("   ✅ Cache updated")
        
        # Push Gold to R2 after processing
        if r2_enabled:
            print("   ☁️ Pushing Gold to R2...")
            upload_gold_to_r2(GOLD_DIR)
            results['r2_push'] = 'success'
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['gold'] = 'error'
    
    print()
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    print("=" * 80)
    print("📊 BATCH MODE COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Results:")
    for step, status in results.items():
        icon = '✅' if status in ['success', 'exists', 'updated', 'skipped'] else '❌'
        print(f"  {icon} {step}: {status}")
    
    print()
    print("Next steps:")
    print("  - Run: streamlit run dashboard/app.py")
    print("  - Or: python run_pipeline.py --mode stream")
    
    return results


def run_stream_mode(target_date: str = None):
    """
    Lightweight streaming update
    - Only processes today's data
    - Uses cached weights
    - Does NOT recompute heavy calculations
    """
    start_time = datetime.now()
    
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 80)
    print("⚡ STREAM MODE: LIGHTWEIGHT UPDATE")
    print("=" * 80)
    print(f"Target date: {target_date}")
    print(f"Start time: {start_time}")
    print()
    
    # Step 1: Check cache validity
    print("-" * 80)
    print("🔍 Step 1: CHECK CACHE")
    print("-" * 80)
    
    from gold.cache_manager import get_cache_manager
    cache = get_cache_manager()
    
    if not cache.is_cache_valid(max_age_days=7):
        print("   ⚠️ Cache is stale or missing!")
        print("   Please run: python run_pipeline.py --mode batch")
        return {'status': 'cache_invalid'}
    
    last_batch = cache.get_last_batch_date()
    print(f"   ✅ Cache valid (last batch: {last_batch})")
    
    print()
    
    # Step 2: Load today's data only
    print("-" * 80)
    print("📥 Step 2: LOAD TODAY'S DATA")
    print("-" * 80)
    
    try:
        # Load only target date partition
        price_partition = BRONZE_DIR / 'prices_partitioned' / f'date={target_date}'
        
        if price_partition.exists():
            today_prices = pd.read_parquet(price_partition)
            print(f"   ✅ Loaded {len(today_prices)} price records for {target_date}")
        else:
            print(f"   ⚠️ No price data for {target_date}")
            today_prices = pd.DataFrame()
            
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        today_prices = pd.DataFrame()
    
    print()
    
    # Step 3: Calculate today's PnL using cached weights
    print("-" * 80)
    print("📈 Step 3: CALCULATE TODAY'S PnL")
    print("-" * 80)
    
    pnl_results = {}
    
    strategies = ['low_beta_quality', 'sector_rotation', 'sentiment_allocation']
    
    for strategy in strategies:
        try:
            weights = cache.load_portfolio_weights(strategy)
            
            if weights.empty or today_prices.empty:
                print(f"   ⚠️ {strategy}: No data")
                continue
            
            # Simple PnL calculation (placeholder)
            # In production: weights * returns
            pnl = 0.0  # Placeholder
            pnl_results[strategy] = pnl
            print(f"   ✅ {strategy}: PnL = {pnl:.2%}")
            
        except Exception as e:
            print(f"   ❌ {strategy}: Error - {e}")
    
    print()
    
    # Step 4: Append to history
    print("-" * 80)
    print("💾 Step 4: UPDATE HISTORY")
    print("-" * 80)
    
    if pnl_results:
        cache.append_daily_performance(target_date, pnl_results)
        print(f"   ✅ Appended {target_date} to performance history")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    print()
    print("=" * 80)
    print("⚡ STREAM MODE COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Memory: Minimal (did not load full dataset)")
    
    return {'status': 'success', 'pnl': pnl_results, 'duration': duration}


def main():
    parser = argparse.ArgumentParser(description='Master Pipeline Script')
    parser.add_argument('--mode', '-m', choices=['batch', 'stream'], 
                        default='batch', help='Pipeline mode')
    parser.add_argument('--date', '-d', type=str, default=None,
                        help='Target date for stream mode (YYYY-MM-DD)')
    parser.add_argument('--skip-bronze', action='store_true',
                        help='Skip Bronze layer (use existing data)')
    parser.add_argument('--skip-silver', action='store_true',
                        help='Skip Silver layer')
    parser.add_argument('--no-r2', action='store_true',
                        help='Disable R2 sync (run locally only)')
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        return run_batch_mode(
            skip_bronze=args.skip_bronze,
            skip_silver=args.skip_silver,
            sync_r2=not args.no_r2
        )
    else:
        return run_stream_mode(target_date=args.date)


if __name__ == "__main__":
    main()
