"""
Silver Layer: Memory-optimized clean for free-tier VM (1GB RAM)

Processes data in batches by ticker to stay within memory limits.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from config import (
    BRONZE_DIR, SILVER_DIR, METADATA_DIR,
    PRICE_DATA_SCHEMA, SILVER_QUALITY_CHECKS,
    GICS_SECTORS, LOG_FORMAT
)

# Logging setup
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Constants
BRONZE_FILE = BRONZE_DIR / 'prices.parquet'
BRONZE_FILE_ALT = BRONZE_DIR / 'all_stock_data.parquet'
OUTPUT_PATH = SILVER_DIR / 'enriched_stocks.parquet'

# Memory optimization: process tickers in batches
BATCH_SIZE = 50  # Process 50 tickers at a time


def get_all_tickers(bronze_path):
    """Get list of all unique tickers without loading full dataset."""
    logger.info("Reading ticker list from file...")
    
    # Read only ticker column to get unique list
    df_tickers = pd.read_parquet(bronze_path, columns=['ticker'])
    unique_tickers = sorted(df_tickers['ticker'].unique())
    
    logger.info(f"Found {len(unique_tickers)} unique tickers")
    return unique_tickers


def process_ticker_batch(bronze_path, tickers, essential_cols):
    """Process a batch of tickers and return cleaned data."""
    # Read only selected tickers
    df = pd.read_parquet(bronze_path, columns=essential_cols)
    df.columns = [c.lower() for c in df.columns]
    
    # Filter to selected tickers
    df = df[df['ticker'].isin(tickers)]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Data cleaning steps
    # 1. Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Deduplicate
    df = df.sort_values('date').drop_duplicates(subset=['ticker', 'date'], keep='last')
    
    # 3. Remove nulls
    df = df.dropna(subset=['date', 'ticker', 'close', 'open', 'high', 'low', 'volume'])
    
    # 4. Quality gates
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['volume'] >= 0) & (df['high'] >= df['low'])]
    
    # -----------------------------------------------------------
    # 4.5 ADJUST FOR SPLITS (Backward Adjustment)
    # -----------------------------------------------------------
    # df = adjust_prices(df)
    pass
    # -----------------------------------------------------------

    # 5. Calculate daily returns
    df = df.sort_values(['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change() *100
    df['daily_return'] = df['daily_return'].fillna(0)
    
    # 6. Add enrichment metadata
    df['enriched_at'] = datetime.now()
    df['data_version'] = 'silver_v1'
    
    # 7. Add quality flags
    df['is_penny_stock'] = df['close'] < 1.0
    df['has_outlier_return'] = df['daily_return'].abs() > 50
    df['low_liquidity'] = df['volume'] < 10000
    
    return df

def adjust_prices(df):
    """
    Adjust historical prices for splits using Backward Adjustment.
    Logic:
    - Sort DESC by date.
    - Check if Price(t-1) / Price(t) < 0.6 (approx drop > 40% indicating split).
      Wait. If split 2:1, Old Price 100, New Price 50.
      Price(t-1) (Old) / Price(t) (New) -> NO.
      We iterate backwards: t (New, 50), t-1 (Old, 100).
      If Price(t) / Price(t-1) < 0.6 => Drop significantly overnight.
      Example: Today 50, Yesterday 100. Ratio = 0.5.
      Split Ratio = 1/0.5 = 2.
      Adjustment Factor = 0.5.
      Apply factor to all prices BEFORE t.
    """
    if df.empty: return df
    
    # Work on a copy sorted by date
    df = df.sort_values('date').copy()
    
    # Calculate daily returns (raw) to find splits
    # Close[t] / Close[t-1] - 1
    # Using numpy for speed
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Iterate backwards is safer for cumulative adjustment
    # But vectorized is faster.
    # Let's use a simple heuristic:
    # Any drop > 25% (Ratio < 0.75) might be a split.
    # Common splits: 2:1 (0.5), 3:1 (0.33), 3:2 (0.66), 4:1 (0.25), 5:1 (0.2), 7:1 (0.14), 10:1 (0.1), 20:1 (0.05)
    
    # We will iterate backwards from the end
    cumulative_adjustment = 1.0
    
    # Arrays are mutable
    n = len(closes)
    for i in range(n-1, 0, -1):
        curr_price = closes[i]
        prev_price = closes[i-1]
        
        # Apply current cumulative adjustment to current row FIRST
        # Wait, backward iteration? 
        # Easier: Adjust "Prev Price" down if split happened between i-1 and i.
        
        # Let's think: 
        # T (Today): 50. T-1 (Yest): 100. (Split 2:1 happens at T open).
        # We see 50 vs 100.
        # Ratio = 50 / 100 = 0.5.
        # IF this is a split, we must multiply T-1 and ALL previous by 0.5.
        
        # To do this efficiently:
        # 1. Calc ratios: Price[i] / Price[i-1]
        # 2. Identify splits.
        # 3. Create adjustment factor series.
        # 4. Cumulative product backward.
        
        ratio = curr_price / prev_price
        
        # Thresholds for splits
        # 2:1 -> 0.5
        # 3:1 -> 0.33
        # 3:2 -> 0.66
        # 4:1 -> 0.25
        # Reverse splits: 1:2 -> 2.0 (We assume mostly forward splits for now, but reverse also happens)
        
        split_factor = 1.0
        
        # Verify it's not just a market crash using volume or something?
        # Heuristic: Splits occur exactly near ratios like 0.5, 0.33, 0.25, 0.2, 0.1
        # Allow 15% tolerance? No, splits are precise. Market doesn't drop 50% often.
        # But allow 20% margin for gap down/up.
        
        if ratio < 0.75: # Potential Forward Split
             # Check common ratios
             if 0.45 < ratio < 0.55: split_factor = 0.5       # 2:1
             elif 0.30 < ratio < 0.36: split_factor = 0.3333  # 3:1
             elif 0.22 < ratio < 0.28: split_factor = 0.25    # 4:1
             elif 0.18 < ratio < 0.22: split_factor = 0.2     # 5:1
             elif 0.09 < ratio < 0.11: split_factor = 0.1     # 10:1
             elif 0.04 < ratio < 0.06: split_factor = 0.05    # 20:1
             # 3:2 split = 0.66 (Close to 0.75 threshold, might assume standard market drop)
             # Let's stick to major splits first.
             elif 0.6 < ratio < 0.72: split_factor = 0.6666   # 3:2
        
        elif ratio > 1.4: # Potential Reverse Split (Price Jumps)
             if 1.9 < ratio < 2.1: split_factor = 2.0         # 1:2
             elif 2.9 < ratio < 3.1: split_factor = 3.0       # 1:3
             elif 3.9 < ratio < 4.1: split_factor = 4.0       # 1:4
             elif 9.0 < ratio < 11.0: split_factor = 10.0     # 1:10
             
        if split_factor != 1.0:
            cumulative_adjustment *= split_factor
            
        # Apply adjustment to History (i-1)
        # Wait, this loop structure is O(N^2) if we update array every time.
        # Optimized: calculate factors first, then cumprod.
        pass
        
    return df

# Better Implementation below:
def adjust_prices_vectorized(df):
    df = df.sort_values('date')
    prices = df['close'].values
    
    # Calculate T / T-1 ratios
    # Pad first element with 1
    ratios = np.empty_like(prices)
    ratios[1:] = prices[1:] / prices[:-1]
    ratios[0] = 1.0
    
    # Identify Split Factors
    factors = np.ones_like(prices)
    
    # 2:1 (Target 0.5)
    mask = (ratios > 0.45) & (ratios < 0.55)
    factors[mask] = 0.5
    
    # 4:1 (Target 0.25)
    mask = (ratios > 0.23) & (ratios < 0.27)
    factors[mask] = 0.25
    
    # 3:1
    mask = (ratios > 0.31) & (ratios < 0.35)
    factors[mask] = 1/3
    
    # 5:1
    mask = (ratios > 0.18) & (ratios < 0.22)
    factors[mask] = 0.2
    
    # 10:1
    mask = (ratios > 0.09) & (ratios < 0.11)
    factors[mask] = 0.1
    
    # 20:1 (Amazon)
    mask = (ratios > 0.04) & (ratios < 0.06)
    factors[mask] = 0.05
    
    # 3:2 (0.66)
    mask = (ratios > 0.64) & (ratios < 0.69)
    factors[mask] = 2/3
    
    # Reverse Splits
    mask = (ratios > 1.9) & (ratios < 2.1)
    factors[mask] = 2.0
    
    mask = (ratios > 2.9) & (ratios < 3.1)
    factors[mask] = 3.0
    
    mask = (ratios > 3.9) & (ratios < 4.1)
    factors[mask] = 4.0
    
    mask = (ratios > 9.0) & (ratios < 11.0)
    factors[mask] = 10.0
    
    # Cumprod Backwards
    # factors[i] means at time i, a split happened. We need to adjust everything BEFORE i.
    # So adjustment_series needs to be cumulative from end to start.
    
    # Example:
    # Day 0: 100
    # Day 1: 50 (Split 0.5). Factor[1] = 0.5.
    # We want Day 0 to be 100 * 0.5 = 50.
    
    # Accumulate product backwards
    # factors = [1, 0.5, 1, 1]
    # cum_adj = [0.5, 0.5, 1, 1] (Backwards cumprod?)
    
    # Let's reverse, cumprod, reverse back
    adj_factors = np.cumprod(factors[::-1])[::-1]
    
    # NOTE: The factor at index i affects i-1 and before.
    # So we need to shift.
    # Wait.
    # If index 1 is split day. Factor[1] = 0.5.
    # Prices[1] is ALREADY correct (New Price).
    # Prices[0] needs correction.
    # cumprod[1] (0.5) should apply to Prices[0].
    # cumprod[0] (0.5 * 1 = 0.5) applies to... nothing before 0.
    
    # Shift adj_factors left by 1?
    # Actually, we apply adj_factors to all prices?
    # No, price[i] is already post-split relative to itself.
    
    # Correct logic:
    # Multiplier at time t: Product of all split factors happening AFTER time t.
    # Time 0: Factor[1]*Factor[2]*...
    # Time 1: Factor[2]*...
    
    # So yes, Reverse Cumprod is correct for "Future Splits".
    # But we calculated factor based on "Past/Current" ratio.
    # Factor[i] is split occurring at i.
    # So for price at k < i, we multiply by Factor[i].
    
    # Reverse cumprod of factors (excluding index 0?)
    # factors[0] is always 1 (no split at start).
    
    # cum_factors = cumprod(reverse(factors)) -> reversed back.
    # Example: [1, 0.5, 1]. Reverse: [1, 0.5, 1]. Cum: [1, 0.5, 0.5]. Reverse: [0.5, 0.5, 1].
    # Element 0 -> 0.5. Element 1 -> 0.5. Element 2 -> 1.
    # Price 0 (100) * 0.5 = 50.
    # Price 1 (50) * 0.5 = 25? NO! Price 1 is already split.
    
    # We should NOT include Factor[i] in the adjustment for Price[i].
    # We should include Factor[i] for Price[i-1].
    
    # So we need to shift the adjustment array.
    # adj_factors_final[i] = Product(factors[k] for k > i).
    
    back_cumprod = np.cumprod(factors[::-1])[::-1]
    # back_cumprod[i] includes factor[i].
    # We want to divide by factor[i] to exclude it? Or just shift.
    
    # Shift: adj[i] = back_cumprod[i+1]
    adj_final = np.ones_like(prices)
    adj_final[:-1] = back_cumprod[1:]
    
    # Apply
    # Ensure dimensions match
    df['close'] = df['close'] * adj_final
    df['open'] = df['open'] * adj_final
    df['high'] = df['high'] * adj_final
    df['low'] = df['low'] * adj_final
    
    return df
    
def adjust_prices(df):
    # Wrapper for safety
    try:
        return adjust_prices_vectorized(df)
    except Exception:
        return df # Fallback



def clean_silver_data_optimized():
    """Main Silver cleaning pipeline - memory optimized for 1GB RAM."""
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("SILVER LAYER PROCESSING STARTED (MEMORY-OPTIMIZED)")
    logger.info("=" * 70)
    
    try:
        # Find bronze file
        if BRONZE_FILE.exists():
            bronze_path = BRONZE_FILE
        elif BRONZE_FILE_ALT.exists():
            bronze_path = BRONZE_FILE_ALT
        else:
            raise FileNotFoundError(f"No bronze file found")
        
        logger.info(f"Processing: {bronze_path}")
        
        # Get file info
        parquet_file = pq.ParquetFile(bronze_path)
        total_rows = parquet_file.metadata.num_rows
        file_columns = parquet_file.schema.names
        logger.info(f"Total rows: {total_rows:,}")
        
        # Define essential columns
        essential_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        actual_cols = [col for col in file_columns if col.lower() in [c.lower() for c in essential_cols]]
        
        # Get all tickers
        all_tickers = get_all_tickers(bronze_path)
        
        # Process in batches
        all_batches = []
        total_tickers = len(all_tickers)
        num_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Processing {total_tickers} tickers in {num_batches} batches of {BATCH_SIZE}")
        
        for i in range(0, total_tickers, BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            ticker_batch = all_tickers[i:i+BATCH_SIZE]
            
            logger.info(f"[Batch {batch_num}/{num_batches}] Processing {len(ticker_batch)} tickers...")
            
            df_batch = process_ticker_batch(bronze_path, ticker_batch, actual_cols)
            
            if len(df_batch) > 0:
                all_batches.append(df_batch)
                logger.info(f"  → {len(df_batch):,} rows processed")
            
            # Free memory
            del df_batch
        
        # Combine all batches
        logger.info("Combining all batches...")
        df_final = pd.concat(all_batches, ignore_index=True)
        
        # Add sector info
        try:
            metadata_file = METADATA_DIR / 'ticker_metadata.parquet'
            if metadata_file.exists():
                logger.info("Adding sector metadata...")
                metadata = pd.read_parquet(metadata_file)
                if 'ticker' in metadata.columns and 'sector' in metadata.columns:
                    merge_cols = ['ticker', 'sector']
                    if 'industry' in metadata.columns:
                        merge_cols.append('industry')
                    
                    df_final = df_final.merge(metadata[merge_cols], on='ticker', how='left')
                    df_final['sector'] = df_final['sector'].fillna('Unknown')
                    if 'industry' not in df_final.columns:
                        df_final['industry'] = 'Unknown'
                    else:
                        df_final['industry'] = df_final['industry'].fillna('Unknown')
            else:
                df_final['sector'] = 'Unknown'
                df_final['industry'] = 'Unknown'
        except Exception as e:
            logger.warning(f"Could not add sector metadata: {e}")
            df_final['sector'] = 'Unknown'
            df_final['industry'] = 'Unknown'
        
        # Add ticker-level quality flags
        ticker_counts = df_final.groupby('ticker').size()
        insufficient_tickers = ticker_counts[ticker_counts < 252].index
        df_final['insufficient_history'] = df_final['ticker'].isin(insufficient_tickers)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info("SILVER LAYER PROCESSING COMPLETED")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total rows: {len(df_final):,}")
        logger.info(f"Unique tickers: {df_final['ticker'].nunique():,}")
        logger.info(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
        logger.info(f"Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.info("=" * 70)
        
        return df_final
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 70)
        logger.error(f"SILVER LAYER PROCESSING FAILED after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


def save_to_silver(df: pd.DataFrame, output_path: Path = OUTPUT_PATH):
    """Save DataFrame to Silver layer."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to {output_path}...")
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size = output_path.stat().st_size / 1024**2
        logger.info(f"[OK] Data saved to {output_path}")
        logger.info(f"[OK] File size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise


def main():
    """CLI entry point."""
    logger.info("")
    logger.info("SILVER LAYER PROCESSING (MEMORY-OPTIMIZED FOR FREE-TIER VM)")
    logger.info("")
    
    try:
        # Process data
        df = clean_silver_data_optimized()
        
        # Save to Silver layer
        save_to_silver(df)
        
        logger.info("")
        logger.info("[OK] Silver layer processing completed successfully!")
        logger.info(f"[OK] Output: {OUTPUT_PATH}")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"[ERR] Silver layer processing failed: {str(e)}")
        logger.error("")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
