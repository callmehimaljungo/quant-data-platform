
import sys
from pathlib import Path
import logging

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import BRONZE_DIR
from utils.realtime_sync import fetch_latest_prices, save_incremental

logging.basicConfig(level=logging.INFO)

def test_fetch_and_save():
    print("Testing fetch_latest_prices...")
    # Test with known good tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    df = fetch_latest_prices(tickers)
    
    if df.empty:
        print("[FAIL] Returned empty DataFrame")
        return
        
    print(f"[OK] Fetched {len(df)} rows")
    print(df.head())
    
    print("\nTesting save_incremental...")
    path = save_incremental(df, 'prices')
    
    print(f"Saved to: {path}")
    
    # Check if raw file exists
    raw_path = BRONZE_DIR / 'prices_raw_view'
    if raw_path.exists():
        print(f"[OK] Raw directory exists: {raw_path}")
        files = list(raw_path.glob("*.json"))
        print(f"Found {len(files)} JSON files: {[f.name for f in files]}")
    else:
        print(f"[FAIL] Raw directory NOT found: {raw_path}")

if __name__ == "__main__":
    test_fetch_and_save()
