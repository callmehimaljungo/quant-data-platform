"""
Phase 1: Economic Data Collection
Fetch complete FRED data from 1962 to present
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys
exec(open('config_api_keys.py').read())

from bronze.collectors.economic.fred_collector import FREDCollector
from config import BRONZE_DIR
from datetime import datetime

print("=" * 80)
print("📊 PHASE 1: ECONOMIC DATA COLLECTION (1962-2026)")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print()

# Initialize collector
print("Initializing FRED collector...")
fred = FREDCollector()

# Fetch all indicators from 1962
print("\nFetching 17 economic indicators from 1962-01-01...")
print("This may take 2-3 hours due to API rate limits...")
print()

start_time = datetime.now()

try:
    econ_data = fred.fetch_all(start_date='1962-01-01')
    
    if econ_data:
        # Save to Bronze
        output_dir = BRONZE_DIR / 'economic_lakehouse'
        output_file = fred.save_to_bronze(econ_data, output_dir)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print()
        print("=" * 80)
        print("✅ PHASE 1 COMPLETE!")
        print("=" * 80)
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Output: {output_file}")
        print()
        print("Next steps:")
        print("  - Verify data quality")
        print("  - Start Phase 2: Recent News collection")
        print("=" * 80)
    else:
        print("❌ No data fetched!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
