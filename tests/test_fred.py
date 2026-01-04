"""
Test FRED API connection and fetch sample data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os

# Set API key
FRED_API_KEY = "c9038fbcb0de4b93d89dedebfbd3937a"
os.environ['FRED_API_KEY'] = FRED_API_KEY

from bronze.collectors.economic.fred_collector import FREDCollector

print("🧪 Testing FRED API Connection...")
print(f"   API Key: {FRED_API_KEY[:10]}..." if len(FRED_API_KEY) > 10 else "   API Key: NOT SET")

try:
    collector = FREDCollector(api_key=FRED_API_KEY)
    
    # Test with single series first
    print("\n📊 Fetching GDP data (test)...")
    gdp = collector.fetch_series('GDP', start_date='2020-01-01')
    
    if not gdp.empty:
        print(f"   ✅ SUCCESS!")
        print(f"   Records: {len(gdp)}")
        print(f"   Date range: {gdp['date'].min()} to {gdp['date'].max()}")
        print(f"\n   Sample data:")
        print(gdp.head())
        
        print("\n🎉 FRED API is working!")
        print("   Ready to fetch all 17 indicators from 1962")
    else:
        print("   ❌ No data returned")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n   Possible issues:")
    print("   1. Invalid API key")
    print("   2. Network connection")
    print("   3. FRED API down")
