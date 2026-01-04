"""
API Keys Configuration
Set environment variables for all collectors
"""
import os

# Economic Data
os.environ['FRED_API_KEY'] = 'c9038fbcb0de4b93d89dedebfbd3937a'

# Stock Prices
os.environ['FINNHUB_API_KEY'] = 'd5cfqthr01qsbmgki9pgd5cfqthr01qsbmgki9q0'
os.environ['POLYGON_API_KEY'] = 'saf0qKWy4gHGEIl_SERNhBY1q_Mo6fY6'
os.environ['TWELVE_DATA_API_KEY'] = 'ee3338ef961b48cca43957dce6a7bed9'

# News
os.environ['NEWS_API_KEY'] = 'd7d396ec1ebb45bfb09c516907d2b3dd'

print("✅ API Keys configured!")
print(f"   FRED: {os.environ['FRED_API_KEY'][:10]}...")
print(f"   Finnhub: {os.environ['FINNHUB_API_KEY'][:10]}...")
print(f"   Polygon: {os.environ['POLYGON_API_KEY'][:10]}...")
print(f"   Twelve Data: {os.environ['TWELVE_DATA_API_KEY'][:10]}...")
print(f"   NewsAPI: {os.environ['NEWS_API_KEY'][:10]}...")
