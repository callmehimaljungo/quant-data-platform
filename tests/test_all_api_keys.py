"""
Test all API keys at once
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env
load_dotenv()

print("🧪 Testing All API Keys (using env vars)\n")
print("=" * 70)

# 1. Finnhub
print("\n1️⃣ Testing Finnhub...")
finnhub_key = os.getenv("FINNHUB_KEY")
if not finnhub_key:
    print("   ⚠️ FINNHUB_KEY not found in environment")
else:
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS! AAPL price: ${data.get('c', 'N/A')}")
        else:
            print(f"   ❌ FAILED: HTTP {response.status_code}")
            print(f"   Response: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

# 2. Polygon (Massive)
print("\n2️⃣ Testing Polygon/Massive...")
polygon_key = os.getenv("POLYGON_KEY")
if not polygon_key:
    print("   ⚠️ POLYGON_KEY not found in environment")
else:
    try:
        # Test with previous trading day
        date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{date}/{date}?apiKey={polygon_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                print(f"   ✅ SUCCESS! Got {len(data['results'])} data points")
            else:
                print(f"   ⚠️ No data, but API works: {data.get('status')}")
        else:
            print(f"   ❌ FAILED: HTTP {response.status_code}")
            print(f"   Response: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

# 3. NewsAPI
print("\n3️⃣ Testing NewsAPI...")
newsapi_key = os.getenv("NEWSAPI_KEY")
if not newsapi_key:
    print("   ⚠️ NEWSAPI_KEY not found in environment")
else:
    try:
        url = f"https://newsapi.org/v2/everything?q=stock+market&apiKey={newsapi_key}&pageSize=5"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"   ✅ SUCCESS! Got {len(articles)} articles")
            if articles:
                print(f"   Sample: {articles[0].get('title', '')[:60]}...")
        else:
            print(f"   ❌ FAILED: HTTP {response.status_code}")
            print(f"   Response: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

print("\n" + "=" * 70)
print("✅ API Testing Complete!")
