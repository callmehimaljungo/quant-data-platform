"""
Test Finnhub API with single ticker
"""
import os
os.environ['FINNHUB_API_KEY'] = 'd5cfqthr01qsbmgki9pgd5cfqthr01qsbmgki9q0'

import requests
import pandas as pd
from datetime import datetime

API_KEY = os.environ['FINNHUB_API_KEY']
BASE_URL = "https://finnhub.io/api/v1"

# Test with single ticker
ticker = 'AAPL'
from_ts = int(pd.to_datetime('2024-12-23').timestamp())
to_ts = int(pd.to_datetime('2024-12-24').timestamp())

url = f"{BASE_URL}/stock/candle"
params = {
    'symbol': ticker,
    'resolution': 'D',
    'from': from_ts,
    'to': to_ts,
    'token': API_KEY
}

print(f"Testing Finnhub API...")
print(f"URL: {url}")
print(f"Ticker: {ticker}")
print(f"Dates: 2024-12-23 to 2024-12-24")

response = requests.get(url, params=params, timeout=30)

print(f"\nResponse:")
print(f"  Status: {response.status_code}")
print(f"  Data: {response.json()}")

if response.status_code == 200:
    data = response.json()
    if data.get('s') == 'ok':
        print(f"\n✅ SUCCESS! Got {len(data.get('c', []))} data points")
    else:
        print(f"\n⚠️ No data: {data}")
else:
    print(f"\n❌ FAILED: HTTP {response.status_code}")
