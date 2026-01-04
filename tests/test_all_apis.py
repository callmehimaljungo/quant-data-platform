"""
Test multiple free stock APIs for recent data
"""
import requests
import pandas as pd
from datetime import datetime

# Test APIs
apis = {
    'Twelve Data': {
        'url': 'https://api.twelvedata.com/time_series',
        'params': {
            'symbol': 'AAPL',
            'interval': '1day',
            'start_date': '2024-12-23',
            'end_date': '2024-12-24',
            'apikey': 'demo'  # Free demo key
        }
    },
    'IEX Cloud': {
        'url': 'https://cloud.iexapis.com/stable/stock/AAPL/chart/1m',
        'params': {
            'token': 'pk_test'  # Test token
        }
    },
    'Financial Modeling Prep': {
        'url': 'https://financialmodelingprep.com/api/v3/historical-price-full/AAPL',
        'params': {
            'from': '2024-12-23',
            'to': '2024-12-24',
            'apikey': 'demo'
        }
    },
    'Polygon.io (Free)': {
        'url': 'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-12-23/2024-12-24',
        'params': {
            'apiKey': 'demo'
        }
    },
    'Market Stack': {
        'url': 'http://api.marketstack.com/v1/eod',
        'params': {
            'access_key': 'demo',
            'symbols': 'AAPL',
            'date_from': '2024-12-23',
            'date_to': '2024-12-24'
        }
    }
}

print("🔍 Testing Free Stock APIs for Dec 23-24, 2024\n")
print("=" * 70)

working_apis = []

for name, config in apis.items():
    print(f"\n📊 Testing: {name}")
    print(f"   URL: {config['url']}")
    
    try:
        response = requests.get(config['url'], params=config.get('params', {}), timeout=10)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {str(data)[:200]}...")
            
            # Check if has data
            if data and not isinstance(data, dict) or (isinstance(data, dict) and data.get('values') or data.get('results') or data.get('historical')):
                print(f"   ✅ HAS DATA!")
                working_apis.append(name)
            else:
                print(f"   ⚠️ No data in response")
        else:
            print(f"   ❌ Failed: {response.text[:100]}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")

print("\n" + "=" * 70)
print(f"\n✅ Working APIs: {working_apis}")
print(f"   Total: {len(working_apis)}/{len(apis)}")

if working_apis:
    print(f"\n🎯 Recommended: {working_apis[0]}")
else:
    print("\n❌ No free APIs working with demo keys")
    print("   Need to sign up for API keys at:")
    print("   - https://twelvedata.com (800 req/day free)")
    print("   - https://financialmodelingprep.com (250 req/day free)")
    print("   - https://polygon.io (5 req/min free)")
