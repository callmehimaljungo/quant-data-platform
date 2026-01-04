"""
New York Times Archive API Collector
Fetch historical news from 1962 to present
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['NYT_API_KEY'] = 'p7EIRuM5NOIQEY0VWhy8QrdAdEH6lXVBCdA8K4f1pM1RuVUy'

import requests
import pandas as pd
from datetime import datetime
import time
import json
from config import BRONZE_DIR

print("=" * 80)
print("📰 NEW YORK TIMES ARCHIVE COLLECTION (1962-2024)")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print()

API_KEY = os.environ['NYT_API_KEY']
BASE_URL = "https://api.nytimes.com/svc/archive/v1"

# Generate year-month pairs from 1962 to 2024
date_ranges = []
for year in range(1962, 2025):
    for month in range(1, 13):
        if year == 2024 and month > 12:  # Stop at current month
            break
        date_ranges.append((year, month))

print(f"Total batches to fetch: {len(date_ranges)}")
print(f"Rate limit: 10 req/min (6 sec between requests)")
print(f"Estimated time: {len(date_ranges) * 6 / 60:.1f} minutes")
print()

total_articles = 0
errors = 0
batch_count = 0

output_dir = BRONZE_DIR / 'nyt_archive_lakehouse'
output_dir.mkdir(parents=True, exist_ok=True)

# Fetch in batches (save every 12 months)
for i, (year, month) in enumerate(date_ranges, 1):
    try:
        url = f"{BASE_URL}/{year}/{month}.json?api-key={API_KEY}"
        
        print(f"[{i}/{len(date_ranges)}] Fetching {year}-{month:02d}...", end=" ")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            
            # Collect ALL articles (no filter)
            all_docs = []
            for doc in docs:
                all_docs.append({
                    'news_id': doc.get('_id'),
                    'title': doc.get('headline', {}).get('main'),
                    'summary': doc.get('abstract'),
                    'content': doc.get('lead_paragraph'),
                    'published_at': doc.get('pub_date'),
                    'section': doc.get('section_name'),
                    'news_desk': doc.get('news_desk'),
                    'url': doc.get('web_url'),
                    'source': 'New York Times',
                    'keywords': json.dumps([kw.get('value') for kw in doc.get('keywords', [])]),
                    'raw_json': json.dumps(doc)
                })
            
            total_articles += len(all_docs)
            print(f"✅ {len(all_docs)} articles (Total: {total_articles:,})")
            
            # Save batch every 12 months
            if i % 12 == 0 or i == len(date_ranges):
                if all_docs:
                    df = pd.DataFrame(all_docs)
                    df['fetched_at'] = datetime.now()
                    
                    output_file = output_dir / f'nyt_{year}.parquet'
                    df.to_parquet(output_file, index=False)
                    batch_count += 1
                    print(f"   💾 Saved batch to {output_file}")
            
            # Rate limiting: 10 req/min = 6 sec between requests
            time.sleep(6.1)
            
        elif response.status_code == 429:
            print(f"⚠️ Rate limit, sleeping 60s...")
            time.sleep(60)
            errors += 1
        else:
            print(f"⚠️ HTTP {response.status_code}")
            errors += 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        errors += 1
        continue

print()
print("=" * 80)
print("✅ NYT ARCHIVE COLLECTION COMPLETE!")
print("=" * 80)
print(f"Total financial articles: {total_articles:,}")
print(f"Batches saved: {batch_count}")
print(f"Errors: {errors}")
print(f"Output: {output_dir}")
print(f"Duration: {(datetime.now() - datetime.strptime(str(datetime.now().date()), '%Y-%m-%d')).seconds / 60:.1f} minutes")
