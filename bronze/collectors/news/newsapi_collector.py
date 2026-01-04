"""
NewsAPI Collector for Market News
Free tier: 100 requests/day
"""

import os
from typing import List
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import json

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class NewsAPICollector:
    """Collector for NewsAPI.org"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("NewsAPI key required")
    
    def fetch_market_news(
        self,
        query: str = "stock market OR economy OR finance",
        days_back: int = 7,
        page_size: int = 100
    ) -> pd.DataFrame:
        """Fetch recent market news"""
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"[NewsAPI] Fetching news from {from_date}")
        
        url = f"{self.BASE_URL}/everything"
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"NewsAPI error: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                logger.warning("No articles returned")
                return pd.DataFrame()
            
            # Parse articles
            records = []
            for article in articles:
                records.append({
                    'news_id': hash(article.get('url', '')),
                    'title': article.get('title'),
                    'summary': article.get('description'),
                    'content': article.get('content'),
                    'source': article.get('source', {}).get('name'),
                    'url': article.get('url'),
                    'published_at': pd.to_datetime(article.get('publishedAt')),
                    'author': article.get('author'),
                    'sentiment_score': None,  # To be analyzed later
                    'sentiment_label': None,
                    'raw_json': json.dumps(article)
                })
            
            df = pd.DataFrame(records)
            df['fetched_at'] = datetime.now()
            
            logger.info(f"[NewsAPI] ✅ Fetched {len(df)} articles")
            
            return df
            
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Load API key
    exec(open('config_api_keys.py').read())
    
    collector = NewsAPICollector()
    df = collector.fetch_market_news(days_back=7)
    
    if not df.empty:
        print(f"\n✅ Got {len(df)} articles")
        print(f"   Date range: {df['published_at'].min()} to {df['published_at'].max()}")
        print(f"\n   Sample titles:")
        for title in df['title'].head(3):
            print(f"   - {title}")
