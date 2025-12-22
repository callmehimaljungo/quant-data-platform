"""
Bronze Layer: Market News Loader
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Fetch market news from free sources (NO API key priority)
- Save raw JSON responses + extracted TEXT content
- Provide sample data fallback if API unavailable
- Extract ticker mentions for universe intersection

Data Format:
- INPUT: News APIs (JSON) or sample data
- OUTPUT: 
  - Parquet in Lakehouse format
  - Raw JSON backup
  - Extracted text files

Business Context:
- News sentiment affects stock prices
- Enable sentiment analysis in Gold layer
- Link news to specific tickers for targeted analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import re
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import requests for API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUT_DIR = BRONZE_DIR / 'market_news_lakehouse'
RAW_JSON_DIR = BRONZE_DIR / 'market_news_raw'
TEXT_DIR = BRONZE_DIR / 'market_news_text'

# Free news sources (no API key required or have generous free tier)
NEWS_SOURCES = {
    'finnhub': {
        'base_url': 'https://finnhub.io/api/v1/news',
        'requires_key': True,  # Free tier available
        'key_env': 'FINNHUB_API_KEY'
    },
    'sample': {
        'base_url': None,  # Use generated sample data
        'requires_key': False
    }
}


# =============================================================================
# SAMPLE DATA GENERATOR (Fallback when no API available)
# =============================================================================

# Real news headlines templates for sample data
SAMPLE_HEADLINES = [
    "{ticker} Reports {direction} Quarterly Earnings, Stock {reaction}",
    "{ticker} Announces New {product} Line, Analysts {sentiment}",
    "Breaking: {ticker} CEO Addresses {event} Concerns",
    "{ticker} Stock {direction_verb} After {catalyst} News",
    "Market Watch: {ticker} {direction_verb} on {volume} Volume",
    "Wall Street {sentiment} on {ticker} After {event}",
    "{ticker} Expands Operations to {region}, Shares {reaction}",
    "Analyst {action} {ticker} Price Target to ${price}",
    "{ticker} Partnership with {partner} Drives {sentiment} Outlook",
    "{industry} Sector {direction_verb}: {ticker} Leads the {movement}",
]

SAMPLE_SOURCES = ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'WSJ', 
                  'Financial Times', 'Barron\'s', 'Investor\'s Business Daily']

SAMPLE_PARAMS = {
    'direction': ['Strong', 'Mixed', 'Weak', 'Record', 'Disappointing'],
    'direction_verb': ['Surges', 'Drops', 'Climbs', 'Falls', 'Rises', 'Dips'],
    'reaction': ['Jumps', 'Tumbles', 'Rallies', 'Slides', 'Soars'],
    'sentiment': ['Bullish', 'Bearish', 'Cautious', 'Optimistic', 'Skeptical'],
    'product': ['AI', 'Cloud', 'Hardware', 'Software', 'Service'],
    'event': ['Supply Chain', 'Inflation', 'Competition', 'Regulation', 'Expansion'],
    'catalyst': ['Earnings', 'FDA Approval', 'Contract Win', 'M&A', 'Guidance'],
    'volume': ['Heavy', 'Light', 'Record', 'Above-Average', 'Moderate'],
    'region': ['Asia', 'Europe', 'Latin America', 'Middle East', 'Africa'],
    'action': ['Raises', 'Cuts', 'Maintains', 'Initiates'],
    'partner': ['Microsoft', 'Google', 'Amazon', 'Tesla', 'Apple'],
    'industry': ['Tech', 'Healthcare', 'Energy', 'Finance', 'Consumer'],
    'movement': ['Charge', 'Decline', 'Recovery', 'Rally', 'Selloff'],
}


def generate_sample_news(
    tickers: List[str],
    num_articles: int = 500,
    date_range_days: int = 30
) -> List[Dict]:
    """
    Generate realistic sample news data for testing
    
    Args:
        tickers: List of tickers to generate news for
        num_articles: Number of articles to generate
        date_range_days: Number of days back to generate articles
        
    Returns:
        List of news article dictionaries
    """
    import random
    
    articles = []
    base_date = datetime.now()
    
    for i in range(num_articles):
        # Select random ticker(s)
        num_tickers = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
        article_tickers = random.sample(tickers, min(num_tickers, len(tickers)))
        
        # Generate headline from template
        template = random.choice(SAMPLE_HEADLINES)
        params = {}
        for key in SAMPLE_PARAMS:
            if '{' + key + '}' in template:
                params[key] = random.choice(SAMPLE_PARAMS[key])
        params['ticker'] = article_tickers[0]
        params['price'] = random.randint(50, 500)
        
        headline = template.format(**{k: v for k, v in params.items() if '{' + k + '}' in template})
        
        # Generate sentiment based on words
        positive_words = ['strong', 'surge', 'rise', 'climb', 'rally', 'bullish', 'optimistic', 'raise']
        negative_words = ['weak', 'drop', 'fall', 'slide', 'tumble', 'bearish', 'skeptical', 'cut']
        
        headline_lower = headline.lower()
        pos_count = sum(1 for w in positive_words if w in headline_lower)
        neg_count = sum(1 for w in negative_words if w in headline_lower)
        
        if pos_count > neg_count:
            sentiment_score = random.uniform(0.3, 0.9)
            sentiment_label = 'positive'
        elif neg_count > pos_count:
            sentiment_score = random.uniform(-0.9, -0.3)
            sentiment_label = 'negative'
        else:
            sentiment_score = random.uniform(-0.2, 0.2)
            sentiment_label = 'neutral'
        
        # Random date within range
        days_ago = random.randint(0, date_range_days)
        article_date = base_date - timedelta(days=days_ago, hours=random.randint(0, 23))
        
        # Generate summary (TEXT content)
        summary = f"Analysis of {article_tickers[0]} performance following recent market developments. " \
                  f"The stock has shown {params.get('direction', 'mixed').lower()} momentum amid " \
                  f"ongoing {params.get('event', 'market').lower()} concerns."
        
        # Full content (longer TEXT)
        content = f"{headline}\n\n{summary}\n\n" \
                  f"Market analysts have expressed {params.get('sentiment', 'cautious').lower()} views " \
                  f"on the company's near-term prospects. Trading volume has been " \
                  f"{params.get('volume', 'moderate').lower()}, with institutional investors " \
                  f"showing {'increased' if sentiment_score > 0 else 'decreased'} interest.\n\n" \
                  f"Related tickers: {', '.join(article_tickers)}"
        
        # Create article dict
        news_id = hashlib.md5(f"{headline}{article_date}".encode()).hexdigest()[:16]
        
        article = {
            'news_id': news_id,
            'title': headline,
            'summary': summary,
            'content': content,
            'source': random.choice(SAMPLE_SOURCES),
            'url': f"https://example.com/news/{news_id}",
            'published_at': article_date.isoformat(),
            'tickers': json.dumps(article_tickers),  # Store as JSON string
            'tickers_list': article_tickers,  # For processing
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': sentiment_label,
            'raw_json': json.dumps({
                'id': news_id,
                'headline': headline,
                'summary': summary,
                'source': random.choice(SAMPLE_SOURCES),
                'tickers': article_tickers,
                'sentiment': sentiment_score,
                'timestamp': article_date.isoformat()
            }),
            'fetched_at': datetime.now().isoformat()
        }
        
        articles.append(article)
    
    return articles


# =============================================================================
# NEWS FETCHER (API-based)
# =============================================================================

def fetch_from_finnhub(
    category: str = 'general',
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Fetch news from Finnhub API (requires API key from env)
    
    Args:
        category: News category
        api_key: API key (or from environment)
        
    Returns:
        List of news articles
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests not installed")
    
    api_key = api_key or os.environ.get('FINNHUB_API_KEY')
    
    if not api_key:
        logger.warning("FINNHUB_API_KEY not set, using sample data")
        return None
    
    try:
        url = f"https://finnhub.io/api/v1/news?category={category}&token={api_key}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        articles = []
        for item in data:
            article = {
                'news_id': str(item.get('id')),
                'title': item.get('headline'),
                'summary': item.get('summary'),
                'content': item.get('summary'),  # Finnhub only provides summary
                'source': item.get('source'),
                'url': item.get('url'),
                'published_at': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                'tickers': json.dumps(item.get('related', '').split(',') if item.get('related') else []),
                'sentiment_score': None,  # Finnhub basic tier doesn't include sentiment
                'sentiment_label': None,
                'raw_json': json.dumps(item),
                'fetched_at': datetime.now().isoformat()
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        logger.error(f"Failed to fetch from Finnhub: {e}")
        return None


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_available_tickers() -> List[str]:
    """Load list of tickers from existing price data"""
    prices_file = BRONZE_DIR / 'all_stock_data.parquet'
    
    if not prices_file.exists():
        # Return sample tickers for testing
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
                'JPM', 'BAC', 'WFC', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    
    df = pd.read_parquet(prices_file, columns=['Ticker'])
    return df['Ticker'].unique().tolist()


def load_market_news(
    source: str = 'auto',
    num_articles: int = 500,
    save_text_files: bool = True
) -> pd.DataFrame:
    """
    Main function to load market news
    
    Args:
        source: Data source ('finnhub', 'sample', 'auto')
        num_articles: Number of articles to fetch/generate
        save_text_files: Whether to save individual text files
        
    Returns:
        DataFrame with news articles
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BRONZE LAYER: MARKET NEWS INGESTION")
    logger.info("=" * 70)
    
    articles = None
    
    # Try API first if available
    if source in ['auto', 'finnhub']:
        if os.environ.get('FINNHUB_API_KEY'):
            logger.info("Fetching from Finnhub API...")
            articles = fetch_from_finnhub()
    
    # Fall back to sample data
    if articles is None:
        logger.info("Using sample news data (no API key available)")
        tickers = load_available_tickers()
        
        # Select subset of tickers for sample news (top 100)
        sample_tickers = tickers[:100] if len(tickers) > 100 else tickers
        
        articles = generate_sample_news(
            tickers=sample_tickers,
            num_articles=num_articles
        )
        logger.info(f"Generated {len(articles)} sample articles")
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Parse dates
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['fetched_at'] = pd.to_datetime(df['fetched_at'])
    
    # Remove helper columns
    if 'tickers_list' in df.columns:
        df = df.drop(columns=['tickers_list'])
    
    # Log summary
    logger.info(f"\n✓ Loaded {len(df):,} news articles")
    logger.info(f"✓ Date range: {df['published_at'].min()} to {df['published_at'].max()}")
    logger.info(f"✓ Sources: {df['source'].nunique()} unique")
    
    # Sentiment distribution
    if df['sentiment_label'].notna().any():
        logger.info("\nSentiment Distribution:")
        for label, count in df['sentiment_label'].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"  {label}: {count:,} ({pct:.1f}%)")
    
    # Save text files (separate TEXT format)
    if save_text_files:
        TEXT_DIR.mkdir(parents=True, exist_ok=True)
        
        for idx, row in df.iterrows():
            text_file = TEXT_DIR / f"{row['news_id']}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"TITLE: {row['title']}\n")
                f.write(f"SOURCE: {row['source']}\n")
                f.write(f"DATE: {row['published_at']}\n")
                f.write(f"TICKERS: {row['tickers']}\n")
                f.write(f"SENTIMENT: {row['sentiment_score']} ({row['sentiment_label']})\n")
                f.write(f"\n{'='*50}\n\n")
                f.write(row['content'] if row['content'] else row['summary'])
        
        logger.info(f"✓ Saved {len(df)} text files to {TEXT_DIR}")
    
    # Save raw JSON
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    json_file = RAW_JSON_DIR / f'news_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(json_file, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2, default=str)
    
    logger.info(f"✓ Saved raw JSON to {json_file}")
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info(f"NEWS INGESTION COMPLETED")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("=" * 70)
    
    return df


def save_to_lakehouse(df: pd.DataFrame) -> str:
    """Save news DataFrame to Lakehouse format"""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"✓ Saved to {path}")
    return path


def register_in_universe(df: pd.DataFrame):
    """Register mentioned tickers in the universe"""
    from utils.ticker_universe import get_universe
    
    universe = get_universe()
    
    # Extract all mentioned tickers
    all_tickers = set()
    for tickers_json in df['tickers'].dropna():
        try:
            tickers = json.loads(tickers_json)
            if isinstance(tickers, list):
                all_tickers.update(tickers)
        except:
            pass
    
    if all_tickers:
        universe.register_source('news', list(all_tickers))
        logger.info(f"✓ Registered {len(all_tickers):,} tickers in universe")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(num_articles: int = 500, test: bool = False):
    """Main execution function"""
    logger.info("")
    logger.info("🚀 BRONZE LAYER: MARKET NEWS LOADER")
    logger.info("")
    
    try:
        if test:
            num_articles = 50
            logger.info("Running in TEST mode (50 articles only)")
        
        # Load news
        df = load_market_news(num_articles=num_articles)
        
        # Save to Lakehouse
        save_to_lakehouse(df)
        
        # Register in universe
        register_in_universe(df)
        
        logger.info("")
        logger.info("✅ Market news loading completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info(f"✅ Text files: {TEXT_DIR}")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ News loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("")
        return 1


if __name__ == "__main__":
    import sys
    
    test_mode = '--test' in sys.argv
    num_articles = 500
    
    for arg in sys.argv[1:]:
        if arg.isdigit():
            num_articles = int(arg)
    
    exit(main(num_articles=num_articles, test=test_mode))
