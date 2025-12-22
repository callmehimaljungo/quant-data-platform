"""
Silver Layer: Process Market News
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Process raw JSON + Text news from Bronze layer
- Extract ticker mentions using regex (NO AI/ML)
- Aggregate sentiment by ticker and date
- Clean and normalize text content
- Link news to ticker universe

Processing Steps:
1. Load raw JSON/Text from Bronze
2. Parse and normalize timestamps
3. Clean text (HTML removal, special chars)
4. Extract ticker mentions via regex
5. Aggregate sentiment by ticker/date
6. Filter to tickers in universe
7. Save to Silver Lakehouse
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import re
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, SILVER_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

INPUT_DIR = BRONZE_DIR / 'market_news_lakehouse'
TEXT_DIR = BRONZE_DIR / 'market_news_text'
OUTPUT_DIR = SILVER_DIR / 'news_lakehouse'

# Common words that look like tickers but aren't
TICKER_BLACKLIST = {
    'A', 'I', 'IT', 'AT', 'GO', 'BE', 'DO', 'IS', 'AS', 'OR', 'AN', 'ON',
    'US', 'UK', 'EU', 'CEO', 'CFO', 'IPO', 'ETF', 'GDP', 'CPI', 'FED',
    'NYSE', 'NASDAQ', 'NYSE', 'SEC', 'FDA', 'DOW', 'USD', 'EUR', 'GBP',
    'Q1', 'Q2', 'Q3', 'Q4', 'YTD', 'YOY', 'MOM', 'WOW', 'EPS', 'PE',
    'BUY', 'SELL', 'HOLD', 'NEW', 'THE', 'AND', 'FOR', 'ALL', 'TOP',
    'INC', 'LLC', 'LTD', 'CORP', 'CO', 'EST', 'PM', 'AM',
}


# =============================================================================
# TEXT PROCESSING FUNCTIONS (NO AI/ML)
# =============================================================================

def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    if pd.isna(text):
        return ""
    return re.sub(r'<[^>]+>', '', str(text))


def clean_special_chars(text: str) -> str:
    """Remove special characters, keeping alphanumeric and basic punctuation"""
    if pd.isna(text):
        return ""
    # Keep letters, numbers, spaces, and basic punctuation
    return re.sub(r'[^\w\s.,!?;:\'-]', ' ', str(text))


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    if pd.isna(text):
        return ""
    return ' '.join(str(text).split())


def clean_text(text: str) -> str:
    """Apply all text cleaning steps"""
    text = clean_html(text)
    text = clean_special_chars(text)
    text = normalize_whitespace(text)
    return text.strip()


def extract_tickers_from_text(text: str, valid_tickers: Set[str] = None) -> List[str]:
    """
    Extract stock ticker mentions from text using regex patterns
    
    Patterns detected:
    - $AAPL format (explicit ticker)
    - AAPL format in uppercase (2-5 letters)
    
    Args:
        text: Text to search
        valid_tickers: Optional set of valid tickers to filter against
        
    Returns:
        List of detected tickers
    """
    if pd.isna(text):
        return []
    
    text = str(text)
    found_tickers = set()
    
    # Pattern 1: Explicit ticker format ($AAPL)
    pattern_explicit = r'\$([A-Z]{1,5})\b'
    matches = re.findall(pattern_explicit, text)
    found_tickers.update(matches)
    
    # Pattern 2: Uppercase words that look like tickers (2-5 letters)
    pattern_implicit = r'\b([A-Z]{2,5})\b'
    matches = re.findall(pattern_implicit, text)
    
    # Filter out blacklisted words
    for match in matches:
        if match not in TICKER_BLACKLIST:
            found_tickers.add(match)
    
    # Filter against valid tickers if provided
    if valid_tickers:
        found_tickers = found_tickers.intersection(valid_tickers)
    
    return list(found_tickers)


def parse_ticker_json(tickers_str: str) -> List[str]:
    """Parse tickers from JSON string format"""
    if pd.isna(tickers_str):
        return []
    
    try:
        if isinstance(tickers_str, list):
            return [t.strip().upper() for t in tickers_str if t]
        tickers = json.loads(str(tickers_str))
        if isinstance(tickers, list):
            return [t.strip().upper() for t in tickers if t]
        return []
    except:
        # Try comma-separated
        return [t.strip().upper() for t in str(tickers_str).split(',') if t.strip()]


def categorize_sentiment(score: float) -> str:
    """Categorize sentiment score into label"""
    if pd.isna(score):
        return 'neutral'
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def load_bronze_news() -> pd.DataFrame:
    """Load raw news from Bronze layer"""
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    if is_lakehouse_table(INPUT_DIR):
        logger.info(f"Loading from Lakehouse: {INPUT_DIR}")
        df = lakehouse_to_pandas(INPUT_DIR)
    else:
        # Try raw JSON
        json_dir = BRONZE_DIR / 'market_news_raw'
        if json_dir.exists():
            json_files = sorted(json_dir.glob('*.json'))
            if json_files:
                with open(json_files[-1], 'r') as f:  # Latest file
                    data = json.load(f)
                df = pd.DataFrame(data)
                logger.info(f"Loaded from JSON: {json_files[-1]}")
            else:
                raise FileNotFoundError(f"No JSON files in {json_dir}")
        else:
            raise FileNotFoundError(f"No Bronze news found")
    
    logger.info(f"✓ Loaded {len(df):,} news articles from Bronze")
    return df


def load_text_files() -> Dict[str, str]:
    """Load text content from text files"""
    texts = {}
    
    if not TEXT_DIR.exists():
        return texts
    
    for txt_file in TEXT_DIR.glob('*.txt'):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            texts[txt_file.stem] = content
        except Exception as e:
            logger.warning(f"Failed to read {txt_file}: {e}")
    
    logger.info(f"✓ Loaded {len(texts)} text files")
    return texts


def get_valid_tickers() -> Set[str]:
    """Get set of valid tickers from universe"""
    from utils.ticker_universe import get_universe
    
    universe = get_universe()
    tickers = universe.get_source_tickers('prices')
    
    if not tickers:
        # Fallback to loading from price data
        prices_file = BRONZE_DIR / 'all_stock_data.parquet'
        if prices_file.exists():
            df = pd.read_parquet(prices_file, columns=['Ticker'])
            tickers = set(df['Ticker'].unique())
    
    logger.info(f"✓ Loaded {len(tickers):,} valid tickers from universe")
    return tickers


def process_news(df: pd.DataFrame, valid_tickers: Set[str]) -> pd.DataFrame:
    """
    Process news articles: clean text, extract tickers, prepare for aggregation
    """
    logger.info("Processing news articles...")
    
    # Clean text content
    for col in ['title', 'summary', 'content']:
        if col in df.columns:
            df[col + '_clean'] = df[col].apply(clean_text)
    
    # Parse published_at
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df['news_date'] = df['published_at'].dt.date
    
    # Extract and combine tickers
    def get_all_tickers(row):
        tickers = set()
        
        # From explicit tickers field
        if 'tickers' in row.index:
            tickers.update(parse_ticker_json(row['tickers']))
        
        # Extract from title
        if 'title_clean' in row.index:
            tickers.update(extract_tickers_from_text(row['title_clean'], valid_tickers))
        
        # Extract from content
        if 'content_clean' in row.index:
            tickers.update(extract_tickers_from_text(row['content_clean'], valid_tickers))
        
        return list(tickers)
    
    df['extracted_tickers'] = df.apply(get_all_tickers, axis=1)
    
    # Filter to articles that mention valid tickers
    df['has_valid_tickers'] = df['extracted_tickers'].apply(
        lambda x: len(set(x).intersection(valid_tickers)) > 0
    )
    
    # Categorize sentiment
    if 'sentiment_score' in df.columns:
        df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    else:
        df['sentiment_category'] = 'neutral'
    
    logger.info(f"✓ Processed {len(df):,} articles")
    logger.info(f"  - With valid tickers: {df['has_valid_tickers'].sum():,}")
    
    return df


def aggregate_by_ticker_date(
    df: pd.DataFrame, 
    valid_tickers: Set[str]
) -> pd.DataFrame:
    """
    Aggregate news by ticker and date
    
    Output schema:
    - date: News date
    - ticker: Stock ticker
    - news_count: Number of articles
    - avg_sentiment: Average sentiment score
    - positive_count, negative_count, neutral_count
    - headlines: Concatenated headlines
    """
    logger.info("Aggregating news by ticker and date...")
    
    # Explode to one row per ticker mention
    exploded_rows = []
    
    for _, row in df.iterrows():
        tickers = row.get('extracted_tickers', [])
        if not tickers:
            continue
            
        for ticker in tickers:
            if ticker in valid_tickers:
                exploded_rows.append({
                    'date': row.get('news_date'),
                    'ticker': ticker,
                    'title': row.get('title', ''),
                    'sentiment_score': row.get('sentiment_score', 0),
                    'sentiment_category': row.get('sentiment_category', 'neutral'),
                })
    
    if not exploded_rows:
        logger.warning("No valid ticker mentions found")
        return pd.DataFrame()
    
    exploded_df = pd.DataFrame(exploded_rows)
    exploded_df['date'] = pd.to_datetime(exploded_df['date'])
    
    # Aggregate
    agg_df = exploded_df.groupby(['date', 'ticker']).agg({
        'title': lambda x: ' | '.join(x.dropna().astype(str)[:5]),  # First 5 headlines
        'sentiment_score': ['count', 'mean'],
        'sentiment_category': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['date', 'ticker', 'headlines', 'news_count', 'avg_sentiment', 'sentiment_dist']
    
    # Extract sentiment counts
    agg_df['positive_count'] = agg_df['sentiment_dist'].apply(lambda x: x.get('positive', 0))
    agg_df['negative_count'] = agg_df['sentiment_dist'].apply(lambda x: x.get('negative', 0))
    agg_df['neutral_count'] = agg_df['sentiment_dist'].apply(lambda x: x.get('neutral', 0))
    agg_df = agg_df.drop(columns=['sentiment_dist'])
    
    # Add metadata
    agg_df['processed_at'] = datetime.now()
    
    logger.info(f"✓ Created {len(agg_df):,} ticker-date combinations")
    logger.info(f"  - Unique tickers: {agg_df['ticker'].nunique():,}")
    logger.info(f"  - Date range: {agg_df['date'].min()} to {agg_df['date'].max()}")
    
    return agg_df


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_all_news() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to process news from Bronze to Silver
    
    Returns:
        Tuple of (processed_articles_df, aggregated_df)
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("SILVER LAYER: NEWS PROCESSING")
    logger.info("=" * 70)
    
    # Step 1: Load from Bronze
    df = load_bronze_news()
    
    # Step 2: Get valid tickers
    valid_tickers = get_valid_tickers()
    
    # Step 3: Process articles
    df = process_news(df, valid_tickers)
    
    # Step 4: Aggregate by ticker/date
    agg_df = aggregate_by_ticker_date(df, valid_tickers)
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info("NEWS PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Input articles: {len(df):,}")
    logger.info(f"  Output aggregations: {len(agg_df):,}")
    logger.info(f"  Duration: {duration:.2f} seconds")
    
    return df, agg_df


def save_to_lakehouse(agg_df: pd.DataFrame) -> str:
    """Save aggregated news to Silver Lakehouse"""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(agg_df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"✓ Saved to {path}")
    return path


def register_in_universe(agg_df: pd.DataFrame):
    """Register tickers with news coverage in universe"""
    from utils.ticker_universe import get_universe
    
    if 'ticker' in agg_df.columns:
        universe = get_universe()
        tickers = agg_df['ticker'].unique().tolist()
        universe.register_source('silver_news', tickers)
        logger.info(f"✓ Registered {len(tickers):,} tickers with news coverage")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    logger.info("")
    logger.info("🚀 SILVER LAYER: NEWS PROCESSOR")
    logger.info("")
    
    try:
        # Process news
        articles_df, agg_df = process_all_news()
        
        if len(agg_df) > 0:
            # Save to Lakehouse
            save_to_lakehouse(agg_df)
            
            # Register in universe
            register_in_universe(agg_df)
        else:
            logger.warning("⚠️ No aggregated news to save")
        
        logger.info("")
        logger.info("✅ News processing completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info("")
        return 0
        
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Bronze news not found: {e}")
        logger.warning("⚠️ Run bronze/news_loader.py first")
        return 1
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("")
        return 1


if __name__ == "__main__":
    exit(main())
