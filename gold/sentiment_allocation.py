"""
Gold Layer: Strategy 3 - Sentiment-Adjusted Allocation
Phân bổ danh mục điều chỉnh theo cảm xúc thị trường (News Sentiment + VIX)

Ý nghĩa kinh tế:
- Behavioral Finance: Cảm xúc nhà đầu tư ảnh hưởng đến giá cổ phiếu
- Tin tức tích cực → Nhà đầu tư lạc quan → Giá tăng
- VIX cao + Tin tức tiêu cực → Nên phòng thủ
- Nobel Prize 2017: Richard Thaler về Behavioral Economics

Data sử dụng:
- news_lakehouse: Sentiment điểm tin tức theo ticker
- economic_lakehouse: VIX, market regime
- enriched_stocks.parquet: Price, returns, sectors

Output: data/gold/sentiment_allocation_lakehouse/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
SILVER_PARQUET = SILVER_DIR / 'enriched_stocks.parquet'
ECONOMIC_PATH = SILVER_DIR / 'economic_lakehouse'
NEWS_PATH = SILVER_DIR / 'news_lakehouse'
OUTPUT_PATH = GOLD_DIR / 'sentiment_allocation_lakehouse'

# Thông số chiến lược
TOP_N_STOCKS = 30           # Số cổ phiếu trong danh mục
SENTIMENT_BOOST = 0.2       # Tăng 20% weight cho tin tốt
SENTIMENT_PENALTY = 0.2     # Giảm 20% weight cho tin xấu


# =============================================================================
# SENTIMENT CLASSIFICATION
# =============================================================================
def classify_sentiment(sentiment_score: float) -> str:
    """
    Phân loại sentiment score thành categories
    
    Ý nghĩa kinh tế:
    - Positive: Tin tức tốt, nhà đầu tư lạc quan
    - Negative: Tin tức xấu, nhà đầu tư bi quan
    - Neutral: Không có tin đáng kể
    """
    if sentiment_score > 0.1:
        return 'positive'
    elif sentiment_score < -0.1:
        return 'negative'
    else:
        return 'neutral'


def get_market_mood(df_economic: pd.DataFrame) -> str:
    """
    Đánh giá "mood" tổng thể của thị trường từ VIX
    
    Logic kinh tế:
    - VIX thấp (<15): Thị trường "bình tĩnh" → risk-on
    - VIX cao (>25): Thị trường "hoảng sợ" → risk-off
    """
    if len(df_economic) == 0:
        return 'neutral'
    
    latest_vix = df_economic.sort_values('date')['vix'].iloc[-1]
    
    if latest_vix > 30:
        return 'fear'        # Rất sợ hãi
    elif latest_vix > 25:
        return 'anxious'     # Lo lắng
    elif latest_vix < 15:
        return 'confident'   # Tự tin
    else:
        return 'neutral'


# =============================================================================
# BUILD SENTIMENT-ADJUSTED PORTFOLIO
# =============================================================================
def build_sentiment_portfolio(df_prices: pd.DataFrame,
                               df_news: pd.DataFrame,
                               df_economic: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng danh mục điều chỉnh theo Sentiment
    
    Logic kinh tế:
    1. Lấy sentiment score cho mỗi ticker từ news data
    2. Đánh giá mood thị trường từ VIX
    3. Điều chỉnh weights:
       - Tin tốt + VIX thấp → Tăng weight
       - Tin xấu + VIX cao → Giảm weight
    4. Chọn top stocks với sentiment-adjusted weights
    """
    logger.info("Building Sentiment-Adjusted Portfolio...")
    
    # Bước 1: Đánh giá market mood
    logger.info("  Step 1: Assessing market mood from VIX...")
    market_mood = get_market_mood(df_economic)
    logger.info(f"  [INFO] Market mood: {market_mood.upper()}")
    
    # Bước 2: Process news sentiment
    logger.info("  Step 2: Processing news sentiment data...")
    
    if len(df_news) == 0:
        logger.warning("  [WARN] No news data, using neutral sentiment for all")
        ticker_sentiment = {}
    else:
        # Aggregate sentiment by ticker (lấy average)
        if 'avg_sentiment' in df_news.columns:
            ticker_sentiment = df_news.groupby('ticker')['avg_sentiment'].mean().to_dict()
        elif 'sentiment' in df_news.columns:
            ticker_sentiment = df_news.groupby('ticker')['sentiment'].mean().to_dict()
        else:
            ticker_sentiment = {}
        
        logger.info(f"  [OK] Got sentiment for {len(ticker_sentiment)} tickers")
    
    # Bước 3: Tính base score cho mỗi ticker
    logger.info("  Step 3: Calculating base scores...")
    
    # Lấy data gần nhất (3 tháng)
    latest_date = df_prices['date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=90)
    df_recent = df_prices[df_prices['date'] >= cutoff_date]
    
    ticker_stats = []
    
    for ticker in df_recent['ticker'].unique():
        ticker_df = df_recent[df_recent['ticker'] == ticker]
        
        if len(ticker_df) < 30:
            continue
        
        returns = ticker_df['daily_return'] / 100
        
        # Base score = Recent momentum (3-month return)
        total_return = (1 + returns).prod() - 1
        
        # Get sentiment
        sentiment = ticker_sentiment.get(ticker, 0)
        sentiment_class = classify_sentiment(sentiment)
        
        # Get sector
        sector = ticker_df['sector'].iloc[0] if 'sector' in ticker_df.columns else 'Unknown'
        
        ticker_stats.append({
            'ticker': ticker,
            'sector': sector,
            'recent_return': total_return,
            'sentiment': sentiment,
            'sentiment_class': sentiment_class,
            'num_days': len(ticker_df),
        })
    
    df_stats = pd.DataFrame(ticker_stats)
    logger.info(f"  [OK] Calculated stats for {len(df_stats)} tickers")
    
    # Bước 4: Điều chỉnh score theo sentiment và market mood
    logger.info("  Step 4: Adjusting scores based on sentiment and market mood...")
    
    # Weight adjustment logic
    def calculate_adjustment(row, market_mood):
        """
        Tính hệ số điều chỉnh dựa trên sentiment và market mood
        
        Logic kinh tế:
        - Khi thị trường confident + tin tốt → Tăng mạnh (risk-on)
        - Khi thị trường fear + tin xấu → Giảm mạnh (risk-off)
        - Khi signals trái chiều → Điều chỉnh nhẹ
        """
        sentiment = row['sentiment']
        
        # Base adjustment từ sentiment
        if sentiment > 0.1:
            sentiment_adj = 1 + SENTIMENT_BOOST  # +20%
        elif sentiment < -0.1:
            sentiment_adj = 1 - SENTIMENT_PENALTY  # -20%
        else:
            sentiment_adj = 1.0
        
        # Market mood adjustment
        if market_mood == 'confident':
            # Thị trường tự tin → Tăng cường positive sentiment
            mood_adj = 1.1 if sentiment > 0 else 1.0
        elif market_mood == 'fear':
            # Thị trường sợ hãi → Giảm tất cả trừ defensive
            if row['sector'] in ['Healthcare', 'Utilities', 'Consumer Staples']:
                mood_adj = 1.1  # Defensive sectors vẫn tốt
            else:
                mood_adj = 0.9  # Các sector khác giảm
        else:
            mood_adj = 1.0
        
        return sentiment_adj * mood_adj
    
    df_stats['adjustment'] = df_stats.apply(
        lambda row: calculate_adjustment(row, market_mood), axis=1
    )
    
    # Tính adjusted score
    df_stats['adjusted_score'] = df_stats['recent_return'] * df_stats['adjustment']
    
    # Bước 5: Chọn top stocks
    logger.info(f"  Step 5: Selecting top {TOP_N_STOCKS} stocks...")
    
    # Lọc bỏ các stocks có điều chỉnh quá tiêu cực khi thị trường fear
    if market_mood in ['fear', 'anxious']:
        # Ưu tiên stocks có sentiment tốt hoặc neutral
        df_stats['selection_score'] = df_stats['adjusted_score'] + df_stats['sentiment'] * 0.5
    else:
        df_stats['selection_score'] = df_stats['adjusted_score']
    
    df_selected = df_stats.nlargest(TOP_N_STOCKS, 'selection_score').copy()
    
    # Bước 6: Tính weights
    logger.info("  Step 6: Calculating weights...")
    
    # Chuyển score về positive (thêm offset)
    min_score = df_selected['adjusted_score'].min()
    if min_score < 0:
        df_selected['weight_base'] = df_selected['adjusted_score'] - min_score + 0.01
    else:
        df_selected['weight_base'] = df_selected['adjusted_score'] + 0.01
    
    total_weight = df_selected['weight_base'].sum()
    df_selected['weight'] = df_selected['weight_base'] / total_weight
    
    # Add metadata
    df_selected['market_mood'] = market_mood
    df_selected['strategy'] = 'sentiment_adjusted'
    df_selected['created_at'] = datetime.now()
    
    # Select output columns
    output_cols = ['ticker', 'sector', 'recent_return', 'sentiment', 'sentiment_class',
                   'adjustment', 'weight', 'market_mood', 'strategy', 'created_at']
    df_output = df_selected[output_cols].reset_index(drop=True)
    
    logger.info(f"  [OK] Portfolio built with {len(df_output)} stocks")
    
    return df_output


# =============================================================================
# MAIN
# =============================================================================
def run_sentiment_allocation() -> pd.DataFrame:
    """Run Sentiment-Adjusted strategy"""
    from utils import is_lakehouse_table, lakehouse_to_pandas, pandas_to_lakehouse
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("GOLD LAYER: SENTIMENT-ADJUSTED PORTFOLIO")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Economic Rationale:")
    logger.info("  - Behavioral Finance: Investor sentiment affects prices")
    logger.info("  - Positive news → Optimism → Price increases")
    logger.info("  - VIX high + Negative news → Defensive positioning")
    logger.info("  - Academic backing: Nobel Prize 2017 (Richard Thaler)")
    logger.info("")
    
    # Load data
    logger.info("Loading data from Silver layer...")
    
    if SILVER_PARQUET.exists():
        df_prices = pd.read_parquet(SILVER_PARQUET)
        logger.info(f"  [OK] Prices: {len(df_prices):,} rows")
    else:
        raise FileNotFoundError(f"Price data not found: {SILVER_PARQUET}")
    
    if is_lakehouse_table(ECONOMIC_PATH):
        df_economic = lakehouse_to_pandas(ECONOMIC_PATH)
        logger.info(f"  [OK] Economic: {len(df_economic):,} rows")
    else:
        df_economic = pd.DataFrame()
        logger.warning("  [WARN] Economic data not found")
    
    if is_lakehouse_table(NEWS_PATH):
        df_news = lakehouse_to_pandas(NEWS_PATH)
        logger.info(f"  [OK] News: {len(df_news):,} rows")
    else:
        df_news = pd.DataFrame()
        logger.warning("  [WARN] News data not found")
    
    # Build portfolio
    df_portfolio = build_sentiment_portfolio(df_prices, df_news, df_economic)
    
    if len(df_portfolio) == 0:
        logger.error("[ERROR] Failed to build portfolio")
        return pd.DataFrame()
    
    # Save to Lakehouse
    logger.info(f"\nSaving to: {OUTPUT_PATH}")
    pandas_to_lakehouse(df_portfolio, OUTPUT_PATH, mode="overwrite")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[OK] SENTIMENT-ADJUSTED PORTFOLIO COMPLETED")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 70)
    
    # Print portfolio
    print("\n--- SENTIMENT-ADJUSTED PORTFOLIO ---")
    print(f"Market Mood: {df_portfolio['market_mood'].iloc[0].upper()}")
    print("")
    print(df_portfolio[['ticker', 'sector', 'sentiment_class', 'adjustment', 'weight']].to_string(index=False))
    print(f"\nTotal stocks: {len(df_portfolio)}")
    print(f"Positive sentiment: {(df_portfolio['sentiment_class'] == 'positive').sum()}")
    print(f"Negative sentiment: {(df_portfolio['sentiment_class'] == 'negative').sum()}")
    print(f"Neutral sentiment: {(df_portfolio['sentiment_class'] == 'neutral').sum()}")
    
    return df_portfolio


def main() -> int:
    try:
        run_sentiment_allocation()
        logger.info("\n[OK] Done! Next: python gold/run_all_strategies.py")
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
