"""
Feature engineering for ML models - technical indicators calculation
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import numpy as np
from typing import List, Optional

from config import LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def add_sma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """Simple Moving Average"""
    df = df.copy()
    for period in periods:
        df[f'sma_{period}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=period, min_periods=period).mean()
        )
    return df


def add_ema(df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
    """Exponential Moving Average"""
    df = df.copy()
    for period in periods:
        df[f'ema_{period}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index"""
    df = df.copy()
    
    def calc_rsi(prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = df.groupby('ticker')['close'].transform(calc_rsi)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD indicator"""
    df = df.copy()
    
    def calc_macd(prices):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    macd_results = df.groupby('ticker')['close'].transform(
        lambda x: calc_macd(x)[0]
    )
    df['macd'] = macd_results
    
    df['macd_signal'] = df.groupby('ticker')['close'].transform(
        lambda x: calc_macd(x)[1]
    )
    
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


def add_roc(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Rate of Change"""
    df = df.copy()
    for period in periods:
        df[f'roc_{period}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change(periods=period) * 100
        )
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    df = df.copy()
    
    df['bb_middle'] = df.groupby('ticker')['close'].transform(
        lambda x: x.rolling(window=period, min_periods=period).mean()
    )
    
    std = df.groupby('ticker')['close'].transform(
        lambda x: x.rolling(window=period, min_periods=period).std()
    )
    
    df['bb_upper'] = df['bb_middle'] + (std_dev * std)
    df['bb_lower'] = df['bb_middle'] - (std_dev * std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range"""
    df = df.copy()
    
    def calc_atr(group):
        high = group['high']
        low = group['low']
        close = group['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr
    
    df['atr'] = df.groupby('ticker', group_keys=False).apply(calc_atr)
    return df


def add_volatility(df: pd.DataFrame, periods: List[int] = [10, 20, 60]) -> pd.DataFrame:
    """Historical Volatility (annualized)"""
    df = df.copy()
    
    returns = df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change()
    )
    
    for period in periods:
        df[f'volatility_{period}'] = returns.groupby(df['ticker']).transform(
            lambda x: x.rolling(window=period, min_periods=period).std() * np.sqrt(252)
        )
    
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features"""
    df = df.copy()
    
    vol_ma = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    df['volume_ratio'] = df['volume'] / vol_ma
    
    def calc_obv_trend(group):
        price_change = group['close'].diff()
        obv_direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        obv = (obv_direction * group['volume']).cumsum()
        obv_ma = pd.Series(obv).rolling(window=10, min_periods=10).mean()
        return pd.Series(np.where(obv > obv_ma, 1, -1), index=group.index)
    
    df['obv_trend'] = df.groupby('ticker', group_keys=False).apply(calc_obv_trend)
    
    return df


def add_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """Multi-period returns"""
    df = df.copy()
    
    for period in periods:
        df[f'return_{period}d'] = df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change(periods=period)
        )
    
    return df


def add_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add macro indicators: VIX (fear), Fed rate"""
    from config import SILVER_DIR
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    df = df.copy()
    econ_path = SILVER_DIR / 'economic_lakehouse'
    
    if not is_lakehouse_table(econ_path):
        logger.warning("Economic data not found, skipping macro features")
        return df
    
    try:
        econ_df = lakehouse_to_pandas(econ_path)
        logger.info(f"  Loaded {len(econ_df):,} economic data rows")
        
        # Standardize date
        if 'date' in econ_df.columns:
            econ_df['date'] = pd.to_datetime(econ_df['date']).dt.date
        
        # Map to standard names
        econ_cols = []
        
        # VIX
        if 'vix' in econ_df.columns:
            econ_df['VIX'] = econ_df['vix']
            econ_cols.append('VIX')
        elif 'VIX' in econ_df.columns:
            econ_cols.append('VIX')
        elif 'VIXCLS' in econ_df.columns:
            econ_df['VIX'] = econ_df['VIXCLS']
            econ_cols.append('VIX')
        
        # Fed rate
        if 'fed_funds_rate' in econ_df.columns:
            econ_df['EFFR'] = econ_df['fed_funds_rate']
            econ_cols.append('EFFR')
        elif 'EFFR' in econ_df.columns:
            econ_cols.append('EFFR')
        elif 'DFF' in econ_df.columns:
            econ_df['EFFR'] = econ_df['DFF']
            econ_cols.append('EFFR')
        
        if not econ_cols:
            logger.warning("No VIX or EFFR columns found")
            return df
        
        # Merge with price data
        df['date_key'] = pd.to_datetime(df['date']).dt.date
        econ_subset = econ_df[['date'] + econ_cols].drop_duplicates('date')
        
        df = df.merge(econ_subset, left_on='date_key', right_on='date', 
                      how='left', suffixes=('', '_econ'))
        
        if 'date_econ' in df.columns:
            df = df.drop(columns=['date_econ'])
        df = df.drop(columns=['date_key'])
        
        # Forward fill missing values
        for col in econ_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Derived features
        if 'VIX' in df.columns:
            # VIX regime
            df['vix_regime'] = pd.cut(df['VIX'], 
                bins=[0, 15, 25, 100], 
                labels=[0, 1, 2]  # 0=low, 1=normal, 2=high
            ).astype(float)
            
            df['vix_change_5d'] = df['VIX'].pct_change(5)
        
        if 'EFFR' in df.columns:
            # Rate direction
            df['rate_change_20d'] = df['EFFR'].diff(20)
        
        logger.info(f"  Added {len(econ_cols)} economic features + derived")
        
    except Exception as e:
        logger.warning(f"Failed to load economic data: {e}")
    
    return df


def add_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add news sentiment features"""
    from config import SILVER_DIR
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    df = df.copy()
    news_path = SILVER_DIR / 'news_lakehouse'
    
    if not is_lakehouse_table(news_path):
        logger.warning("News data not found, skipping sentiment features")
        return df
    
    try:
        news_df = lakehouse_to_pandas(news_path)
        logger.info(f"  Loaded {len(news_df):,} news sentiment rows")
        
        # Standardize date
        if 'date' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        
        # Select columns
        news_cols = ['date', 'ticker']
        if 'avg_sentiment' in news_df.columns:
            news_cols.append('avg_sentiment')
        if 'news_count' in news_df.columns:
            news_cols.append('news_count')
        
        if len(news_cols) <= 2:
            logger.warning("No sentiment columns found in news data")
            return df
        
        news_subset = news_df[news_cols].copy()
        
        # Merge with price data
        df['date_key'] = pd.to_datetime(df['date']).dt.date
        df = df.merge(news_subset, left_on=['date_key', 'ticker'], 
                      right_on=['date', 'ticker'], how='left', suffixes=('', '_news'))
        
        if 'date_news' in df.columns:
            df = df.drop(columns=['date_news'])
        df = df.drop(columns=['date_key'])
        
        # Rename for clarity
        if 'avg_sentiment' in df.columns:
            df = df.rename(columns={'avg_sentiment': 'news_sentiment'})
            df['news_sentiment'] = df['news_sentiment'].fillna(0)  # No news = neutral
        
        if 'news_count' in df.columns:
            df['news_count'] = df['news_count'].fillna(0)
            # Log transform for heavy-tailed distribution
            df['news_attention'] = np.log1p(df['news_count'])
        
        # Sentiment momentum (5-day rolling)
        if 'news_sentiment' in df.columns:
            df['sentiment_ma5'] = df.groupby('ticker')['news_sentiment'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
        
        logger.info("  Added news sentiment features")
        
    except Exception as e:
        logger.warning(f"Failed to load news data: {e}")
    
    return df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create target variable for ML models"""
    df = df.copy()
    
    df['target_return'] = df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change(periods=horizon).shift(-horizon)
    )
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    return df


def engineer_features(df: pd.DataFrame, 
                       include_target: bool = True,
                       target_horizon: int = 1) -> pd.DataFrame:
    """Apply all feature engineering"""
    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 50)
    
    logger.info(f"Input: {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    
    df = df.sort_values(['ticker', 'date']).copy()
    
    logger.info("Adding Moving Averages...")
    df = add_sma(df)
    df = add_ema(df)
    
    logger.info("Adding Momentum Indicators...")
    df = add_rsi(df)
    df = add_macd(df)
    df = add_roc(df)
    
    logger.info("Adding Volatility Indicators...")
    df = add_bollinger_bands(df)
    if 'high' in df.columns and 'low' in df.columns:
        df = add_atr(df)
    df = add_volatility(df)
    
    logger.info("Adding Volume Features...")
    if 'volume' in df.columns:
        df = add_volume_features(df)
    
    logger.info("Adding Returns...")
    df = add_returns(df)
    
    logger.info("Adding Economic Features (VIX, EFFR)...")
    df = add_economic_features(df)
    
    logger.info("Adding News Sentiment Features...")
    df = add_news_features(df)
    
    if include_target:
        logger.info(f"Adding Target (horizon={target_horizon})...")
        df = add_target(df, horizon=target_horizon)
    
    feature_cols = [c for c in df.columns if c not in 
                    ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 
                     'sector', 'daily_return', 'ingested_at', 'cleaned_at']]
    
    logger.info(f"[OK] Created {len(feature_cols)} features")
    logger.info("=" * 50)
    
    return df


def get_feature_columns() -> List[str]:
    """Return list of feature column names"""
    return [
        # Technical indicators
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'roc_5', 'roc_10', 'roc_20',
        'bb_width', 'bb_pct',
        'volatility_10', 'volatility_20', 'volatility_60',
        'volume_ratio', 'obv_trend',
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        # Economic features
        'VIX', 'vix_regime', 'vix_change_5d',
        'EFFR', 'rate_change_20d',
        # News sentiment features
        'news_sentiment', 'news_count', 'news_attention', 'sentiment_ma5',
    ]


def main():
    """Test feature engineering"""
    from config import SILVER_DIR
    
    parquet_path = SILVER_DIR / 'enriched_stocks.parquet'
    
    if parquet_path.exists():
        logger.info(f"Loading data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        test_tickers = df['ticker'].value_counts().head(10).index.tolist()
        df_test = df[df['ticker'].isin(test_tickers)]
        
        logger.info(f"Testing on {len(test_tickers)} tickers, {len(df_test):,} rows")
        
        df_features = engineer_features(df_test)
        
        print("\n--- Sample Features ---")
        print(df_features[['ticker', 'date', 'close', 'rsi', 'macd', 'bb_pct', 'target_direction']].tail(10))
        
        print("\n--- Feature Stats ---")
        feature_cols = get_feature_columns()
        available = [c for c in feature_cols if c in df_features.columns]
        print(df_features[available].describe())
    else:
        logger.error(f"File not found: {parquet_path}")


if __name__ == "__main__":
    main()
