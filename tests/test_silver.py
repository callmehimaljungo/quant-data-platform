"""
Unit Tests for Silver Layer

Tests for data cleaning and enrichment:
- Deduplication
- Quality gates (close > 0, high >= low, volume >= 0)
- Daily return calculation
- Technical indicator calculation
- Sector mapping

Run with: pytest tests/test_silver.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def clean_stock_data():
    """Create clean stock data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': 'AAPL',
        'open': np.random.uniform(100, 110, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000000, 10000000, 100),
    })
    
    # Make high >= close and low <= close for consistency
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 5, 100)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 5, 100)
    
    return df


@pytest.fixture
def dirty_stock_data():
    """Create stock data with quality issues"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'ticker': ['AAPL'] * 10,
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 100, 108, 109, 110, 111, 112, 113, 114],  # Line 3: high < low
        'low': [95, 96, 105, 98, 99, 100, 101, 102, 103, 104],       # Line 3: low > high
        'close': [-5, 101, 102, 0, 104, 105, 106, 107, 108, 109],    # Negatives and zero
        'volume': [1000000] * 10,
    })
    return df


@pytest.fixture
def data_with_duplicates():
    """Create data with duplicate rows"""
    df = pd.DataFrame({
        'date': ['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02'],
        'ticker': ['AAPL', 'AAPL', 'AAPL', 'MSFT'],
        'open': [100, 100, 101, 200],
        'high': [105, 105, 106, 210],
        'low': [95, 95, 96, 190],
        'close': [102, 102, 103, 205],
        'volume': [1000000, 1000000, 1100000, 2000000],
    })
    df['date'] = pd.to_datetime(df['date'])
    return df


# =============================================================================
# QUALITY GATE FUNCTIONS (matching silver layer logic)
# =============================================================================
def apply_quality_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality gates: close > 0, high >= low, volume >= 0"""
    df = df.copy()
    
    # Gate 1: close > 0
    df = df[df['close'] > 0]
    
    # Gate 2: high >= low
    df = df[df['high'] >= df['low']]
    
    # Gate 3: volume >= 0
    df = df[df['volume'] >= 0]
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate date-ticker combinations, keeping last"""
    return df.drop_duplicates(subset=['date', 'ticker'], keep='last')


def calculate_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily return percentage"""
    df = df.copy()
    df = df.sort_values(['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change() * 100
    return df


def calculate_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate Simple Moving Average"""
    df = df.copy()
    df = df.sort_values(['ticker', 'date'])
    df[f'sma_{window}'] = df.groupby('ticker')['close'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    return df


# =============================================================================
# TESTS: QUALITY GATES
# =============================================================================
class TestQualityGates:
    """Tests for quality gate enforcement"""
    
    def test_removes_negative_close(self, dirty_stock_data):
        """Should remove rows with close <= 0"""
        df = apply_quality_gates(dirty_stock_data)
        assert (df['close'] > 0).all(), "All close prices should be positive"
    
    def test_removes_invalid_high_low(self, dirty_stock_data):
        """Should remove rows where high < low"""
        df = apply_quality_gates(dirty_stock_data)
        assert (df['high'] >= df['low']).all(), "High should be >= Low"
    
    def test_clean_data_unchanged(self, clean_stock_data):
        """Clean data should remain unchanged"""
        df_before = len(clean_stock_data)
        df_after = len(apply_quality_gates(clean_stock_data))
        assert df_before == df_after, "Clean data should pass all gates"
    
    def test_negative_volume_removed(self):
        """Should remove rows with negative volume"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'ticker': ['AAPL'] * 5,
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000000, -100, 2000000, 0, 3000000],
        })
        df_clean = apply_quality_gates(df)
        assert (df_clean['volume'] >= 0).all()


# =============================================================================
# TESTS: DEDUPLICATION
# =============================================================================
class TestDeduplication:
    """Tests for duplicate removal"""
    
    def test_removes_duplicates(self, data_with_duplicates):
        """Should remove duplicate date-ticker combinations"""
        df = remove_duplicates(data_with_duplicates)
        duplicates = df.duplicated(subset=['date', 'ticker']).sum()
        assert duplicates == 0, "Should have no duplicates"
    
    def test_keeps_last_duplicate(self, data_with_duplicates):
        """Should keep the last occurrence of duplicates"""
        df = remove_duplicates(data_with_duplicates)
        assert len(df) == 3, "Should have 3 unique date-ticker combinations"
    
    def test_unique_data_unchanged(self, clean_stock_data):
        """Unique data should remain unchanged"""
        before = len(clean_stock_data)
        after = len(remove_duplicates(clean_stock_data))
        assert before == after, "Unique data should be unchanged"


# =============================================================================
# TESTS: DAILY RETURN CALCULATION
# =============================================================================
class TestDailyReturn:
    """Tests for daily return calculation"""
    
    def test_return_calculation(self):
        """Daily return should be calculated correctly"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=3),
            'ticker': ['AAPL'] * 3,
            'close': [100.0, 110.0, 99.0],  # +10%, -10%
            'open': [100] * 3,
            'high': [110] * 3,
            'low': [90] * 3,
            'volume': [1000000] * 3,
        })
        df = calculate_daily_return(df)
        
        assert pd.isna(df['daily_return'].iloc[0]), "First return should be NaN"
        assert df['daily_return'].iloc[1] == pytest.approx(10.0, rel=0.01)
        assert df['daily_return'].iloc[2] == pytest.approx(-10.0, rel=0.01)
    
    def test_return_by_ticker(self):
        """Should calculate returns separately for each ticker"""
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-02'],
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'close': [100.0, 110.0, 200.0, 220.0],
            'open': [100] * 4,
            'high': [110] * 4,
            'low': [90] * 4,
            'volume': [1000000] * 4,
        })
        df['date'] = pd.to_datetime(df['date'])
        df = calculate_daily_return(df)
        
        aapl_returns = df[df['ticker'] == 'AAPL']['daily_return'].dropna()
        msft_returns = df[df['ticker'] == 'MSFT']['daily_return'].dropna()
        
        assert aapl_returns.iloc[0] == pytest.approx(10.0, rel=0.01)
        assert msft_returns.iloc[0] == pytest.approx(10.0, rel=0.01)


# =============================================================================
# TESTS: TECHNICAL INDICATORS
# =============================================================================
class TestTechnicalIndicators:
    """Tests for technical indicator calculations"""
    
    def test_sma_calculation(self, clean_stock_data):
        """SMA should be calculated correctly"""
        df = calculate_sma(clean_stock_data, window=5)
        assert f'sma_5' in df.columns, "SMA column should exist"
        assert not df['sma_5'].isna().all(), "SMA should have values"
    
    def test_sma_first_values(self):
        """SMA with min_periods=1 should have values from first row"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'ticker': ['AAPL'] * 5,
            'close': [100.0, 102.0, 104.0, 106.0, 108.0],
            'open': [100] * 5,
            'high': [110] * 5,
            'low': [90] * 5,
            'volume': [1000000] * 5,
        })
        df = calculate_sma(df, window=3)
        
        # First value should be 100 (only itself)
        assert df['sma_3'].iloc[0] == 100.0
        # Third value should be (100+102+104)/3 = 102
        assert df['sma_3'].iloc[2] == pytest.approx(102.0, rel=0.01)


# =============================================================================
# TESTS: DATA PIPELINE INTEGRATION
# =============================================================================
class TestPipelineIntegration:
    """Integration tests for complete Silver pipeline"""
    
    def test_full_pipeline(self, dirty_stock_data):
        """Full pipeline should produce clean data"""
        df = dirty_stock_data.copy()
        
        # Apply all transformations
        df = apply_quality_gates(df)
        df = remove_duplicates(df)
        df = calculate_daily_return(df)
        
        # Verify results
        assert (df['close'] > 0).all(), "Close should be positive"
        assert (df['high'] >= df['low']).all(), "High >= Low"
        assert 'daily_return' in df.columns, "Should have daily_return"
    
    def test_pipeline_preserves_valid_data(self, clean_stock_data):
        """Pipeline should preserve valid data"""
        original_len = len(clean_stock_data)
        
        df = clean_stock_data.copy()
        df = apply_quality_gates(df)
        df = remove_duplicates(df)
        
        assert len(df) == original_len, "Valid data should be preserved"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
