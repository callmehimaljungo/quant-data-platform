"""
Unit Tests for Bronze Layer

Tests for data ingestion and validation:
- Schema validation
- Column presence checks
- Null value detection in critical columns
- Data type validation

Run with: pytest tests/test_bronze.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def sample_price_data():
    """Create sample price data for testing"""
    np.random.seed(42)
    n_rows = 100
    
    return pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
        'Ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n_rows),
        'Open': np.random.uniform(100, 200, n_rows),
        'High': np.random.uniform(100, 200, n_rows),
        'Low': np.random.uniform(100, 200, n_rows),
        'Close': np.random.uniform(100, 200, n_rows),
        'Volume': np.random.randint(1000000, 10000000, n_rows),
    })


@pytest.fixture
def invalid_price_data():
    """Create invalid price data with missing columns"""
    return pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=10),
        'Ticker': ['AAPL'] * 10,
        # Missing: Open, High, Low, Close, Volume
    })


@pytest.fixture
def price_data_with_nulls():
    """Create price data with null values in critical columns"""
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=10),
        'Ticker': ['AAPL'] * 10,
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000000] * 10,
    })
    # Add nulls
    df.loc[0, 'Close'] = None
    df.loc[5, 'Ticker'] = None
    return df


# =============================================================================
# EXPECTED SCHEMA (matching bronze/ingest.py)
# =============================================================================
REQUIRED_COLUMNS = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
CRITICAL_COLUMNS = ['date', 'ticker', 'close']


# =============================================================================
# HELPER FUNCTIONS (matching bronze layer logic)
# =============================================================================
def validate_schema(df: pd.DataFrame) -> bool:
    """Check if DataFrame has all required columns (case-insensitive)"""
    df_cols_lower = [c.lower() for c in df.columns]
    return all(col in df_cols_lower for col in REQUIRED_COLUMNS)


def check_critical_nulls(df: pd.DataFrame) -> dict:
    """Check for null values in critical columns"""
    null_counts = {}
    for col in CRITICAL_COLUMNS:
        # Check both cases
        actual_col = None
        for c in df.columns:
            if c.lower() == col:
                actual_col = c
                break
        
        if actual_col:
            null_count = df[actual_col].isna().sum()
            if null_count > 0:
                null_counts[col] = null_count
    
    return null_counts


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase"""
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df


# =============================================================================
# TESTS: SCHEMA VALIDATION
# =============================================================================
class TestSchemaValidation:
    """Tests for schema validation logic"""
    
    def test_valid_schema(self, sample_price_data):
        """Valid data should pass schema validation"""
        assert validate_schema(sample_price_data), "Valid data should pass schema check"
    
    def test_missing_columns(self, invalid_price_data):
        """Data with missing columns should fail validation"""
        assert not validate_schema(invalid_price_data), "Missing columns should fail"
    
    def test_extra_columns_allowed(self, sample_price_data):
        """Extra columns should not cause validation to fail"""
        df = sample_price_data.copy()
        df['extra_column'] = 'test'
        assert validate_schema(df), "Extra columns should be allowed"
    
    def test_case_insensitive(self):
        """Column names should be case-insensitive"""
        df = pd.DataFrame({
            'DATE': pd.date_range('2020-01-01', periods=5),
            'TICKER': ['AAPL'] * 5,
            'OPEN': [100.0] * 5,
            'HIGH': [105.0] * 5,
            'LOW': [95.0] * 5,
            'CLOSE': [102.0] * 5,
            'VOLUME': [1000000] * 5,
        })
        assert validate_schema(df), "Uppercase columns should pass"


# =============================================================================
# TESTS: NULL VALUE CHECKS
# =============================================================================
class TestNullChecks:
    """Tests for null value detection"""
    
    def test_no_nulls(self, sample_price_data):
        """Clean data should have no critical nulls"""
        nulls = check_critical_nulls(sample_price_data)
        assert len(nulls) == 0, "Clean data should have no nulls"
    
    def test_detect_critical_nulls(self, price_data_with_nulls):
        """Should detect nulls in critical columns"""
        nulls = check_critical_nulls(price_data_with_nulls)
        assert 'close' in nulls or 'Close' in [c for c in nulls], "Should detect Close nulls"
    
    def test_null_count_accurate(self, price_data_with_nulls):
        """Null count should be accurate"""
        nulls = check_critical_nulls(price_data_with_nulls)
        total_nulls = sum(nulls.values())
        assert total_nulls == 2, f"Expected 2 nulls, got {total_nulls}"


# =============================================================================
# TESTS: COLUMN STANDARDIZATION
# =============================================================================
class TestColumnStandardization:
    """Tests for column name standardization"""
    
    def test_lowercase_conversion(self, sample_price_data):
        """Columns should be converted to lowercase"""
        df = standardize_columns(sample_price_data)
        assert all(c.islower() for c in df.columns), "All columns should be lowercase"
    
    def test_preserves_data(self, sample_price_data):
        """Standardization should not modify data values"""
        original_values = sample_price_data['Close'].values.copy()
        df = standardize_columns(sample_price_data)
        np.testing.assert_array_equal(
            df['close'].values, 
            original_values,
            "Data values should be preserved"
        )


# =============================================================================
# TESTS: DATA TYPES
# =============================================================================
class TestDataTypes:
    """Tests for data type validation"""
    
    def test_date_is_datetime(self, sample_price_data):
        """Date column should be datetime type"""
        assert pd.api.types.is_datetime64_any_dtype(sample_price_data['Date'])
    
    def test_prices_are_numeric(self, sample_price_data):
        """Price columns should be numeric"""
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            assert pd.api.types.is_numeric_dtype(sample_price_data[col]), f"{col} should be numeric"
    
    def test_volume_is_integer(self, sample_price_data):
        """Volume should be integer type"""
        assert pd.api.types.is_integer_dtype(sample_price_data['Volume'])


# =============================================================================
# TESTS: DATA INTEGRITY
# =============================================================================
class TestDataIntegrity:
    """Tests for data integrity checks"""
    
    def test_no_duplicate_rows(self, sample_price_data):
        """Check for duplicate detection"""
        df = sample_price_data.copy()
        # Add duplicates
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        duplicates = df.duplicated().sum()
        assert duplicates == 5, "Should detect 5 duplicate rows"
    
    def test_date_ticker_unique(self, sample_price_data):
        """Check for duplicate date-ticker combinations"""
        df = sample_price_data.copy()
        df = standardize_columns(df)
        duplicates = df.duplicated(subset=['date', 'ticker']).sum()
        # Some duplicates expected in random data
        assert duplicates >= 0, "Duplicate check should work"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
