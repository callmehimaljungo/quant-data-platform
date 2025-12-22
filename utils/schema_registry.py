"""
Schema Registry for Lakehouse
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Define schemas for all tables
- Validate data against schema
- Track schema versions
- Support schema evolution

Business Context:
- Ensures data consistency across medallion layers
- Enables schema-on-read validation
- Documents expected data structure
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import json
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

@dataclass
class ColumnSchema:
    """Definition of a single column"""
    name: str
    dtype: str  # pandas dtype string
    nullable: bool = True
    description: str = ""
    allowed_values: Optional[List[Any]] = None
    
    def validate(self, series: pd.Series) -> Tuple[bool, List[str]]:
        """Validate a pandas Series against this schema"""
        errors = []
        
        # Check nulls
        if not self.nullable and series.isna().any():
            null_count = series.isna().sum()
            errors.append(f"Column '{self.name}' has {null_count} null values but is not nullable")
        
        # Check allowed values
        if self.allowed_values is not None:
            invalid = series[~series.isin(self.allowed_values) & ~series.isna()]
            if len(invalid) > 0:
                unique_invalid = invalid.unique()[:5]
                errors.append(f"Column '{self.name}' has invalid values: {unique_invalid}")
        
        return len(errors) == 0, errors


@dataclass
class TableSchema:
    """Definition of a table schema"""
    name: str
    version: str
    columns: List[ColumnSchema]
    description: str = ""
    primary_key: Optional[List[str]] = None
    partition_by: Optional[List[str]] = None
    
    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column schema by name"""
        for col in self.columns:
            if col.name == name:
                return col
        return None
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a DataFrame against this schema"""
        errors = []
        
        # Check for missing required columns
        schema_cols = {col.name for col in self.columns}
        df_cols = set(df.columns)
        
        missing = schema_cols - df_cols
        if missing:
            errors.append(f"Missing columns: {missing}")
        
        # Validate each column
        for col_schema in self.columns:
            if col_schema.name in df.columns:
                is_valid, col_errors = col_schema.validate(df[col_schema.name])
                errors.extend(col_errors)
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Convert schema to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'columns': [
                {
                    'name': col.name,
                    'dtype': col.dtype,
                    'nullable': col.nullable,
                    'description': col.description
                }
                for col in self.columns
            ],
            'primary_key': self.primary_key,
            'partition_by': self.partition_by
        }


# =============================================================================
# BRONZE LAYER SCHEMAS
# =============================================================================

BRONZE_PRICES = TableSchema(
    name="bronze_prices",
    version="1.0",
    description="Raw OHLCV price data from Kaggle",
    columns=[
        ColumnSchema("Date", "datetime64[ns]", nullable=False, description="Trading date"),
        ColumnSchema("Ticker", "object", nullable=False, description="Stock symbol"),
        ColumnSchema("Open", "float64", nullable=False, description="Opening price"),
        ColumnSchema("High", "float64", nullable=False, description="Highest price"),
        ColumnSchema("Low", "float64", nullable=False, description="Lowest price"),
        ColumnSchema("Close", "float64", nullable=False, description="Closing price"),
        ColumnSchema("Volume", "int64", nullable=False, description="Trading volume"),
        ColumnSchema("ingested_at", "datetime64[ns]", nullable=False, description="Ingestion timestamp"),
    ],
    primary_key=["Date", "Ticker"]
)

BRONZE_METADATA = TableSchema(
    name="bronze_stock_metadata",
    version="1.0",
    description="Raw stock metadata from yfinance (JSON format)",
    columns=[
        ColumnSchema("ticker", "object", nullable=False, description="Stock symbol"),
        ColumnSchema("company_name", "object", nullable=True, description="Company name"),
        ColumnSchema("sector", "object", nullable=True, description="GICS Sector"),
        ColumnSchema("industry", "object", nullable=True, description="GICS Industry"),
        ColumnSchema("market_cap", "float64", nullable=True, description="Market capitalization USD"),
        ColumnSchema("exchange", "object", nullable=True, description="Stock exchange"),
        ColumnSchema("currency", "object", nullable=True, description="Trading currency"),
        ColumnSchema("country", "object", nullable=True, description="Country of incorporation"),
        ColumnSchema("website", "object", nullable=True, description="Company website URL"),
        ColumnSchema("description", "object", nullable=True, description="Business description (TEXT)"),
        ColumnSchema("raw_json", "object", nullable=True, description="Original JSON response"),
        ColumnSchema("fetched_at", "datetime64[ns]", nullable=False, description="Fetch timestamp"),
    ],
    primary_key=["ticker"]
)

BRONZE_NEWS = TableSchema(
    name="bronze_market_news",
    version="1.0",
    description="Raw market news from multiple sources (JSON + Text)",
    columns=[
        ColumnSchema("news_id", "object", nullable=False, description="Unique news identifier"),
        ColumnSchema("title", "object", nullable=False, description="News headline"),
        ColumnSchema("summary", "object", nullable=True, description="News summary"),
        ColumnSchema("content", "object", nullable=True, description="Full article text (TEXT)"),
        ColumnSchema("source", "object", nullable=True, description="News source"),
        ColumnSchema("url", "object", nullable=True, description="Article URL"),
        ColumnSchema("published_at", "datetime64[ns]", nullable=True, description="Publication time"),
        ColumnSchema("tickers", "object", nullable=True, description="Related tickers (JSON array)"),
        ColumnSchema("sentiment_score", "float64", nullable=True, description="Sentiment -1 to 1"),
        ColumnSchema("sentiment_label", "object", nullable=True, description="positive/negative/neutral"),
        ColumnSchema("raw_json", "object", nullable=True, description="Original JSON response"),
        ColumnSchema("fetched_at", "datetime64[ns]", nullable=False, description="Fetch timestamp"),
    ],
    primary_key=["news_id"]
)

BRONZE_ECONOMIC = TableSchema(
    name="bronze_economic_indicators",
    version="1.0",
    description="Raw economic indicators from FRED (CSV format)",
    columns=[
        ColumnSchema("date", "datetime64[ns]", nullable=False, description="Observation date"),
        ColumnSchema("indicator_id", "object", nullable=False, description="FRED series ID"),
        ColumnSchema("indicator_name", "object", nullable=True, description="Indicator full name"),
        ColumnSchema("value", "float64", nullable=True, description="Indicator value"),
        ColumnSchema("unit", "object", nullable=True, description="Measurement unit"),
        ColumnSchema("frequency", "object", nullable=True, description="Data frequency"),
        ColumnSchema("source_file", "object", nullable=True, description="Original CSV filename"),
        ColumnSchema("fetched_at", "datetime64[ns]", nullable=False, description="Fetch timestamp"),
    ],
    primary_key=["date", "indicator_id"]
)

BRONZE_BENCHMARKS = TableSchema(
    name="bronze_benchmarks",
    version="1.0",
    description="Raw benchmark data (SPY, QQQ, VIX) from yfinance",
    columns=[
        ColumnSchema("date", "datetime64[ns]", nullable=False, description="Trading date"),
        ColumnSchema("ticker", "object", nullable=False, description="Benchmark ticker"),
        ColumnSchema("open", "float64", nullable=False, description="Opening price"),
        ColumnSchema("high", "float64", nullable=False, description="Highest price"),
        ColumnSchema("low", "float64", nullable=False, description="Lowest price"),
        ColumnSchema("close", "float64", nullable=False, description="Closing price"),
        ColumnSchema("volume", "int64", nullable=False, description="Trading volume"),
        ColumnSchema("fetched_at", "datetime64[ns]", nullable=False, description="Fetch timestamp"),
    ],
    primary_key=["date", "ticker"]
)


# =============================================================================
# SILVER LAYER SCHEMAS
# =============================================================================

SILVER_PRICES = TableSchema(
    name="silver_enriched_stocks",
    version="1.0",
    description="Cleaned and enriched stock data with metadata joined",
    columns=[
        ColumnSchema("date", "datetime64[ns]", nullable=False, description="Trading date (lowercase)"),
        ColumnSchema("ticker", "object", nullable=False, description="Stock symbol (lowercase)"),
        ColumnSchema("open", "float64", nullable=False, description="Opening price (validated > 0)"),
        ColumnSchema("high", "float64", nullable=False, description="Highest price (validated >= low)"),
        ColumnSchema("low", "float64", nullable=False, description="Lowest price (validated)"),
        ColumnSchema("close", "float64", nullable=False, description="Closing price (validated > 0)"),
        ColumnSchema("volume", "int64", nullable=False, description="Trading volume (validated >= 0)"),
        ColumnSchema("daily_return", "float64", nullable=True, description="Daily return %"),
        ColumnSchema("sector", "object", nullable=False, description="GICS Sector ('Unknown' if missing)"),
        ColumnSchema("industry", "object", nullable=False, description="GICS Industry ('Unknown' if missing)"),
        ColumnSchema("market_cap_category", "object", nullable=True, description="Large/Mid/Small"),
        ColumnSchema("spy_return", "float64", nullable=True, description="SPY daily return for beta calc"),
        ColumnSchema("enriched_at", "datetime64[ns]", nullable=False, description="Processing timestamp"),
        ColumnSchema("data_version", "object", nullable=False, description="Version string"),
    ],
    primary_key=["date", "ticker"]
)

SILVER_NEWS = TableSchema(
    name="silver_processed_news",
    version="1.0",
    description="Processed news with extracted entities and aggregated sentiment",
    columns=[
        ColumnSchema("date", "datetime64[ns]", nullable=False, description="News date"),
        ColumnSchema("ticker", "object", nullable=False, description="Related stock ticker"),
        ColumnSchema("news_count", "int64", nullable=False, description="Number of news articles"),
        ColumnSchema("avg_sentiment", "float64", nullable=True, description="Average sentiment score"),
        ColumnSchema("positive_count", "int64", nullable=True, description="Count of positive news"),
        ColumnSchema("negative_count", "int64", nullable=True, description="Count of negative news"),
        ColumnSchema("neutral_count", "int64", nullable=True, description="Count of neutral news"),
        ColumnSchema("headlines", "object", nullable=True, description="Concatenated headlines (TEXT)"),
        ColumnSchema("processed_at", "datetime64[ns]", nullable=False, description="Processing timestamp"),
    ],
    primary_key=["date", "ticker"]
)

SILVER_ECONOMIC = TableSchema(
    name="silver_economic_indicators",
    version="1.0",
    description="Normalized economic indicators in wide format",
    columns=[
        ColumnSchema("date", "datetime64[ns]", nullable=False, description="Observation date"),
        ColumnSchema("gdp", "float64", nullable=True, description="GDP value"),
        ColumnSchema("cpi", "float64", nullable=True, description="CPI value"),
        ColumnSchema("unemployment_rate", "float64", nullable=True, description="Unemployment %"),
        ColumnSchema("fed_funds_rate", "float64", nullable=True, description="Fed Funds Rate %"),
        ColumnSchema("treasury_10y", "float64", nullable=True, description="10-Year Treasury %"),
        ColumnSchema("vix", "float64", nullable=True, description="VIX Index"),
        ColumnSchema("processed_at", "datetime64[ns]", nullable=False, description="Processing timestamp"),
    ],
    primary_key=["date"]
)


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

SCHEMAS = {
    # Bronze layer
    'bronze_prices': BRONZE_PRICES,
    'bronze_metadata': BRONZE_METADATA,
    'bronze_news': BRONZE_NEWS,
    'bronze_economic': BRONZE_ECONOMIC,
    'bronze_benchmarks': BRONZE_BENCHMARKS,
    
    # Silver layer
    'silver_prices': SILVER_PRICES,
    'silver_news': SILVER_NEWS,
    'silver_economic': SILVER_ECONOMIC,
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_schema(df: pd.DataFrame, schema_name: str) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame against a registered schema
    
    Args:
        df: DataFrame to validate
        schema_name: Name of schema in registry
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    if schema_name not in SCHEMAS:
        return False, [f"Schema '{schema_name}' not found in registry"]
    
    schema = SCHEMAS[schema_name]
    return schema.validate(df)


def get_schema_info(schema_name: str) -> Optional[Dict]:
    """Get schema information as dictionary"""
    if schema_name not in SCHEMAS:
        return None
    return SCHEMAS[schema_name].to_dict()


def list_schemas() -> List[str]:
    """List all registered schemas"""
    return list(SCHEMAS.keys())


def print_schema(schema_name: str):
    """Print schema documentation"""
    if schema_name not in SCHEMAS:
        print(f"Schema '{schema_name}' not found")
        return
    
    schema = SCHEMAS[schema_name]
    print(f"\n{'=' * 70}")
    print(f"Schema: {schema.name} (v{schema.version})")
    print(f"{'=' * 70}")
    print(f"Description: {schema.description}")
    print(f"\nColumns:")
    for col in schema.columns:
        nullable = "NULL" if col.nullable else "NOT NULL"
        print(f"  - {col.name}: {col.dtype} ({nullable})")
        if col.description:
            print(f"    └─ {col.description}")
    
    if schema.primary_key:
        print(f"\nPrimary Key: {schema.primary_key}")
    if schema.partition_by:
        print(f"Partition By: {schema.partition_by}")
    print()


# =============================================================================
# MAIN (for testing/documentation)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCHEMA REGISTRY - QUANT DATA PLATFORM")
    print("=" * 70)
    
    print("\nRegistered Schemas:")
    for name in list_schemas():
        schema = SCHEMAS[name]
        print(f"  - {name}: {len(schema.columns)} columns (v{schema.version})")
    
    # Print each schema
    for name in list_schemas():
        print_schema(name)
    
    print("=" * 70)
