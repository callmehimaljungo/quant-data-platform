"""Silver Layer Module - Data cleaning, processing, and schema joining"""

# Main price data cleaning
from .clean import clean_silver_data, save_to_silver

# Metadata processing (JSON flattening, sector standardization)
from .process_metadata import process_metadata

# News processing (text cleaning, ticker extraction, sentiment aggregation)
from .process_news import process_all_news

# Economic data processing (CSV pivot, interpolation, regime classification)
from .process_economic import process_economic

# Schema joining (unified dataset with all enrichments)
from .join_schemas import join_all_schemas

__all__ = [
    # Price cleaning
    'clean_silver_data',
    'save_to_silver',
    # Metadata
    'process_metadata',
    # News
    'process_all_news',
    # Economic
    'process_economic',
    # Schema joining
    'join_all_schemas'
]

