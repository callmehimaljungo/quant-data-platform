"""Bronze Layer Module - Raw data ingestion from multiple sources"""

# Price data ingestion
from .ingest import ingest_all_stocks, save_to_bronze

# Metadata ingestion (JSON format from yfinance)
from .metadata_loader import load_stock_metadata

# News ingestion (JSON + Text format)
from .news_loader import load_market_news

# Economic indicators (CSV format from FRED)
from .economic_loader import load_economic_data

# Benchmark data (Parquet format from yfinance)
from .benchmark_loader import load_benchmark_data

__all__ = [
    # Price data
    'ingest_all_stocks',
    'save_to_bronze',
    # Metadata
    'load_stock_metadata',
    # News
    'load_market_news',
    # Economic
    'load_economic_data',
    # Benchmarks
    'load_benchmark_data'
]