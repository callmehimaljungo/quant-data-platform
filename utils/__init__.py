"""Utils module for Quant Data Platform"""

# Use DuckDB-based Lakehouse helper (Windows compatible)
from .lakehouse_helper import (
    pandas_to_lakehouse,
    lakehouse_to_pandas,
    show_history,
    convert_parquet_to_lakehouse,
    is_lakehouse_table,
    get_table_info,
    get_version_count,
    get_metadata_path,
    LAKEHOUSE_AVAILABLE,
    # Aliases for backward compatibility with Delta Lake naming
    pandas_to_delta,
    delta_to_pandas,
    is_delta_table,
    convert_parquet_to_delta,
    DELTA_AVAILABLE
)

# Schema Registry
from .schema_registry import (
    SCHEMAS,
    validate_schema,
    get_schema_info,
    list_schemas,
    print_schema
)

# Ticker Universe
from .ticker_universe import (
    TickerUniverse,
    get_universe,
    register_tickers_from_bronze,
    get_analyzed_tickers
)

__all__ = [
    # Lakehouse functions
    'pandas_to_lakehouse',
    'lakehouse_to_pandas',
    'show_history',
    'convert_parquet_to_lakehouse',
    'is_lakehouse_table',
    'get_table_info',
    'get_version_count',
    'get_metadata_path',
    'LAKEHOUSE_AVAILABLE',
    # Delta aliases
    'pandas_to_delta',
    'delta_to_pandas',
    'is_delta_table',
    'convert_parquet_to_delta',
    'DELTA_AVAILABLE',
    # Schema Registry
    'SCHEMAS',
    'validate_schema',
    'get_schema_info',
    'list_schemas',
    'print_schema',
    # Ticker Universe
    'TickerUniverse',
    'get_universe',
    'register_tickers_from_bronze',
    'get_analyzed_tickers'
]

# R2 Sync (lazy import to avoid boto3 requirement)
def get_r2_sync():
    """Get R2 sync module (requires boto3)"""
    from . import r2_sync
    return r2_sync

