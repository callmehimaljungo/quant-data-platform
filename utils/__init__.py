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
    LAKEHOUSE_AVAILABLE,
    # Aliases for backward compatibility with Delta Lake naming
    pandas_to_delta,
    delta_to_pandas,
    is_delta_table,
    convert_parquet_to_delta,
    DELTA_AVAILABLE
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
    'LAKEHOUSE_AVAILABLE',
    # Delta aliases
    'pandas_to_delta',
    'delta_to_pandas',
    'is_delta_table',
    'convert_parquet_to_delta',
    'DELTA_AVAILABLE'
]
