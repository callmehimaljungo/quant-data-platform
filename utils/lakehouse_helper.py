"""
Data Lakehouse Helper Functions (Windows Compatible)
Using DuckDB for ACID transactions and data lakehouse features

DuckDB provides:
- ACID transactions
- SQL interface
- Parquet read/write with metadata
- Schema tracking
- Versioning via metadata tables

This is a Windows-friendly alternative to Delta Lake.

Usage:
    from utils.lakehouse_helper import pandas_to_lakehouse, lakehouse_to_pandas
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict
import pandas as pd
import logging
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import DuckDB
try:
    import duckdb
    LAKEHOUSE_AVAILABLE = True
    logger.info(f"✓ DuckDB version: {duckdb.__version__}")
except ImportError:
    LAKEHOUSE_AVAILABLE = False
    logger.warning("⚠️ DuckDB not installed. Install with: pip install duckdb")


# =============================================================================
# METADATA MANAGEMENT
# =============================================================================

def get_metadata_path(table_path: Union[str, Path]) -> Path:
    """Get metadata file path for a table"""
    table_path = Path(table_path)
    return table_path.parent / f".{table_path.name}_metadata.json"


def load_metadata(table_path: Union[str, Path]) -> Dict:
    """Load metadata for a table"""
    meta_path = get_metadata_path(table_path)
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {
        'versions': [],
        'current_version': 0,
        'created_at': None,
        'updated_at': None
    }


def save_metadata(table_path: Union[str, Path], metadata: Dict):
    """Save metadata for a table"""
    meta_path = get_metadata_path(table_path)
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


# =============================================================================
# PANDAS <-> LAKEHOUSE CONVERSION
# =============================================================================

def pandas_to_lakehouse(
    df: pd.DataFrame, 
    path: Union[str, Path], 
    mode: str = "overwrite",
    partition_by: Optional[List[str]] = None
) -> str:
    """
    Save pandas DataFrame to Lakehouse format (Parquet with versioning)
    
    Args:
        df: pandas DataFrame to save
        path: Output path for table
        mode: 'overwrite' or 'append'
        partition_by: Optional list of columns to partition by
        
    Returns:
        str: Path to the saved table
        
    Example:
        >>> pandas_to_lakehouse(df, './data/bronze/prices_lakehouse')
    """
    if not LAKEHOUSE_AVAILABLE:
        raise ImportError("DuckDB not installed. Run: pip install duckdb")
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving DataFrame ({len(df):,} rows) to Lakehouse: {path}")
    
    # Load existing metadata
    metadata = load_metadata(path)
    
    # Determine version
    if mode == "overwrite":
        new_version = 0
    else:
        new_version = metadata['current_version'] + 1
    
    # Generate versioned filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = path / f"data_v{new_version}_{timestamp}.parquet"
    
    # Save with DuckDB (handles complex types better)
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    
    # Register DataFrame and save as Parquet
    con.register('df_table', df)
    con.execute(f"COPY df_table TO '{data_file}' (FORMAT PARQUET, COMPRESSION 'snappy')")
    con.close()
    
    # Update metadata
    version_info = {
        'version': new_version,
        'timestamp': datetime.now().isoformat(),
        'file': str(data_file.name),
        'rows': len(df),
        'columns': df.columns.tolist(),
        'operation': mode
    }
    
    if mode == "overwrite":
        metadata['versions'] = [version_info]
    else:
        metadata['versions'].append(version_info)
    
    metadata['current_version'] = new_version
    metadata['updated_at'] = datetime.now().isoformat()
    if metadata['created_at'] is None:
        metadata['created_at'] = datetime.now().isoformat()
    
    save_metadata(path, metadata)
    
    logger.info(f"✓ Saved to Lakehouse (version {new_version}): {path}")
    
    return str(path)


def lakehouse_to_pandas(
    path: Union[str, Path],
    version: Optional[int] = None
) -> pd.DataFrame:
    """
    Read Lakehouse table to pandas DataFrame
    
    Supports Time Travel via version number
    
    Args:
        path: Path to Lakehouse table
        version: Optional version number (for Time Travel)
        
    Returns:
        pandas DataFrame
        
    Example:
        >>> df = lakehouse_to_pandas('./data/bronze/prices_lakehouse')
        >>> df_v0 = lakehouse_to_pandas('./data/bronze/prices_lakehouse', version=0)
    """
    if not LAKEHOUSE_AVAILABLE:
        raise ImportError("DuckDB not installed. Run: pip install duckdb")
    
    path = Path(path)
    
    # Load metadata
    metadata = load_metadata(path)
    
    if not metadata['versions']:
        raise FileNotFoundError(f"No data found in: {path}")
    
    # Get version info
    if version is not None:
        version_info = next((v for v in metadata['versions'] if v['version'] == version), None)
        if not version_info:
            raise ValueError(f"Version {version} not found. Available: {[v['version'] for v in metadata['versions']]}")
        logger.info(f"Reading Lakehouse (version {version}): {path}")
    else:
        version_info = metadata['versions'][-1]  # Latest
        logger.info(f"Reading Lakehouse (latest v{version_info['version']}): {path}")
    
    # Read Parquet file
    data_file = path / version_info['file']
    
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{data_file}')").fetchdf()
    con.close()
    
    logger.info(f"✓ Loaded {len(df):,} rows from Lakehouse")
    
    return df


# =============================================================================
# LAKEHOUSE OPERATIONS
# =============================================================================

def show_history(path: Union[str, Path], limit: int = 10) -> pd.DataFrame:
    """
    Show transaction history of a Lakehouse table
    """
    path = Path(path)
    metadata = load_metadata(path)
    
    history_df = pd.DataFrame(metadata['versions'][-limit:])
    
    logger.info(f"\n--- Lakehouse History: {path} ---")
    if len(history_df) > 0:
        print(history_df[['version', 'timestamp', 'operation', 'rows']].to_string(index=False))
    else:
        print("  No history available")
    
    return history_df


def get_version_count(path: Union[str, Path]) -> int:
    """Get the number of versions"""
    metadata = load_metadata(path)
    return len(metadata['versions'])


def get_table_info(path: Union[str, Path]) -> dict:
    """Get metadata about a Lakehouse table"""
    path = Path(path)
    metadata = load_metadata(path)
    
    if metadata['versions']:
        latest = metadata['versions'][-1]
        info = {
            'path': str(path),
            'version': latest['version'],
            'rows': latest['rows'],
            'columns': latest['columns'],
            'created_at': metadata['created_at'],
            'updated_at': metadata['updated_at']
        }
    else:
        info = {'path': str(path), 'version': None}
    
    return info


def is_lakehouse_table(path: Union[str, Path]) -> bool:
    """Check if a path is a valid Lakehouse table"""
    meta_path = get_metadata_path(path)
    return meta_path.exists()


def convert_parquet_to_lakehouse(
    parquet_path: Union[str, Path],
    lakehouse_path: Union[str, Path]
) -> str:
    """Convert existing Parquet file to Lakehouse table"""
    logger.info(f"Converting Parquet to Lakehouse: {parquet_path} -> {lakehouse_path}")
    
    df = pd.read_parquet(parquet_path)
    pandas_to_lakehouse(df, lakehouse_path)
    
    logger.info(f"✓ Converted to Lakehouse: {lakehouse_path}")
    return str(lakehouse_path)


# =============================================================================
# ALIAS FOR COMPATIBILITY (delta names -> lakehouse)
# =============================================================================

# For backwards compatibility
pandas_to_delta = pandas_to_lakehouse
delta_to_pandas = lakehouse_to_pandas
is_delta_table = is_lakehouse_table
convert_parquet_to_delta = convert_parquet_to_lakehouse
DELTA_AVAILABLE = LAKEHOUSE_AVAILABLE


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LAKEHOUSE HELPER - STATUS CHECK (Windows Compatible)")
    print("=" * 70)
    
    if LAKEHOUSE_AVAILABLE:
        print(f"✓ DuckDB version: {duckdb.__version__}")
        
        # Test with sample data
        try:
            test_df = pd.DataFrame({
                'id': [1, 2, 3],
                'value': ['a', 'b', 'c'],
                'timestamp': [datetime.now()] * 3
            })
            
            test_path = Path('./data/test_lakehouse')
            
            # Test write
            pandas_to_lakehouse(test_df, test_path)
            print(f"✓ Test write successful: {test_path}")
            
            # Test read
            read_df = lakehouse_to_pandas(test_path)
            print(f"✓ Test read successful: {len(read_df)} rows")
            
            # Test append
            pandas_to_lakehouse(test_df, test_path, mode="append")
            print("✓ Test append successful")
            
            # Test history
            show_history(test_path)
            
            # Test time travel
            df_v0 = lakehouse_to_pandas(test_path, version=0)
            print(f"✓ Time travel (v0): {len(df_v0)} rows")
            
            # Show table info
            info = get_table_info(test_path)
            print(f"✓ Table info: {info}")
            
            # Cleanup
            import shutil
            shutil.rmtree(test_path, ignore_errors=True)
            print("✓ Test cleanup complete")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ DuckDB not available")
        print("  Install with: pip install duckdb")
    
    print("=" * 70)
