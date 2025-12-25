"""
Delta Lake Helper Functions (Windows Compatible)
Using deltalake-rs (Rust-based, no Spark required)

This module provides helper functions to:
- Write pandas DataFrames to Delta Lake format
- Read Delta Tables to pandas
- Support Time Travel (version history)
- List table versions

Works on Windows without needing Java/Spark!

Install: pip install deltalake

Usage:
    from utils.delta_helper import pandas_to_delta, delta_to_pandas
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import deltalake (Rust-based, Windows compatible)
try:
    import deltalake
    from deltalake import DeltaTable, write_deltalake
    DELTA_AVAILABLE = True
    logger.info(f"[OK] Delta Lake (deltalake-rs) version: {deltalake.__version__}")
except ImportError:
    DELTA_AVAILABLE = False
    logger.warning("[WARN] deltalake not installed. Install with: pip install deltalake")


# =============================================================================
# PANDAS <-> DELTA CONVERSION
# =============================================================================

def pandas_to_delta(
    df: pd.DataFrame, 
    path: Union[str, Path], 
    mode: str = "overwrite",
    partition_by: Optional[List[str]] = None
) -> str:
    """
    Save pandas DataFrame as Delta Table
    
    Args:
        df: pandas DataFrame to save
        path: Output path for Delta Table
        mode: 'overwrite', 'append', or 'error'
        partition_by: Optional list of columns to partition by
        
    Returns:
        str: Path to the saved Delta Table
        
    Example:
        >>> pandas_to_delta(df, './data/bronze/prices_delta')
    """
    if not DELTA_AVAILABLE:
        raise ImportError("deltalake not installed. Run: pip install deltalake")
    
    import pyarrow as pa
    
    path = str(path)
    
    logger.info(f"Saving DataFrame ({len(df):,} rows) to Delta Table: {path}")
    
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert pandas to PyArrow Table (required for deltalake 1.x)
    table = pa.Table.from_pandas(df)
    
    # Write to Delta (compatible with deltalake 1.x)
    write_deltalake(
        path,
        table,
        mode=mode,
        partition_by=partition_by
    )
    
    logger.info(f"[OK] Saved to Delta Table: {path}")
    
    return path


def delta_to_pandas(
    path: Union[str, Path],
    version: Optional[int] = None
) -> pd.DataFrame:
    """
    Read Delta Table to pandas DataFrame
    
    Supports Time Travel via version number
    
    Args:
        path: Path to Delta Table
        version: Optional version number (for Time Travel)
        
    Returns:
        pandas DataFrame
        
    Example:
        >>> df = delta_to_pandas('./data/bronze/prices_delta')
        >>> df_v1 = delta_to_pandas('./data/bronze/prices_delta', version=0)
    """
    if not DELTA_AVAILABLE:
        raise ImportError("deltalake not installed. Run: pip install deltalake")
    
    path = str(path)
    
    if version is not None:
        logger.info(f"Reading Delta Table (version {version}): {path}")
        dt = DeltaTable(path, version=version)
    else:
        logger.info(f"Reading Delta Table (latest): {path}")
        dt = DeltaTable(path)
    
    df = dt.to_pandas()
    
    logger.info(f"[OK] Loaded {len(df):,} rows from Delta Table")
    
    return df


# =============================================================================
# DELTA TABLE OPERATIONS
# =============================================================================

def get_delta_table(path: Union[str, Path]) -> "DeltaTable":
    """
    Get DeltaTable object for a path
    """
    if not DELTA_AVAILABLE:
        raise ImportError("deltalake not installed")
    return DeltaTable(str(path))


def show_history(path: Union[str, Path], limit: int = 10) -> pd.DataFrame:
    """
    Show transaction history of a Delta Table
    
    Args:
        path: Path to Delta Table
        limit: Number of versions to show
        
    Returns:
        pandas DataFrame with version history
    """
    dt = get_delta_table(path)
    history = dt.history(limit=limit)
    
    history_df = pd.DataFrame(history)
    
    logger.info(f"\n--- Delta Table History: {path} ---")
    if len(history_df) > 0:
        display_cols = ['version', 'timestamp', 'operation']
        available_cols = [c for c in display_cols if c in history_df.columns]
        print(history_df[available_cols].to_string(index=False))
    else:
        print("  No history available")
    
    return history_df


def get_version_count(path: Union[str, Path]) -> int:
    """
    Get the number of versions in a Delta Table
    """
    dt = get_delta_table(path)
    history = dt.history()
    return len(history)


def get_table_info(path: Union[str, Path]) -> dict:
    """
    Get metadata about a Delta Table
    """
    dt = get_delta_table(path)
    
    info = {
        'path': str(path),
        'version': dt.version(),
        'num_files': len(dt.files()),
        'schema': dt.schema().to_pyarrow(),
    }
    
    return info


def is_delta_table(path: Union[str, Path]) -> bool:
    """
    Check if a path is a valid Delta Table
    """
    try:
        DeltaTable(str(path))
        return True
    except:
        return False


def convert_parquet_to_delta(
    parquet_path: Union[str, Path],
    delta_path: Union[str, Path]
) -> str:
    """
    Convert existing Parquet file to Delta Table
    """
    logger.info(f"Converting Parquet to Delta: {parquet_path} -> {delta_path}")
    
    # Load Parquet
    df = pd.read_parquet(parquet_path)
    
    # Save as Delta
    pandas_to_delta(df, delta_path)
    
    logger.info(f"[OK] Converted to Delta Table: {delta_path}")
    
    return str(delta_path)


def vacuum_table(path: Union[str, Path], retention_hours: int = 168, dry_run: bool = True):
    """
    Vacuum Delta Table to remove old files
    
    Args:
        path: Path to Delta Table
        retention_hours: Hours to retain (default 7 days)
        dry_run: If True, just list files that would be deleted
    """
    dt = get_delta_table(path)
    
    if dry_run:
        files = dt.vacuum(retention_hours=retention_hours, dry_run=True)
        logger.info(f"Would delete {len(files)} files")
        return files
    else:
        files = dt.vacuum(retention_hours=retention_hours, dry_run=False)
        logger.info(f"[OK] Vacuumed {len(files)} files from: {path}")
        return files


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DELTA LAKE HELPER - STATUS CHECK (Windows Compatible)")
    print("=" * 70)
    
    if DELTA_AVAILABLE:
        print(f"[OK] deltalake version: {deltalake.__version__}")
        
        # Test with sample data
        try:
            test_df = pd.DataFrame({
                'id': [1, 2, 3],
                'value': ['a', 'b', 'c'],
                'timestamp': [datetime.now()] * 3
            })
            
            test_path = './data/test_delta'
            
            # Test write
            pandas_to_delta(test_df, test_path)
            print(f"[OK] Test write successful: {test_path}")
            
            # Test read
            read_df = delta_to_pandas(test_path)
            print(f"[OK] Test read successful: {len(read_df)} rows")
            
            # Test history
            show_history(test_path)
            
            # Cleanup
            import shutil
            shutil.rmtree(test_path, ignore_errors=True)
            print("[OK] Test cleanup complete")
            
        except Exception as e:
            print(f"[ERR] Error: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("[ERR] Delta Lake not available")
        print("  Install with: pip install deltalake")
    
    print("=" * 70)
