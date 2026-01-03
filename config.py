"""
Configuration Management
Centralized configuration for R2, paths, and system settings
"""

import os
from pathlib import Path
from typing import Dict

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'

# Layer directories
BRONZE_DIR = DATA_DIR / 'bronze'
SILVER_DIR = DATA_DIR / 'silver'
GOLD_DIR = DATA_DIR / 'gold'
METADATA_DIR = DATA_DIR / 'metadata'
BACKTEST_DIR = PROJECT_ROOT / 'backtest' / 'results'

# Create directories if they don't exist
for directory in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, METADATA_DIR, BACKTEST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# R2 STORAGE CONFIGURATION (Section 8)
# =============================================================================
R2_CONFIG = {
    'endpoint': os.environ.get('R2_ENDPOINT', ''),
    'access_key': os.environ.get('R2_ACCESS_KEY', ''),
    'secret_key': os.environ.get('R2_SECRET_KEY', ''),
    'bucket': os.environ.get('R2_BUCKET', ''),
}

# R2 Paths 
R2_PATHS = {
    'raw_prices': 'raw/prices/',
    'bronze': 'processed/bronze/',
    'silver': 'processed/silver/',
    'gold': 'processed/gold/',
    'backtest': 'results/backtest/'
}

# =============================================================================
# DATA SCHEMA 
# =============================================================================
PRICE_DATA_SCHEMA = {
    'date': 'datetime64[ns]',
    'ticker': 'object',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64'
}

METADATA_SCHEMA = {
    'ticker': 'object',
    'company_name': 'object',
    'sector': 'object',
    'industry': 'object',
    'market_cap': 'float64',
    'market_cap_category': 'object',
    'exchange': 'object'
}

# =============================================================================
# GICS SECTORS  - Official GICS Standard Names
# =============================================================================
GICS_SECTORS = [
    'Information Technology',   # AAPL, MSFT, NVDA (or 'Technology' for compatibility)
    'Health Care',              # JNJ, PFE, UNH
    'Financials',               # JPM, BAC, GS (NOT 'Finance')
    'Consumer Discretionary',   # AMZN, TSLA, HD (NOT 'Consumer Cyclical')
    'Communication Services',   # GOOGL, META, NFLX
    'Industrials',              # BA, CAT, UPS
    'Consumer Staples',         # PG, KO, WMT (NOT 'Consumer Defensive')
    'Energy',                   # XOM, CVX, COP
    'Utilities',                # NEE, DUK, SO
    'Real Estate',              # AMT, PLD, CCI
    'Materials'                 # LIN, APD, ECL (NOT 'Basic Materials')
]

# Alternative sector names (for data source compatibility)
SECTOR_ALIASES = {
    'Technology': 'Information Technology',
    'Healthcare': 'Health Care',
    'Finance': 'Financials',
    'Consumer Cyclical': 'Consumer Discretionary',
    'Consumer Defensive': 'Consumer Staples',
    'Basic Materials': 'Materials',
}

# Market cap categories
MARKET_CAP_THRESHOLDS = {
    'Large': 10_000_000_000,   # > $10B
    'Mid': 2_000_000_000,      # $2B - $10B
    'Mid': 2_000_000_000,      # $2B - $10B
    'Small': 0                 # < $2B
}

# =============================================================================
# REAL-TIME CONFIGURATION
# =============================================================================
# Realtime data ingestion settings
REALTIME_PRICE_INTERVAL = 900  # 15 minutes (was 300 = 5 mins, reduced for efficiency)
REALTIME_NEWS_INTERVAL = 900    # 15 minutes
REALTIME_MAX_ITERATIONS = None  # Infinite loop (or set int for testing)

# =============================================================================
# QUALITY GATES 
# =============================================================================
BRONZE_QUALITY_CHECKS = [
    'schema_validation',
    'null_check_critical_columns'
]

SILVER_QUALITY_CHECKS = [
    'close > 0',
    'high >= low',
    'volume >= 0',
    'no_null_dates'
]

# =============================================================================
# BACKTEST CONFIGURATION (Section 6)
# =============================================================================
BACKTEST_CONFIG = {
    'benchmark': 'SPY',  # S&P 500 ETF
    'initial_capital': 100_000,
    'commission_rate': 0.001,  # 0.1% per trade
    'slippage': 0.0005,        # 0.05% slippage
}

# Evaluation metrics 
BACKTEST_METRICS = [
    'total_return',
    'annualized_return',
    'volatility',
    'sharpe_ratio',
    'max_drawdown',
    'win_rate',
    'calmar_ratio'
]

# =============================================================================
# LOGGING CONFIGURATION 
# =============================================================================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def validate_r2_config() -> bool:
    """
    Validate R2 configuration is complete
    
    Returns:
        bool: True if all R2 credentials are set
    """
    required = ['endpoint', 'access_key', 'secret_key', 'bucket']
    missing = [key for key in required if not R2_CONFIG[key]]
    
    if missing:
        print(f"WARNING: Missing R2 configuration: {missing}")
        return False
    
    return True


def get_market_cap_category(market_cap: float) -> str:
    """
    Categorize stock by market cap
    
    Args:
        market_cap: Market capitalization in USD
        
    Returns:
        str: 'Large', 'Mid', or 'Small'
    """
    if market_cap >= MARKET_CAP_THRESHOLDS['Large']:
        return 'Large'
    elif market_cap >= MARKET_CAP_THRESHOLDS['Mid']:
        return 'Mid'
    else:
        return 'Small'


def get_output_path(layer: str, filename: str) -> Path:
    """
    Get standardized output path for a layer
    
    Args:
        layer: 'bronze', 'silver', or 'gold'
        filename: Output filename
        
    Returns:
        Path: Full path to output file
    """
    layer_map = {
        'bronze': BRONZE_DIR,
        'silver': SILVER_DIR,
        'gold': GOLD_DIR,
        'metadata': METADATA_DIR
    }
    
    if layer not in layer_map:
        raise ValueError(f"Invalid layer: {layer}. Must be one of {list(layer_map.keys())}")
    
    return layer_map[layer] / filename


# =============================================================================
# STARTUP VALIDATION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)
    
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    
    print("\nDirectory Structure:")
    for name, path in [
        ('Bronze', BRONZE_DIR),
        ('Silver', SILVER_DIR),
        ('Gold', GOLD_DIR),
        ('Metadata', METADATA_DIR),
        ('Backtest', BACKTEST_DIR)
    ]:
        exists = "[OK]" if path.exists() else "[FAIL]"
        print(f"  {exists} {name}: {path}")
    
    print("\nR2 Configuration:")
    r2_valid = validate_r2_config()
    if r2_valid:
        print("  [OK] All R2 credentials configured")
    else:
        print("  [FAIL] R2 credentials incomplete - check environment variables")
    
    print("\nGICS Sectors:")
    for sector in GICS_SECTORS:
        print(f"  - {sector}")
    
    print("\n" + "=" * 70)
