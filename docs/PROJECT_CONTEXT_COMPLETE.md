# PROJECT CONTEXT DOCUMENT - QUANT DATA PLATFORM
## Đồ Án Tốt Nghiệp: Data Lakehouse cho Phân Tích Chứng Khoán

---

# ⚠️ QUAN TRỌNG - ĐỌC TRƯỚC KHI CODE

**Document này chứa TẤT CẢ context cần thiết để AI sinh code chính xác.**

**Ngày cập nhật:** 2024-12-21  
**Version:** 2.0 (Complete)  
**Trạng thái:** Bronze ✅ | Silver ✅ | Gold ⚠️ (partial) | Backtest ❌ | Dashboard ❌

---

# 1. TỔNG QUAN DỰ ÁN

## 1.1 Mục Tiêu

Xây dựng **Data Lakehouse Platform** cho phân tích và quản lý danh mục đầu tư chứng khoán Mỹ.

## 1.2 Thông Số Quan Trọng

| Aspect | Detail |
|--------|--------|
| **Dataset** | 9,315 US stock tickers, 34.6M rows raw → 33.5M rows cleaned |
| **Date Range** | 1962-01-02 → 2024-11-04 |
| **Storage** | Local Parquet + DuckDB Lakehouse (Windows compatible) |
| **Processing** | Python + pandas + DuckDB |
| **Visualization** | Streamlit (chưa implement) |
| **Focus** | Business/Economic analysis - KHÔNG phải Toán phức tạp |

## 1.3 Điều KHÔNG LÀM ❌

```
❌ Machine Learning phức tạp (LSTM, Transformer, Neural Networks)
❌ Technical indicators với lý thuyết Toán sâu (không cần chứng minh công thức)
❌ High-frequency trading algorithms
❌ Real-time streaming (chỉ batch processing)
❌ Distributed computing (không dùng Spark cluster)
```

## 1.4 Điều CẦN LÀM ✅

```
✅ Business analytics (Sector, Market Cap, Risk Metrics)
✅ Portfolio allocation strategies (công thức có sẵn, không cần derive)
✅ Backtesting với evaluation rõ ràng
✅ Code có logging và error handling
✅ Comments giải thích ý nghĩa KINH TẾ (không cần giải thích Toán)
```

---

# 2. TRẠNG THÁI HIỆN TẠI

## 2.1 Files ĐÃ CÓ ✅

```
quant-data-platform/
├── config.py                    ✅ Configuration management
├── requirements.txt             ✅ Dependencies
├── README.md                    ✅ Documentation
│
├── bronze/
│   ├── __init__.py              ✅ Module init
│   ├── ingest.py                ✅ Kaggle ingestion (Parquet)
│   ├── ingest_delta.py          ✅ Lakehouse migration
│   └── test_bronze.py           ✅ Validation tests
│
├── silver/
│   ├── __init__.py              ✅ Module init
│   ├── clean.py                 ✅ Data cleaning (Parquet)
│   ├── clean_delta.py           ✅ Lakehouse processing
│   └── test_silver.py           ✅ Validation tests
│
├── gold/
│   ├── __init__.py              ✅ Module init
│   └── sector_analysis.py       ⚠️ Partial (thiếu nhiều metrics)
│
├── utils/
│   ├── __init__.py              ✅ Module init
│   ├── delta_helper.py          ✅ Delta Lake helper (legacy)
│   └── lakehouse_helper.py      ✅ DuckDB Lakehouse helper
│
├── data/
│   ├── bronze/
│   │   ├── all_stock_data.parquet    ✅ 917 MB, 34.6M rows
│   │   └── prices_lakehouse/         ✅ Lakehouse format
│   ├── silver/
│   │   ├── enriched_stocks.parquet   ✅ 992 MB, 33.5M rows
│   │   └── enriched_lakehouse/       ✅ Lakehouse format
│   └── gold/
│       ├── sector_metrics_lakehouse/ ✅ 26 sectors
│       ├── ticker_metrics_lakehouse/ ✅ 100 tickers
│       └── monthly_performance_lakehouse/ ✅ 755 months
│
└── docs/                        ✅ 10+ documentation files
```

## 2.2 Files CẦN TẠO MỚI ❌

```
quant-data-platform/
├── bronze/
│   └── metadata_loader.py       ❌ Fetch sector/industry từ yfinance
│
├── gold/
│   ├── risk_metrics.py          ❌ VaR, Sharpe, Max Drawdown, Beta
│   └── portfolio.py             ❌ Equal Weight, Risk Parity, Momentum
│
├── backtest/
│   ├── __init__.py              ❌ Module init
│   └── evaluate.py              ❌ Compare strategies vs SPY
│
└── dashboard/
    ├── __init__.py              ❌ Module init
    └── app.py                   ❌ Streamlit 4 tabs
```

---

# 3. DATA SCHEMA SPECIFICATIONS

## 3.1 Bronze Layer Schema

**File:** `data/bronze/all_stock_data.parquet`  
**Rows:** 34,646,258  
**Tickers:** 9,315

| Column | Type | Description | Nullable | Notes |
|--------|------|-------------|----------|-------|
| Date | datetime64[ns] | Trading date | NO | PascalCase từ Kaggle |
| Ticker | object | Stock symbol | NO | e.g., AAPL, MSFT |
| Open | float64 | Opening price | NO | USD |
| High | float64 | Highest price | NO | USD |
| Low | float64 | Lowest price | NO | USD |
| Close | float64 | Closing price | NO | USD |
| Volume | int64 | Trading volume | NO | Shares traded |
| Dividends | float64 | Dividends | YES | Optional |
| Stock Splits | float64 | Stock splits | YES | Optional |
| ingested_at | datetime64[ns] | Ingestion timestamp | NO | Added by Bronze |

**⚠️ LƯU Ý:** Kaggle data dùng **PascalCase** (Date, Ticker, Close)

## 3.2 Silver Layer Schema

**File:** `data/silver/enriched_stocks.parquet`  
**Rows:** 33,454,012 (sau cleaning)  
**Tickers:** 9,314

| Column | Type | Description | Nullable | Notes |
|--------|------|-------------|----------|-------|
| date | datetime64[ns] | Trading date | NO | **lowercase** |
| ticker | object | Stock symbol | NO | **lowercase** |
| open | float64 | Opening price | NO | Validated > 0 |
| high | float64 | Highest price | NO | Validated >= low |
| low | float64 | Lowest price | NO | Validated |
| close | float64 | Closing price | NO | Validated > 0 |
| volume | int64 | Trading volume | NO | Validated >= 0 |
| daily_return | float64 | Daily return % | NO | Calculated |
| sector | object | GICS Sector | NO | 'Unknown' if missing |
| industry | object | GICS Industry | NO | 'Unknown' if missing |
| enriched_at | datetime64[ns] | Processing timestamp | NO | Added by Silver |
| data_version | object | Version string | NO | 'silver_v1' |

**⚠️ LƯU Ý:** Silver layer chuyển sang **lowercase** columns

## 3.3 Gold Layer Outputs

### 3.3.1 Sector Metrics (ĐÃ CÓ)

**File:** `data/gold/sector_metrics_lakehouse/`

| Column | Type | Description |
|--------|------|-------------|
| sector | object | Sector name (e.g., 'Sector_A') |
| num_tickers | int64 | Number of stocks in sector |
| total_records | int64 | Total data points |
| avg_daily_return | float64 | Average daily return % |
| total_return | float64 | Cumulative return % |
| volatility | float64 | Annualized volatility % |
| sharpe_ratio | float64 | Risk-adjusted return |
| sortino_ratio | float64 | Downside risk-adjusted return |
| max_drawdown | float64 | Maximum drawdown % |

### 3.3.2 Risk Metrics (CẦN TẠO) ❌

**File:** `data/gold/risk_metrics.parquet`

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| ticker | object | Stock symbol | - |
| sector | object | GICS Sector | - |
| var_95 | float64 | Value at Risk 95% | `percentile(returns, 5)` |
| sharpe_ratio | float64 | Sharpe Ratio | `(mean - rf) / std * sqrt(252)` |
| sortino_ratio | float64 | Sortino Ratio | `(mean - rf) / downside_std * sqrt(252)` |
| max_drawdown | float64 | Max Drawdown % | `min((price - peak) / peak)` |
| volatility | float64 | Annualized Vol % | `std * sqrt(252) * 100` |
| beta | float64 | Market Beta | `cov(stock, SPY) / var(SPY)` |
| alpha | float64 | Jensen's Alpha | `stock_return - (rf + beta * (market - rf))` |

### 3.3.3 Portfolio Weights (CẦN TẠO) ❌

**File:** `data/gold/portfolio_weights.parquet`

| Column | Type | Description |
|--------|------|-------------|
| ticker | object | Stock symbol |
| sector | object | GICS Sector |
| weight_equal | float64 | Equal weight allocation |
| weight_risk_parity | float64 | Risk parity allocation |
| weight_momentum | float64 | Momentum-based allocation |
| return_30d | float64 | 30-day return (for momentum) |
| volatility | float64 | Historical volatility |

### 3.3.4 Backtest Results (CẦN TẠO) ❌

**File:** `data/gold/backtest_results.parquet`

| Column | Type | Description |
|--------|------|-------------|
| strategy | object | Strategy name |
| period | object | 'bull' / 'bear' / 'recovery' |
| start_date | datetime64 | Period start |
| end_date | datetime64 | Period end |
| total_return | float64 | Total return % |
| annualized_return | float64 | Annualized return % |
| volatility | float64 | Annualized volatility % |
| sharpe_ratio | float64 | Sharpe Ratio |
| max_drawdown | float64 | Max Drawdown % |
| win_rate | float64 | % days with positive return |
| calmar_ratio | float64 | Return / Max Drawdown |

---

# 4. BUSINESS FORMULAS - COPY CHÍNH XÁC

## 4.1 Return Calculations

```python
# Daily Return (Simple) - Dùng cho hầu hết calculations
daily_return = (close_today - close_yesterday) / close_yesterday
# HOẶC dùng pct_change:
df['daily_return'] = df.groupby('ticker')['close'].pct_change()

# Cumulative Return
cumulative_return = (1 + daily_return).cumprod() - 1
```

## 4.2 Risk Metrics - FORMULAS CHÍNH XÁC

```python
import numpy as np
import pandas as pd

# =============================================================================
# VALUE AT RISK (VaR) 95%
# =============================================================================
def calculate_var_95(daily_returns):
    """
    Value at Risk tại mức 95% confidence
    
    Ý nghĩa kinh tế: "Với 95% xác suất, loss trong 1 ngày không vượt quá |VaR|%"
    Ví dụ: VaR = -2% nghĩa là 95% ngày, bạn không mất quá 2%
    """
    return np.percentile(daily_returns.dropna(), 5)  # percentile 5 = VaR 95%


# =============================================================================
# SHARPE RATIO
# =============================================================================
def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.05):
    """
    Sharpe Ratio = Risk-adjusted return
    
    risk_free_rate = 5% per year (US Treasury rate)
    252 = số ngày giao dịch trong năm
    
    Ý nghĩa kinh tế: Return per unit of risk
    Benchmark: > 1.0 = good, > 2.0 = very good, > 3.0 = excellent
    """
    daily_rf = risk_free_rate / 252
    excess_return = daily_returns.mean() - daily_rf
    
    if daily_returns.std() == 0:
        return 0.0
    
    sharpe = (excess_return / daily_returns.std()) * np.sqrt(252)
    return sharpe


# =============================================================================
# SORTINO RATIO
# =============================================================================
def calculate_sortino_ratio(daily_returns, risk_free_rate=0.05):
    """
    Sortino Ratio = Chỉ xét downside volatility
    
    Ý nghĩa kinh tế: Giống Sharpe nhưng chỉ phạt khi loss, không phạt khi gain cao
    Hữu ích khi: Returns có positive skew (nhiều ngày gain lớn)
    """
    daily_rf = risk_free_rate / 252
    excess_return = daily_returns.mean() - daily_rf
    
    downside_returns = daily_returns[daily_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = (excess_return / downside_returns.std()) * np.sqrt(252)
    return sortino


# =============================================================================
# MAX DRAWDOWN
# =============================================================================
def calculate_max_drawdown(prices):
    """
    Maximum Drawdown = Worst peak-to-trough decline
    
    Ý nghĩa kinh tế: Nếu bạn mua tại đỉnh, bạn sẽ mất tối đa bao nhiêu %?
    Ví dụ: -20% drawdown = từ đỉnh xuống đáy mất 20%
    
    ⚠️ Input là PRICES, không phải returns
    """
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


# =============================================================================
# VOLATILITY (ANNUALIZED)
# =============================================================================
def calculate_volatility(daily_returns):
    """
    Annualized Volatility = Độ biến động theo năm
    
    252 = số ngày giao dịch trong năm
    
    Ý nghĩa kinh tế: Giá dao động bao nhiêu % trung bình mỗi năm?
    Benchmark: < 20% = low vol, 20-40% = medium, > 40% = high vol
    """
    return daily_returns.std() * np.sqrt(252)


# =============================================================================
# BETA (Market Sensitivity)
# =============================================================================
def calculate_beta(stock_returns, market_returns):
    """
    Beta = Sensitivity to market movements
    
    Ý nghĩa kinh tế:
    - Beta > 1: Biến động mạnh hơn thị trường (high risk, high reward)
    - Beta = 1: Biến động bằng thị trường
    - Beta < 1: Ít biến động hơn thị trường (defensive stock)
    - Beta < 0: Ngược chiều thị trường (rare, e.g., gold)
    
    ⚠️ market_returns = SPY (S&P 500 ETF) returns
    """
    # Align the two series
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    
    if len(aligned) < 30:  # Need minimum data points
        return np.nan
    
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    market_variance = aligned.iloc[:, 1].var()
    
    if market_variance == 0:
        return np.nan
    
    return covariance / market_variance
```

## 4.3 Portfolio Allocation Strategies

```python
import numpy as np

# =============================================================================
# STRATEGY 1: EQUAL WEIGHT
# =============================================================================
def equal_weight(n_stocks):
    """
    Đơn giản nhất: Chia đều tiền cho tất cả stocks
    
    w_i = 1/N cho mỗi stock
    
    WHY: Baseline strategy, không cần prediction
    WHEN: Khi không có view đặc biệt về stock nào tốt hơn
    """
    return np.ones(n_stocks) / n_stocks


# =============================================================================
# STRATEGY 2: RISK PARITY
# =============================================================================
def risk_parity(volatilities):
    """
    Phân bổ ngược với volatility
    
    w_i = (1/σ_i) / Σ(1/σ_j)
    
    WHY: Stock rủi ro cao → weight thấp hơn
    WHEN: Muốn cân bằng risk contribution từ mỗi asset
    
    Ví dụ:
    - Stock A: vol = 20% → weight cao
    - Stock B: vol = 40% → weight thấp
    
    ⚠️ volatilities phải là array of floats > 0
    """
    # Handle edge cases
    volatilities = np.array(volatilities)
    volatilities = np.where(volatilities <= 0, 0.01, volatilities)  # Min 1%
    
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    
    return weights


# =============================================================================
# STRATEGY 3: MOMENTUM
# =============================================================================
def momentum_weights(returns_30d, top_pct=0.2, bottom_pct=0.2):
    """
    Overweight recent winners, underweight losers
    
    Logic: "Winners keep winning" - behavioral finance
    
    Allocation:
    - Top 20% by 30-day return → weight = 2x
    - Bottom 20% → weight = 0.5x
    - Middle 60% → weight = 1x
    
    WHY: Momentum effect is well-documented in finance literature
    WHEN: Trending markets
    CAUTION: Performs POORLY in mean-reverting markets
    
    ⚠️ returns_30d phải là pandas Series hoặc array
    """
    returns_30d = np.array(returns_30d)
    n = len(returns_30d)
    
    # Rank by percentile
    ranks = pd.Series(returns_30d).rank(pct=True)
    
    # Assign weights
    weights = np.where(
        ranks > (1 - top_pct), 2.0,      # Top 20% → 2x weight
        np.where(ranks < bottom_pct, 0.5, 1.0)  # Bottom 20% → 0.5x, middle → 1x
    )
    
    # Normalize to sum = 1
    return weights / weights.sum()
```

## 4.4 Backtest Periods

```python
BACKTEST_PERIODS = {
    'bull_market': {
        'start': '2021-01-01',
        'end': '2021-12-31',
        'description': 'Post-COVID rally, strong growth stocks'
    },
    'bear_market': {
        'start': '2022-01-01',
        'end': '2022-12-31',
        'description': 'Fed rate hikes, inflation fears, tech crash'
    },
    'recovery': {
        'start': '2023-01-01',
        'end': '2024-06-30',
        'description': 'Market stabilization, AI boom (NVDA, etc.)'
    }
}

BENCHMARK = 'SPY'  # S&P 500 ETF - Tất cả strategies phải so sánh với SPY
```

---

# 5. GICS SECTORS & MARKET CAP

## 5.1 GICS Sectors (11 categories)

```python
GICS_SECTORS = [
    'Technology',              # AAPL, MSFT, NVDA
    'Healthcare',              # JNJ, PFE, UNH
    'Financials',              # JPM, BAC, GS (Lưu ý: Finance → Financials)
    'Consumer Discretionary',  # AMZN, TSLA, HD (Consumer Cyclical)
    'Communication Services',  # GOOGL, META, NFLX
    'Industrials',             # BA, CAT, UPS
    'Consumer Staples',        # PG, KO, WMT (Consumer Defensive)
    'Energy',                  # XOM, CVX, COP
    'Utilities',               # NEE, DUK, SO
    'Real Estate',             # AMT, PLD, CCI
    'Materials'                # LIN, APD, ECL (Basic Materials)
]
```

## 5.2 Market Cap Classification

```python
def classify_market_cap(market_cap_usd):
    """
    Standard market cap classification (theo S&P guidelines)
    """
    if market_cap_usd >= 10_000_000_000:   # >= $10B
        return 'Large Cap'
    elif market_cap_usd >= 2_000_000_000:  # >= $2B
        return 'Mid Cap'
    else:
        return 'Small Cap'
```

---

# 6. CODE CONVENTIONS - BẮT BUỘC TUÂN THỦ

## 6.1 File Structure

```python
"""
Module description - 1-2 sentences mô tả module làm gì
Author: Quant Data Platform Team
Date: YYYY-MM-DD

Purpose:
- Bullet points explaining what this module does
- Business context for thesis defense
"""

# =============================================================================
# IMPORTS (theo thứ tự: Standard → Third-party → Local)
# =============================================================================
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

# Add project root to path (BẮT BUỘC cho mọi file)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, LOG_FORMAT

# =============================================================================
# LOGGING SETUP (BẮT BUỘC)
# =============================================================================
import logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (định nghĩa ở đầu file)
# =============================================================================
INPUT_PATH = SILVER_DIR / 'enriched_stocks.parquet'
OUTPUT_PATH = GOLD_DIR / 'risk_metrics.parquet'

# =============================================================================
# FUNCTIONS (docstring BẮT BUỘC)
# =============================================================================
def example_function(param1: pd.DataFrame, param2: float = 0.05) -> pd.DataFrame:
    """
    Mô tả ngắn gọn function làm gì.
    
    Args:
        param1: Description of param1
        param2: Description with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
        
    Example:
        >>> result = example_function(df, 0.03)
    """
    pass

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    logger.info("=" * 70)
    logger.info("MODULE NAME STARTED")
    logger.info("=" * 70)
    
    try:
        # Processing logic
        pass
        
        logger.info("=" * 70)
        logger.info("✓✓✓ MODULE NAME COMPLETED ✓✓✓")
        logger.info("=" * 70)
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
```

## 6.2 Logging Requirements - BẮT BUỘC

```python
# PHẢI log các thông tin sau:
logger.info(f"Loaded {len(df):,} rows from {source}")
logger.info(f"Processed {n_tickers:,} tickers")
logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
logger.info(f"Removed {n_duplicates:,} duplicates")
logger.warning(f"Missing data for {n_missing:,} tickers")
logger.error(f"Quality gate failed: {reason}")

# Format số lớn với comma separator
logger.info(f"Total rows: {len(df):,}")  # Output: "Total rows: 33,454,012"
```

## 6.3 Error Handling - KHÔNG ĐƯỢC SILENT FAIL

```python
# ❌ SAI - Silent fail
try:
    df = pd.read_parquet(file)
except:
    pass

# ✅ ĐÚNG - Explicit error with logging
try:
    df = pd.read_parquet(file)
    logger.info(f"✓ Loaded {len(df):,} rows from {file}")
except FileNotFoundError:
    logger.error(f"File not found: {file}")
    raise
except Exception as e:
    logger.error(f"Failed to load {file}: {str(e)}")
    raise
```

## 6.4 Quality Gates - FAIL LOUD

```python
def quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality checks - RAISE EXCEPTION if fail
    """
    # Check 1: Close > 0
    invalid = df[df['close'] <= 0]
    if len(invalid) > 0:
        logger.error(f"Quality Gate FAILED: {len(invalid)} rows with close <= 0")
        raise ValueError(f"Quality Gate FAILED: {len(invalid)} rows with close <= 0")
    
    logger.info("✓ Quality Gate PASSED: All close prices > 0")
    return df
```

---

# 7. ANTI-PATTERNS - TUYỆT ĐỐI KHÔNG LÀM

## 7.1 Code Anti-patterns

```python
# ❌ KHÔNG: Import * (gây conflict và khó debug)
from pandas import *
from numpy import *

# ✅ ĐÚNG: Import cụ thể
import pandas as pd
import numpy as np


# ❌ KHÔNG: Hardcode paths
df = pd.read_parquet('C:/Users/name/project/data/file.parquet')

# ✅ ĐÚNG: Dùng config.py
from config import BRONZE_DIR
df = pd.read_parquet(BRONZE_DIR / 'file.parquet')


# ❌ KHÔNG: Magic numbers không có comment
result = returns * 252 ** 0.5

# ✅ ĐÚNG: Explain business meaning
# 252 = trading days per year (standard for annualization)
result = returns * np.sqrt(252)


# ❌ KHÔNG: Silent exception handling
try:
    process_data()
except:
    pass

# ✅ ĐÚNG: Log and re-raise
try:
    process_data()
except Exception as e:
    logger.error(f"Failed to process: {str(e)}")
    raise
```

## 7.2 Data Anti-patterns

```python
# ❌ KHÔNG: Modify Bronze data directly
bronze_df['close'] = bronze_df['close'].fillna(0)  # NEVER modify Bronze!

# ✅ ĐÚNG: Transform in Silver layer
silver_df = bronze_df.copy()
silver_df['close'] = silver_df['close'].fillna(method='ffill')


# ❌ KHÔNG: Mix column naming conventions
df['Date'] = ...  # PascalCase
df['ticker'] = ... # lowercase
df['Daily_Return'] = ... # Mixed

# ✅ ĐÚNG: Consistent lowercase in Silver onwards
df['date'] = ...
df['ticker'] = ...
df['daily_return'] = ...


# ❌ KHÔNG: Calculate metrics without handling edge cases
sharpe = returns.mean() / returns.std()  # Crashes if std = 0

# ✅ ĐÚNG: Handle edge cases
if returns.std() == 0:
    sharpe = 0.0
else:
    sharpe = returns.mean() / returns.std()
```

## 7.3 Business Logic Anti-patterns

```python
# ❌ KHÔNG: Use indicators without explaining WHY
df['RSI'] = calculate_rsi(df, 14)  # Why 14?

# ✅ ĐÚNG: Comment business meaning
# RSI 14-day: Industry standard period (J. Welles Wilder, 1978)
# RSI > 70: Overbought - consider reducing position
# RSI < 30: Oversold - consider buying opportunity
df['RSI'] = calculate_rsi(df, period=14)


# ❌ KHÔNG: Calculate Beta without market data
beta = calculate_beta(stock_returns)  # What market?

# ✅ ĐÚNG: Explicitly use SPY as market proxy
# SPY = S&P 500 ETF, standard benchmark for US stocks
spy_returns = load_spy_returns()
beta = calculate_beta(stock_returns, spy_returns)
```

---

# 8. LAKEHOUSE HELPER USAGE

## 8.1 Import Pattern

```python
# Từ bất kỳ file nào trong project
from utils import (
    pandas_to_lakehouse,
    lakehouse_to_pandas,
    show_history,
    is_lakehouse_table
)
```

## 8.2 Write to Lakehouse

```python
from utils import pandas_to_lakehouse

# Save DataFrame to Lakehouse format
output_path = GOLD_DIR / 'risk_metrics_lakehouse'
pandas_to_lakehouse(df, output_path, mode="overwrite")

# mode options:
# - "overwrite": Replace all data (default)
# - "append": Add new data (increments version)
```

## 8.3 Read from Lakehouse

```python
from utils import lakehouse_to_pandas

# Read latest version
df = lakehouse_to_pandas(SILVER_DIR / 'enriched_lakehouse')

# Time Travel - read specific version
df_v0 = lakehouse_to_pandas(SILVER_DIR / 'enriched_lakehouse', version=0)
```

## 8.4 Check Lakehouse Status

```python
from utils import is_lakehouse_table, show_history

# Check if path is valid Lakehouse table
if is_lakehouse_table(path):
    # Show version history
    show_history(path, limit=5)
```

---

# 9. FILE-BY-FILE IMPLEMENTATION GUIDE

## 9.1 gold/risk_metrics.py (CẦN TẠO)

```python
"""
INPUT:  data/silver/enriched_stocks.parquet (hoặc enriched_lakehouse)
OUTPUT: data/gold/risk_metrics_lakehouse/

CALCULATIONS:
1. Load Silver data
2. Calculate per-ticker metrics:
   - VaR 95%
   - Sharpe Ratio
   - Sortino Ratio
   - Max Drawdown
   - Volatility
   - Beta (vs SPY)
3. Calculate per-sector aggregates
4. Save to Lakehouse

⚠️ SPY data: Cần tách SPY từ dataset (ticker == 'SPY')
"""
```

## 9.2 gold/portfolio.py (CẦN TẠO)

```python
"""
INPUT:  data/gold/risk_metrics_lakehouse/
OUTPUT: data/gold/portfolio_weights_lakehouse/

CALCULATIONS:
1. Load risk metrics
2. Filter to tradeable universe (e.g., top 100 by liquidity)
3. Calculate weights for 3 strategies:
   - Equal Weight: 1/N
   - Risk Parity: 1/volatility normalized
   - Momentum: Based on 30-day returns
4. Save weights to Lakehouse

⚠️ Weights phải sum = 1.0 cho mỗi strategy
"""
```

## 9.3 backtest/evaluate.py (CẦN TẠO)

```python
"""
INPUT:  
- data/gold/portfolio_weights_lakehouse/
- data/silver/enriched_lakehouse/

OUTPUT: data/gold/backtest_results_lakehouse/

PROCESS:
1. Load weights and price data
2. For each period (bull, bear, recovery):
   a. Filter data to period
   b. For each strategy:
      - Calculate daily portfolio returns
      - Calculate metrics (total return, Sharpe, drawdown, etc.)
3. Compare all strategies vs SPY benchmark
4. Save results to Lakehouse

EXPECTED OUTPUT FORMAT:
| strategy     | period   | total_return | sharpe | max_drawdown |
|--------------|----------|--------------|--------|--------------|
| equal_weight | bull     | 25%          | 0.8    | -10%         |
| risk_parity  | bull     | 20%          | 0.9    | -8%          |
| momentum     | bull     | 35%          | 0.7    | -15%         |
| SPY          | bull     | 28%          | 0.75   | -12%         |
"""
```

## 9.4 dashboard/app.py (CẦN TẠO)

```python
"""
Streamlit Dashboard với 4 tabs:

TAB 1: SECTOR OVERVIEW
- Heatmap: Sector performance (color = return)
- Bar chart: Sector YTD returns
- Selector: Period (1M, 3M, YTD, 1Y)

TAB 2: RISK ANALYSIS
- Table: Risk metrics by sector
- Scatter plot: Return vs Volatility
- Selector: Filter by market cap

TAB 3: PORTFOLIO ALLOCATION
- Pie chart: Weights per strategy
- Line chart: Cumulative return comparison
- Selector: Strategy

TAB 4: STOCK DETAIL
- Candlestick chart
- Price + SMA overlay
- Risk metrics for selected ticker
- Selector: Ticker

LOAD DATA FROM:
- data/gold/sector_metrics_lakehouse/
- data/gold/risk_metrics_lakehouse/
- data/gold/portfolio_weights_lakehouse/
- data/gold/backtest_results_lakehouse/
"""
```

---

# 10. DEFENSE TALKING POINTS

## 10.1 Kiến Trúc

> "Em triển khai **Medallion Architecture** - Bronze/Silver/Gold layers theo best practice của Databricks. Đây là kiến trúc chuẩn cho Data Lakehouse."

## 10.2 Tại Sao Lakehouse?

> "Lakehouse kết hợp ưu điểm của Data Warehouse (ACID, schema enforcement) với Data Lake (flexibility, low cost). Em dùng DuckDB để đạt được ACID transactions và versioning mà không cần Spark cluster."

## 10.3 Risk Metrics

> "Em tính các metrics chuẩn công nghiệp: Sharpe Ratio (risk-adjusted return), VaR 95% (maximum expected loss), và Beta (market sensitivity). Đây là những metrics mà quỹ đầu tư thực tế sử dụng."

## 10.4 Portfolio Strategies

> "3 strategies em implement đều có cơ sở học thuật:
> - Equal Weight: Baseline đơn giản
> - Risk Parity: Bridgewater Associates (Ray Dalio) nổi tiếng với approach này
> - Momentum: Hiệu ứng momentum được document trong nhiều paper học thuật (Jegadeesh & Titman, 1993)"

---

# 11. QUICK REFERENCE

## 11.1 Path Constants

```python
from config import (
    PROJECT_ROOT,    # Root của project
    DATA_DIR,        # data/
    BRONZE_DIR,      # data/bronze/
    SILVER_DIR,      # data/silver/
    GOLD_DIR,        # data/gold/
    METADATA_DIR,    # data/metadata/
    LOG_FORMAT       # Logging format string
)
```

## 11.2 Common Imports

```python
# Standard template cho mọi file
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT
from utils import pandas_to_lakehouse, lakehouse_to_pandas

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
```

## 11.3 Risk-Free Rate

```python
# US Treasury rate (standard assumption)
RISK_FREE_RATE = 0.05  # 5% per year
DAILY_RF = RISK_FREE_RATE / 252  # Daily rate
```

## 11.4 Trading Days

```python
TRADING_DAYS_PER_YEAR = 252  # Standard for US markets
```

---

# END OF DOCUMENT

**Khi đưa document này cho AI, thêm yêu cầu cụ thể ở cuối.**

**Ví dụ:**
```
[Paste toàn bộ document này]

---

Yêu cầu: Tạo file gold/risk_metrics.py theo specifications ở Section 9.1
```
