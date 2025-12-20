# 🔄 SO SÁNH CODE: PARQUET → DELTA LAKE

## TÓM TẮT

**Thay đổi tối thiểu, hiệu quả tối đa!**

```
Parquet (hiện tại)        →    Delta Lake (Lakehouse)
─────────────────────────      ───────────────────────────
✅ Giữ nguyên 95% code           ✅ Chỉ sửa read/write
✅ Logic không đổi                ✅ Thêm ACID, Time Travel
✅ Schema validation giữ          ✅ Transaction log miễn phí
```

---

## 📦 DEPENDENCIES

### Hiện Tại (Parquet)
```txt
# requirements.txt - HIỆN TẠI
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0  # Cho Parquet
kaggle>=1.5.0
```

### Sau Migration (Delta Lake)
```txt
# requirements.txt - SAU KHI MIGRATE
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0       # Vẫn cần (Delta dùng Parquet)
delta-spark>=2.4.0    # THÊM MỚI
pyspark>=3.4.0        # THÊM MỚI
kaggle>=1.5.0
```

**Thay đổi:** Chỉ thêm 2 packages!

---

## 🔧 HELPER FUNCTIONS (Tạo 1 lần, dùng mãi)

```python
# utils/delta_helper.py - FILE MỚI

"""
Delta Lake Helper Functions
Tạo 1 lần, dùng cho tất cả layers
"""

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import pandas as pd

def get_spark_session(app_name="QuanPlatform"):
    """
    Initialize Spark with Delta Lake support
    
    Chỉ cần gọi 1 lần khi bắt đầu script
    """
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "4g")  # Tùy chỉnh theo RAM
    
    return configure_spark_with_delta_pip(builder).getOrCreate()

def pandas_to_delta(df_pandas, path, mode="overwrite"):
    """
    Save pandas DataFrame as Delta Table
    
    Args:
        df_pandas: pandas DataFrame
        path: Output path (e.g., './data/bronze/prices_delta')
        mode: 'overwrite' or 'append'
    """
    spark = get_spark_session()
    spark_df = spark.createDataFrame(df_pandas)
    
    spark_df.write.format("delta") \
        .mode(mode) \
        .option("overwriteSchema", "true") \
        .save(path)
    
    return path

def delta_to_pandas(path, version=None, timestamp=None):
    """
    Read Delta Table to pandas DataFrame
    
    Args:
        path: Delta Table path
        version: Optional version number (for Time Travel)
        timestamp: Optional timestamp (for Time Travel)
    
    Returns:
        pandas DataFrame
    """
    spark = get_spark_session()
    
    reader = spark.read.format("delta")
    
    # Time Travel support
    if version is not None:
        reader = reader.option("versionAsOf", version)
    elif timestamp is not None:
        reader = reader.option("timestampAsOf", timestamp)
    
    spark_df = reader.load(path)
    return spark_df.toPandas()
```

---

## 📄 BRONZE LAYER

### BEFORE (Parquet)

```python
# bronze/ingest.py - HIỆN TẠI

import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def ingest_from_kaggle():
    """Download from Kaggle"""
    # ... logic download ... (GIỮ NGUYÊN)
    df = pd.read_csv('./temp/all_stock_data.csv')
    return df

def save_to_bronze(df):
    """Save to Bronze layer"""
    output_path = './data/bronze/prices.parquet'
    
    # Add metadata
    df['ingested_at'] = datetime.now()
    
    # Save as Parquet
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df)} rows to {output_path}")

def main():
    df = ingest_from_kaggle()
    save_to_bronze(df)

if __name__ == "__main__":
    main()
```

### AFTER (Delta Lake)

```python
# bronze/ingest.py - SAU KHI MIGRATE

import pandas as pd
import logging
from datetime import datetime
from utils.delta_helper import pandas_to_delta  # THÊM MỚI

logger = logging.getLogger(__name__)

def ingest_from_kaggle():
    """Download from Kaggle"""
    # ... logic download ... (GIỮ NGUYÊN - KHÔNG ĐỔI)
    df = pd.read_csv('./temp/all_stock_data.csv')
    return df

def save_to_bronze(df):
    """Save to Bronze layer as Delta Table"""
    output_path = './data/bronze/prices_delta'  # Đổi path
    
    # Add metadata (GIỮ NGUYÊN)
    df['ingested_at'] = datetime.now()
    
    # Save as Delta Table (THAY 1 DÒNG)
    pandas_to_delta(df, output_path, mode="overwrite")
    
    logger.info(f"Saved {len(df)} rows to Delta Table: {output_path}")
    
    # THÊM: Log transaction history
    from delta.tables import DeltaTable
    spark = get_spark_session()
    deltaTable = DeltaTable.forPath(spark, output_path)
    logger.info("Latest transactions:")
    deltaTable.history().show(5)

def main():
    df = ingest_from_kaggle()  # GIỮ NGUYÊN
    save_to_bronze(df)         # GIỮ NGUYÊN

if __name__ == "__main__":
    main()
```

**THAY ĐỔI:**
- ✏️ Import helper function
- ✏️ Đổi path (thêm `_delta`)
- ✏️ Thay `.to_parquet()` → `pandas_to_delta()`
- ➕ Thêm log transaction history (optional)

**GIỮ NGUYÊN:**
- ✅ Kaggle download logic
- ✅ Schema validation
- ✅ Metadata columns
- ✅ Logging

---

## 📄 SILVER LAYER

### BEFORE (Parquet)

```python
# silver/clean.py - HIỆN TẠI

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_bronze():
    """Load from Bronze"""
    return pd.read_parquet('./data/bronze/prices.parquet')

def clean_data(df):
    """Clean and enrich data"""
    # Deduplicate (GIỮ NGUYÊN)
    df = df.drop_duplicates(subset=['ticker', 'date'])
    
    # Quality gates (GIỮ NGUYÊN)
    if (df['close'] <= 0).any():
        raise ValueError("Invalid prices detected!")
    
    # Calculate daily return (GIỮ NGUYÊN)
    df = df.sort_values(['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    
    return df

def save_to_silver(df):
    """Save to Silver layer"""
    output_path = './data/silver/enriched_stocks.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

def main():
    df = load_bronze()
    df_clean = clean_data(df)
    save_to_silver(df_clean)

if __name__ == "__main__":
    main()
```

### AFTER (Delta Lake)

```python
# silver/clean.py - SAU KHI MIGRATE

import pandas as pd
import logging
from utils.delta_helper import delta_to_pandas, pandas_to_delta  # THÊM

logger = logging.getLogger(__name__)

def load_bronze():
    """Load from Bronze Delta Table"""
    return delta_to_pandas('./data/bronze/prices_delta')  # ĐỔI 1 DÒNG

def clean_data(df):
    """Clean and enrich data"""
    # Deduplicate (GIỮ NGUYÊN - KHÔNG ĐỔI)
    df = df.drop_duplicates(subset=['ticker', 'date'])
    
    # Quality gates (GIỮ NGUYÊN - KHÔNG ĐỔI)
    if (df['close'] <= 0).any():
        raise ValueError("Invalid prices detected!")
    
    # Calculate daily return (GIỮ NGUYÊN - KHÔNG ĐỔI)
    df = df.sort_values(['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    
    return df

def save_to_silver(df):
    """Save to Silver layer as Delta Table"""
    output_path = './data/silver/enriched_stocks_delta'  # Đổi path
    pandas_to_delta(df, output_path, mode="overwrite")  # ĐỔI 1 DÒNG
    logger.info(f"Saved to Delta Table: {output_path}")

def main():
    df = load_bronze()          # GIỮ NGUYÊN
    df_clean = clean_data(df)   # GIỮ NGUYÊN
    save_to_silver(df_clean)    # GIỮ NGUYÊN

if __name__ == "__main__":
    main()
```

**THAY ĐỔI:**
- ✏️ `pd.read_parquet()` → `delta_to_pandas()`
- ✏️ `df.to_parquet()` → `pandas_to_delta()`
- ✏️ Đổi path

**GIỮ NGUYÊN:**
- ✅ Deduplication logic
- ✅ Quality gates
- ✅ Daily return calculation
- ✅ Tất cả business logic

---

## 📄 GOLD LAYER (Sector Analysis Example)

### BEFORE (Parquet)

```python
# gold/sector_analysis.py - HIỆN TẠI

import pandas as pd
import numpy as np

def load_silver():
    return pd.read_parquet('./data/silver/enriched_stocks.parquet')

def calculate_sector_performance(df):
    """Calculate sector metrics"""
    # Sector average return (GIỮ NGUYÊN)
    sector_perf = df.groupby(['date', 'sector'])['daily_return'].mean()
    
    # Sector volatility (GIỮ NGUYÊN)
    sector_vol = df.groupby('sector')['daily_return'].std() * np.sqrt(252)
    
    return sector_perf, sector_vol

def save_results(sector_perf, sector_vol):
    sector_perf.to_parquet('./data/gold/sector_performance.parquet')
    sector_vol.to_parquet('./data/gold/sector_volatility.parquet')

def main():
    df = load_silver()
    perf, vol = calculate_sector_performance(df)
    save_results(perf, vol)

if __name__ == "__main__":
    main()
```

### AFTER (Delta Lake)

```python
# gold/sector_analysis.py - SAU KHI MIGRATE

import pandas as pd
import numpy as np
from utils.delta_helper import delta_to_pandas, pandas_to_delta  # THÊM

def load_silver():
    return delta_to_pandas('./data/silver/enriched_stocks_delta')  # ĐỔI

def calculate_sector_performance(df):
    """Calculate sector metrics"""
    # Sector average return (GIỮ NGUYÊN - KHÔNG ĐỔI)
    sector_perf = df.groupby(['date', 'sector'])['daily_return'].mean()
    
    # Sector volatility (GIỮ NGUYÊN - KHÔNG ĐỔI)
    sector_vol = df.groupby('sector')['daily_return'].std() * np.sqrt(252)
    
    return sector_perf, sector_vol

def save_results(sector_perf, sector_vol):
    # Convert Series to DataFrame for Delta
    perf_df = sector_perf.reset_index()
    vol_df = sector_vol.reset_index()
    
    pandas_to_delta(perf_df, './data/gold/sector_performance_delta')  # ĐỔI
    pandas_to_delta(vol_df, './data/gold/sector_volatility_delta')    # ĐỔI

def main():
    df = load_silver()                          # GIỮ NGUYÊN
    perf, vol = calculate_sector_performance(df)  # GIỮ NGUYÊN
    save_results(perf, vol)                     # GIỮ NGUYÊN

if __name__ == "__main__":
    main()
```

**THAY ĐỔI:**
- ✏️ Read/Write functions
- ✏️ Paths

**GIỮ NGUYÊN:**
- ✅ All calculation logic
- ✅ Business formulas
- ✅ GroupBy operations

---

## 🎁 BONUS: TIME TRAVEL (TIER 2 Feature)

### Code Mới Thêm (Không Sửa Code Cũ)

```python
# utils/time_travel.py - FILE MỚI

from utils.delta_helper import delta_to_pandas, get_spark_session
from delta.tables import DeltaTable

def compare_versions(table_path, version1, version2):
    """
    So sánh 2 versions của Delta Table
    
    Use case: Xem data thay đổi như thế nào sau mỗi lần update
    """
    df_v1 = delta_to_pandas(table_path, version=version1)
    df_v2 = delta_to_pandas(table_path, version=version2)
    
    # Find differences
    new_rows = len(df_v2) - len(df_v1)
    print(f"Version {version1} → {version2}:")
    print(f"  Rows changed: {new_rows:+d}")
    
    return df_v1, df_v2

def rollback_to_version(table_path, target_version):
    """
    Rollback Delta Table về version cũ
    
    Use case: Phát hiện lỗi, cần quay lại version trước
    """
    spark = get_spark_session()
    deltaTable = DeltaTable.forPath(spark, table_path)
    
    # Restore to target version
    deltaTable.restoreToVersion(target_version)
    
    print(f"✓ Rolled back to version {target_version}")

def show_history(table_path, num_versions=10):
    """
    Xem lịch sử thay đổi của Delta Table
    
    Use case: Audit trail, biết ai làm gì khi nào
    """
    spark = get_spark_session()
    deltaTable = DeltaTable.forPath(spark, table_path)
    
    history = deltaTable.history(num_versions)
    history.select("version", "timestamp", "operation", "operationMetrics").show()
    
    return history

# DEMO Usage
if __name__ == "__main__":
    table_path = './data/silver/enriched_stocks_delta'
    
    # 1. Xem lịch sử
    print("=== Transaction History ===")
    show_history(table_path)
    
    # 2. So sánh versions
    print("\n=== Compare Versions ===")
    df_v1, df_v2 = compare_versions(table_path, version1=0, version2=1)
    
    # 3. Rollback (nếu cần)
    # rollback_to_version(table_path, target_version=0)
```

---

## 📊 MIGRATION CHECKLIST

### Bước 1: Chuẩn Bị (30 phút)

```bash
# Cài packages
pip install delta-spark>=2.4.0 pyspark>=3.4.0

# Tạo helper file
# Copy code từ phần "HELPER FUNCTIONS" ở trên
mkdir utils
# Tạo utils/delta_helper.py
```

### Bước 2: Test Với Sample (1 giờ)

```python
# test_delta.py - File test

from utils.delta_helper import pandas_to_delta, delta_to_pandas
import pandas as pd

# Test với data nhỏ
df_test = pd.DataFrame({
    'ticker': ['AAPL', 'MSFT'],
    'close': [150.0, 250.0]
})

# Test write
pandas_to_delta(df_test, './test_delta')
print("✓ Write OK")

# Test read
df_read = delta_to_pandas('./test_delta')
print("✓ Read OK")
print(df_read)
```

### Bước 3: Migrate Bronze (1 giờ)

```bash
# Backup code cũ
cp bronze/ingest.py bronze/ingest_parquet_backup.py

# Sửa bronze/ingest.py theo template "AFTER" ở trên

# Test
python bronze/ingest.py
```

### Bước 4: Migrate Silver (1 giờ)

```bash
# Backup
cp silver/clean.py silver/clean_parquet_backup.py

# Sửa silver/clean.py

# Test
python silver/clean.py
```

### Bước 5: Migrate Gold (1-2 giờ)

```bash
# Sửa tất cả files trong gold/

# Test từng file
python gold/sector_analysis.py
python gold/risk_metrics.py
python gold/portfolio.py
```

### Bước 6: End-to-End Test (30 phút)

```bash
# Chạy full pipeline
python bronze/ingest.py
python silver/clean.py
python gold/sector_analysis.py

# Verify outputs
ls -lh data/bronze/
ls -lh data/silver/
ls -lh data/gold/
```

---

## 💡 TIPS & TRICKS

### Tip 1: Hybrid Approach (Giữ Cả Hai)

Nếu lo lắng, bạn có thể giữ cả Parquet và Delta:

```python
def save_to_bronze(df):
    """Save to both Parquet and Delta"""
    # Parquet (backup)
    df.to_parquet('./data/bronze/prices.parquet')
    
    # Delta (main)
    pandas_to_delta(df, './data/bronze/prices_delta')
    
    # Best of both worlds!
```

### Tip 2: Conditional Import

```python
# bronze/ingest.py - Smart import

try:
    from utils.delta_helper import pandas_to_delta
    USE_DELTA = True
except ImportError:
    USE_DELTA = False
    print("Delta Lake not available, using Parquet")

def save_to_bronze(df):
    if USE_DELTA:
        pandas_to_delta(df, './data/bronze/prices_delta')
    else:
        df.to_parquet('./data/bronze/prices.parquet')
```

### Tip 3: Gradual Migration

```
Week 1: Chỉ migrate Bronze → Test kỹ
Week 2: Migrate Silver → Test kỹ
Week 3: Migrate Gold → Test kỹ

→ Từ từ, chắc chắn hơn build lại từ đầu!
```

---

## 🎯 TÓM TẮT SO SÁNH

### Effort Required

| Task | Thời Gian | Độ Khó |
|------|-----------|--------|
| **Setup Delta** | 30 phút | ⭐ Dễ |
| **Create Helper Functions** | 1 giờ | ⭐⭐ Trung bình |
| **Migrate Bronze** | 1 giờ | ⭐ Dễ |
| **Migrate Silver** | 1 giờ | ⭐ Dễ |
| **Migrate Gold** | 2 giờ | ⭐⭐ Trung bình |
| **Testing** | 2 giờ | ⭐ Dễ |
| **TOTAL** | **~7-8 giờ** | **⭐⭐ Trung bình** |

### Code Changes

```
Total Files:      ~8 files
Changed Lines:    ~50 lines
New Lines:        ~100 lines (helper functions)
Business Logic:   0 changes! ✅

→ 95% code GIỮ NGUYÊN!
```

### Benefits

```
PARQUET (Hiện tại)          DELTA LAKE (Sau migrate)
─────────────────────       ──────────────────────────
❌ No ACID                  ✅ ACID transactions
❌ No versioning            ✅ Time Travel
❌ No metadata              ✅ Transaction log
❌ Overwrite only           ✅ Upsert/Merge
❌ No rollback              ✅ Rollback to any version
⚠️ Manual validation        ✅ Built-in constraints
⚠️ Risk of data loss        ✅ Safe concurrent writes

→ Upgrade XỨNG ĐÁNG với 7-8 giờ effort!
```

---

## ✅ KẾT LUẬN

**Migration từ Parquet → Delta Lake:**
- ⏰ Thời gian: 1-2 ngày
- 💪 Độ khó: Trung bình
- 📝 Code changes: Tối thiểu
- 🎁 Benefits: Rất lớn
- ⚠️ Risk: Thấp (có thể giữ Parquet backup)

**→ TOTALLY WORTH IT cho đồ án Lakehouse!**

---

**Ready to migrate? Let's do it! 🚀**
