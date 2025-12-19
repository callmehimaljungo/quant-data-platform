# Quant Data Platform - Bronze Layer

## 📋 Overview

Bronze Layer thực hiện **raw data ingestion** từ Cloudflare R2 storage, theo kiến trúc Medallion.

**Nhiệm vụ chính:**
- Load dữ liệu OHLCV của 9000+ US stocks từ R2
- Validate schema theo Section 3.1
- **KHÔNG transform** data (giữ nguyên raw)
- Add metadata: `ingested_at` timestamp
- Quality checks: schema validation + null checks

## 🏗️ Architecture

```
R2 Storage (raw/prices/)
    ↓
Bronze Layer (bronze/ingest.py)
    ↓
./data/bronze/prices.parquet
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd quant-data-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your R2 credentials
```

### 2. Configure R2 Credentials

Edit `.env` file:
```bash
R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY=your_access_key
R2_SECRET_KEY=your_secret_key
R2_BUCKET=your_bucket_name
```

### 3. Validate Configuration

```bash
python config.py
```

Expected output:
```
======================================================================
CONFIGURATION VALIDATION
======================================================================

Project Root: /path/to/quant-data-platform
Data Directory: /path/to/quant-data-platform/data

Directory Structure:
  ✓ Bronze: /path/to/quant-data-platform/data/bronze
  ✓ Silver: /path/to/quant-data-platform/data/silver
  ✓ Gold: /path/to/quant-data-platform/data/gold
  ✓ Metadata: /path/to/quant-data-platform/data/metadata
  ✓ Backtest: /path/to/quant-data-platform/backtest/results

R2 Configuration:
  ✓ All R2 credentials configured
```

### 4. Run Bronze Layer Ingestion

```bash
python bronze/ingest.py
```

## 📊 Expected Output

### Console Logs

```
2024-01-20 10:00:00 - bronze.ingest - INFO - ======================================================================
2024-01-20 10:00:00 - bronze.ingest - INFO - BRONZE LAYER INGESTION STARTED
2024-01-20 10:00:00 - bronze.ingest - INFO - ======================================================================
2024-01-20 10:00:05 - bronze.ingest - INFO - Successfully connected to R2 storage
2024-01-20 10:00:10 - bronze.ingest - INFO - Found 9315 parquet files to process
2024-01-20 10:01:00 - bronze.ingest - INFO - Progress: 100/9315 files loaded
...
2024-01-20 10:15:00 - bronze.ingest - INFO - Total rows loaded: 2,500,000
2024-01-20 10:15:00 - bronze.ingest - INFO - Total unique tickers: 9,315
2024-01-20 10:15:00 - bronze.ingest - INFO - Date range: 1962-01-02 to 2025-04-02
2024-01-20 10:15:05 - bronze.ingest - INFO - Schema validation PASSED: All checks successful
2024-01-20 10:15:10 - bronze.ingest - INFO - Data saved to ./data/bronze/prices.parquet
2024-01-20 10:15:10 - bronze.ingest - INFO - File size: 450.23 MB
2024-01-20 10:15:10 - bronze.ingest - INFO - ======================================================================
2024-01-20 10:15:10 - bronze.ingest - INFO - BRONZE LAYER INGESTION COMPLETED SUCCESSFULLY
2024-01-20 10:15:10 - bronze.ingest - INFO - Duration: 910.50 seconds
2024-01-20 10:15:10 - bronze.ingest - INFO - ======================================================================
```

### Output Files

```
data/
└── bronze/
    └── prices.parquet  (450+ MB)
```

### Data Schema

```python
import pandas as pd

df = pd.read_parquet('./data/bronze/prices.parquet')
print(df.info())

# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2500000 entries, 0 to 2499999
# Data columns (total 8 columns):
#  #   Column        Non-Null Count    Dtype         
# ---  ------        --------------    -----         
#  0   date          2500000 non-null  datetime64[ns]
#  1   ticker        2500000 non-null  object        
#  2   open          2500000 non-null  float64       
#  3   high          2500000 non-null  float64       
#  4   low           2500000 non-null  float64       
#  5   close         2500000 non-null  float64       
#  6   volume        2500000 non-null  int64         
#  7   ingested_at   2500000 non-null  datetime64[ns]
```

## 🔍 Data Validation

Bronze layer thực hiện các checks sau:

### 1. Schema Validation
- ✅ All required columns present: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`
- ✅ Correct data types (float64 for prices, int64 for volume, datetime for date)

### 2. Quality Checks
- ✅ No nulls in critical columns: `date`, `ticker`, `close`
- ⚠️  Other columns có thể có nulls (sẽ clean ở Silver layer)

### 3. Metadata
- ✅ `ingested_at` timestamp added to track when data was loaded

## 🐛 Troubleshooting

### Issue 1: Connection Error to R2

```
ERROR - Failed to connect to R2: ...
```

**Solution:** Check R2 credentials in `.env` file

```bash
# Verify credentials
python -c "from config import validate_r2_config; validate_r2_config()"
```

### Issue 2: Missing Files in R2

```
ERROR - No files found in raw/prices/
```

**Solution:** Verify R2 bucket structure

```bash
# List files in R2
aws s3 ls s3://your-bucket/raw/prices/ \
  --endpoint-url=$R2_ENDPOINT \
  --profile r2
```

### Issue 3: Schema Validation Failed

```
ValueError: Schema validation FAILED: Missing columns {'ticker'}
```

**Solution:** Check R2 data format matches expected schema

### Issue 4: Out of Memory

```
MemoryError: Unable to allocate array
```

**Solution:** Process data in chunks (modify `ingest.py` to use chunking)

## 📚 References

- **Context Document:** Section 3.1 (Price Data Schema)
- **Logging Standards:** Section 7.2
- **Error Handling:** Section 7.3
- **R2 Configuration:** Section 8

## 🔄 Next Steps

After successful Bronze layer ingestion:

1. ✅ Bronze Layer Complete
2. ➡️ **Next:** Silver Layer (data cleaning + enrichment)
   ```bash
   python silver/clean.py
   ```

## 📝 Notes

- Bronze layer **KHÔNG transform** data - chỉ validate và add metadata
- Raw data được preserve hoàn toàn
- Ingestion có retry logic với exponential backoff
- Failed files được log nhưng không block toàn bộ process

## 👥 Support

Nếu gặp issues:
1. Check logs trong console output
2. Verify R2 credentials
3. Validate data schema
4. Review Section 7.3 (Error Handling) trong context document
