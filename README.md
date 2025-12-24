# Quant Data Platform

## 📋 Overview

A **Data Lake Platform** for quantitative stock analysis, built with Medallion Architecture (Bronze → Silver → Gold layers).

> **Note:** This platform uses a simplified Data Lake pattern with Parquet files and versioning metadata.
> It is NOT a full Data Lakehouse (which requires Delta Lake/Iceberg for ACID transactions).

**Data Sources:**

- **Primary:** Kaggle dataset (`hmingjungo/stock-price`)
- **Alternative:** Cloudflare R2 storage

**Nhiệm vụ chính:**

- Load dữ liệu OHLCV của 9000+ US stocks
- Validate schema theo Section 3.1
- **KHÔNG transform** data trong Bronze (giữ nguyên raw)
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

# bronze/ingest.py - Updated Ingestion Script

## Changes

- Added detailed logging
- Included data validation steps
- Enhanced error handling
- Cleaned up temporary files after ingestion

## Usage

Run the script as part of the Bronze layer ingestion process.

```python
import os
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from config import R2_BUCKET, R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY
import logging
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bronze.ingest')

def ingest():
    logger.info("="*70)
    logger.info("BRONZE LAYER INGESTION STARTED")
    logger.info("="*70)

    # Step 1: Connect to R2 storage
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError

        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name='auto',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY
        )
        logger.info("✓ Successfully connected to R2 storage")
    except (NoCredentialsError, PartialCredentialsError) as e:
        logger.error("✗ R2 credentials not found or incomplete")
        logger.exception(e)
        return
    except Exception as e:
        logger.error("✗ Failed to connect to R2")
        logger.exception(e)
        return

    # Step 2: List and process all parquet files in the raw/prices/ directory
    try:
        objects = client.list_objects_v2(Bucket=R2_BUCKET, Prefix='raw/prices/')
        files = [obj['Key'] for obj in objects.get('Contents', []) if obj['Key'].endswith('.parquet')]

        logger.info(f"✓ Found {len(files)} parquet files to process")
    except Exception as e:
        logger.error("✗ Failed to list files in R2 bucket")
        logger.exception(e)
        return

    all_data = []
    for file in files:
        # Download file
        try:
            logger.info(f"Processing file: {file}")

            with tempfile.TemporaryDirectory() as tmpdirname:
                # Download file to temp directory
                tmp_file = os.path.join(tmpdirname, os.path.basename(file))
                client.download_file(R2_BUCKET, file, tmp_file)
                logger.info(f"✓ Downloaded {file} to {tmp_file}")

                # Read parquet file
                df = pd.read_parquet(tmp_file)
                logger.info(f"✓ Read {len(df)} rows from {tmp_file}")

                # Validate schema
                required_columns = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume'}
                if not required_columns.issubset(df.columns):
                    logger.error(f"✗ Schema validation FAILED: Missing columns {required_columns - set(df.columns)}")
                    continue

                # Check for nulls in critical columns
                null_checks = df[['date', 'ticker', 'close']].isnull().sum()
                if null_checks.any():
                    logger.warning(f"⚠️ Null values found in critical columns: {null_checks[null_checks > 0]}")

                # Add metadata column
                df['ingested_at'] = datetime.now()
                logger.info("✓ Added metadata column: ingested_at")

                # Append to all data
                all_data.append(df)
        except Exception as e:
            logger.error(f"✗ Error processing file {file}")
            logger.exception(e)
            continue

    # Combine all data into single DataFrame
    if all_data:
        try:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"✓ Combined data from {len(files)} files: {len(combined_df)} rows")

            # Save to final destination
            output_file = './data/bronze/prices.parquet'
            combined_df.to_parquet(output_file, index=False)
            logger.info(f"✓ Data saved to {output_file}")

            # Log data statistics
            logger.info(f"Total rows loaded: {len(combined_df)}")
            logger.info(f"Total unique tickers: {combined_df['ticker'].nunique()}")
            logger.info(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Step 4: Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            logger.info(f"✓ Standardized column names: {df.columns.tolist()}")
            
            # Clean up temp files
            shutil.rmtree(tmpdirname)
            logger.info(f"✓ Cleaned up temporary files")
        except Exception as e:
            logger.error("✗ Failed to combine or save data")
            logger.exception(e)
    else:
        logger.warning("No data to process")

    logger.info("="*70)
    logger.info("BRONZE LAYER INGESTION COMPLETED")
    logger.info("="*70)

if __name__ == "__main__":
    ingest()
```
