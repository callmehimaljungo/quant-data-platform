# 🚀 QUICK START - Bronze Layer

## 3-Minute Setup

### 1️⃣ Install Dependencies (30 seconds)
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 pyarrow>=14.0.0 boto3>=1.28.0 python-dotenv>=1.0.0
```

### 2️⃣ Configure R2 (1 minute)
Create `.env` file:
```bash
R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY=your_access_key_here
R2_SECRET_KEY=your_secret_key_here
R2_BUCKET=your_bucket_name
```

### 3️⃣ Run Bronze Ingestion (90 seconds)
```bash
python bronze/ingest.py
```

**Done!** ✅ Output: `./data/bronze/prices.parquet`

---

## Verify Results

```bash
# Test Bronze output
python test_bronze.py

# Quick check in Python
python -c "
import pandas as pd
df = pd.read_parquet('./data/bronze/prices.parquet')
print(f'Rows: {len(df):,}')
print(f'Tickers: {df[\"ticker\"].nunique():,}')
print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
"
```

---

## Expected Output

```
✓ Rows: 2,500,000
✓ Tickers: 9,315
✓ Date range: 1962-01-02 to 2025-04-02
```

---

## Troubleshooting

**Error:** `Missing required environment variables`
→ Check `.env` file has all 4 R2 credentials

**Error:** `No files found in raw/prices/`
→ Verify R2 bucket path: should be `raw/prices/`

**Error:** `Schema validation FAILED`
→ Check R2 data format matches expected schema

---

## What's Next?

✅ Bronze Layer Complete → Next: **Silver Layer**

```bash
# Coming next: Silver layer (data cleaning)
python silver/clean.py
```

---

## Files Overview

```
bronze/ingest.py      ← Main ingestion script (run this)
config.py            ← Configuration management
test_bronze.py       ← Validation tests
.env                 ← Your R2 credentials (create this)
```

---

## Key Commands

```bash
# Validate configuration
python config.py

# Run Bronze ingestion
python bronze/ingest.py

# Test results
python test_bronze.py

# View data
python -c "import pandas as pd; print(pd.read_parquet('./data/bronze/prices.parquet').head())"
```

---

**Need help?** → Check `README.md` for detailed documentation
