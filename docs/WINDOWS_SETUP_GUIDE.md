# 🪟 HƯỚNG DẪN CHI TIẾT - BRONZE LAYER TRÊN WINDOWS

## 📋 TỔNG QUAN QUY TRÌNH

```
KAGGLE DATASET → Bronze Layer → prices.parquet
     ↓
Cần: Kaggle API Token
     ↓
Tải tự động qua Python
     ↓
Validate schema
     ↓
Lưu vào ./data/bronze/
```

---

## ✅ CHECKLIST TRƯỚC KHI BẮT ĐẦU

### Phần 1: Môi Trường Python
- [ ] Python 3.8+ đã cài (kiểm tra: `python --version`)
- [ ] pip hoạt động (kiểm tra: `pip --version`)

### Phần 2: Kaggle Account
- [ ] Có tài khoản Kaggle (đăng ký tại kaggle.com)
- [ ] Đã tải file `kaggle.json` (API token)

### Phần 3: Cấu Trúc Folder
- [ ] Đã tạo folder project
- [ ] Đã có các file code cần thiết

---

## 📂 BƯỚC 1: TẠO CẤU TRÚC FOLDER

### 1.1 Mở Command Prompt hoặc PowerShell

**Command Prompt:**
```cmd
# Tìm kiếm "cmd" trong Windows Start Menu
```

**PowerShell (Khuyến nghị):**
```powershell
# Tìm kiếm "PowerShell" trong Windows Start Menu
# Nhấn phải → Run as Administrator
```

### 1.2 Tạo Folder Project

```powershell
# Chọn vị trí lưu project (ví dụ: Desktop)
cd Desktop

# Tạo folder project
mkdir quant-data-platform
cd quant-data-platform

# Tạo cấu trúc thư mục
mkdir bronze
mkdir data\bronze
mkdir data\silver
mkdir data\gold
mkdir data\metadata
mkdir docs
mkdir temp

# Kiểm tra
tree /F
```

**Kết quả mong đợi:**
```
quant-data-platform/
├── bronze/
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── metadata/
├── docs/
└── temp/
```

---

## 🔑 BƯỚC 2: LẤY KAGGLE API TOKEN

### 2.1 Đăng Nhập Kaggle

1. Vào https://www.kaggle.com/
2. Đăng nhập hoặc tạo tài khoản mới

### 2.2 Lấy API Credentials

1. Click vào **avatar** (góc trên bên phải)
2. Chọn **Settings**
3. Kéo xuống phần **API**
4. Click **"Create New API Token"**
5. File `kaggle.json` sẽ tự động tải về folder Downloads

**File kaggle.json trông như thế này:**
```json
{
  "username": "your_username",
  "key": "abc123def456..."
}
```

### 2.3 Đặt File vào Đúng Chỗ (QUAN TRỌNG!)

**Trên Windows:**

```powershell
# Cách 1: Dùng PowerShell (Khuyến nghị)
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\

# Cách 2: Dùng Command Prompt
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# Cách 3: Thủ công (nếu lệnh không chạy)
# 1. Mở File Explorer
# 2. Gõ vào address bar: %USERPROFILE%
# 3. Tạo folder tên là ".kaggle" (có dấu chấm đầu)
# 4. Copy file kaggle.json từ Downloads vào folder .kaggle
```

**Vị trí cuối cùng:**
```
C:\Users\YourName\.kaggle\kaggle.json
```

### 2.4 Kiểm Tra

```powershell
# Xem file có đúng chỗ không
type $env:USERPROFILE\.kaggle\kaggle.json

# Hoặc Command Prompt
type %USERPROFILE%\.kaggle\kaggle.json
```

---

## 📦 BƯỚC 3: CÀI ĐẶT THỦ VIỆN PYTHON

### 3.1 Tạo Virtual Environment (Khuyến nghị)

```powershell
# Trong folder project
python -m venv venv

# Kích hoạt virtual environment
# PowerShell:
.\venv\Scripts\Activate.ps1

# Command Prompt:
venv\Scripts\activate.bat

# Nếu gặp lỗi "script execution disabled":
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3.2 Cài Đặt Packages

**Option A: Từ requirements.txt (nếu có)**
```powershell
pip install -r requirements.txt
```

**Option B: Cài thủ công (nếu chưa có requirements.txt)**
```powershell
# Packages BẮT BUỘC cho Bronze Layer
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install pyarrow>=14.0.0
pip install kaggle>=1.5.0
pip install python-dotenv>=1.0.0

# Packages OPTIONAL (cho R2)
pip install boto3>=1.28.0
```

### 3.3 Kiểm Tra Cài Đặt

```powershell
# Test Kaggle
python -c "import kaggle; print('Kaggle OK')"

# Test pandas
python -c "import pandas as pd; print('Pandas OK')"

# Test pyarrow
python -c "import pyarrow; print('PyArrow OK')"
```

**Kết quả mong đợi:**
```
Kaggle OK
Pandas OK
PyArrow OK
```

---

## 📝 BƯỚC 4: TẠO CÁC FILE CODE

### 4.1 File: config.py

**Tạo file trong folder gốc:**

```powershell
# Tạo file mới
notepad config.py
```

**Copy nội dung từ document index="9"** (file config.py đã có)

**Hoặc tạo version đơn giản:**

```python
"""Configuration Management"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
BRONZE_DIR = DATA_DIR / 'bronze'
SILVER_DIR = DATA_DIR / 'silver'
GOLD_DIR = DATA_DIR / 'gold'
METADATA_DIR = DATA_DIR / 'metadata'

# Create directories
for directory in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Expected schema (Kaggle format - PascalCase)
PRICE_DATA_SCHEMA = {
    'Date': 'datetime64[ns]',
    'Ticker': 'object',
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Volume': 'int64'
}

# Critical columns (cannot have nulls)
CRITICAL_COLUMNS = ['Date', 'Ticker', 'Close']

# Output path
OUTPUT_PATH = BRONZE_DIR / 'prices.parquet'
KAGGLE_DATASET = 'hmingjungo/stock-price'
TEMP_DIR = PROJECT_ROOT / 'temp'

print(f"✓ Config loaded")
print(f"✓ Project root: {PROJECT_ROOT}")
print(f"✓ Data directory: {DATA_DIR}")
```

**Lưu file (Ctrl+S) và đóng Notepad**

### 4.2 File: bronze/__init__.py

```powershell
# Tạo file
notepad bronze\__init__.py
```

**Nội dung:**
```python
"""Bronze Layer Module"""
from .ingest import ingest_all_stocks, save_to_bronze

__all__ = ['ingest_all_stocks', 'save_to_bronze']
```

### 4.3 File: bronze/ingest.py

**File QUAN TRỌNG NHẤT - Copy từ document index="7"**

```powershell
notepad bronze\ingest.py
```

**Copy toàn bộ nội dung từ document index="7"** (file bronze/ingest.py đã có)

---

## 🚀 BƯỚC 5: CHẠY BRONZE INGESTION

### 5.1 Kiểm Tra Trước Khi Chạy

```powershell
# 1. Check Kaggle token
type %USERPROFILE%\.kaggle\kaggle.json

# 2. Check config
python config.py

# 3. Test import
python -c "from bronze.ingest import ingest_all_stocks; print('OK')"
```

### 5.2 Chạy Bronze Layer

**Option A: Auto-detect (Khuyến nghị)**
```powershell
python bronze\ingest.py
```

**Option B: Force Kaggle**
```powershell
python bronze\ingest.py kaggle
```

**Option C: Force R2 (nếu đã setup)**
```powershell
python bronze\ingest.py r2
```

### 5.3 Theo Dõi Quá Trình

Bạn sẽ thấy output như sau:

```
🚀 BRONZE LAYER INGESTION
📊 Data Source: auto

======================================================================
BRONZE LAYER INGESTION FROM KAGGLE
======================================================================
Downloading dataset: hmingjungo/stock-price
Downloading...
100%|████████████████████████████| 300M/300M [02:15<00:00, 2.22MB/s]
✓ Download completed
Loading file: all_stock_data.csv
Loading CSV file... (this may take a few minutes)
✓ Loaded 2,500,000 rows
✓ Unique tickers: 9,315
✓ Date range: 1962-01-02 to 2025-04-02
✓ Cleaned up temp files
======================================================================
KAGGLE INGESTION COMPLETED
Duration: 320.50 seconds (~5 phút)
======================================================================
✓ Added ingestion timestamp
Running schema validation...
✓ Schema validation PASSED: All checks successful
======================================================================
✓✓✓ BRONZE LAYER INGESTION COMPLETED SUCCESSFULLY ✓✓✓
Duration: 325.00 seconds
Total rows: 2,500,000
Memory usage: 450.25 MB
======================================================================
Saving to ./data/bronze/prices.parquet...
✓ Data saved to ./data/bronze/prices.parquet
✓ File size: 380.50 MB

✅ Bronze layer ingestion completed successfully!
✅ Output: ./data/bronze/prices.parquet
```

**Thời gian:**
- Download: 2-5 phút (tùy tốc độ mạng)
- Load CSV: 2-3 phút
- Validate + Save: 30 giây
- **Tổng:** 5-10 phút

---

## 📊 BƯỚC 6: KIỂM TRA KẾT QUẢ

### 6.1 Xem File Output

```powershell
# Kiểm tra file có tồn tại
dir data\bronze\

# Xem kích thước
dir data\bronze\prices.parquet
```

**Kết quả mong đợi:**
```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        20/12/2024   6:00 PM      380500000 prices.parquet
```

### 6.2 Test Đọc Data

```powershell
python -c "import pandas as pd; df = pd.read_parquet('./data/bronze/prices.parquet'); print(f'Rows: {len(df):,}'); print(f'Columns: {df.columns.tolist()}'); print(df.head())"
```

**Kết quả mong đợi:**
```
Rows: 2,500,000
Columns: ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'ingested_at']

         Date Ticker   Open   High    Low  Close      Volume         ingested_at
0  1962-01-02   AAPL  0.422  0.422  0.422  0.422  117258400 2024-12-20 18:00:00
1  1962-01-03   AAPL  0.422  0.422  0.422  0.422   67649600 2024-12-20 18:00:00
...
```

### 6.3 Quick Stats

```powershell
python -c "import pandas as pd; df = pd.read_parquet('./data/bronze/prices.parquet'); print(f'Total rows: {len(df):,}'); print(f'Unique tickers: {df[\"Ticker\"].nunique():,}'); print(f'Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}'); print(f'Nulls in critical columns:'); print(df[['Date', 'Ticker', 'Close']].isnull().sum())"
```

**Kết quả mong đợi:**
```
Total rows: 2,500,000
Unique tickers: 9,315
Date range: 1962-01-02 to 2025-04-02
Nulls in critical columns:
Date      0
Ticker    0
Close     0
```

---

## 🔍 LUỒNG DỮ LIỆU CHI TIẾT

### Giai Đoạn 1: Download từ Kaggle

```
1. Kaggle API check credentials
   ↓ File: C:\Users\YourName\.kaggle\kaggle.json
   
2. Request dataset: hmingjungo/stock-price
   ↓ API call: kaggle.api.dataset_download_files()
   
3. Download ZIP (~300MB compressed)
   ↓ Lưu vào: ./temp/
   
4. Unzip tự động
   ↓ Extract: all_stock_data.csv (~3.5GB)
   
5. CSV được tạo
   ✓ File: ./temp/all_stock_data.csv
```

### Giai Đoạn 2: Load vào Python

```
1. Đọc CSV với pandas
   ↓ pd.read_csv('./temp/all_stock_data.csv', parse_dates=['Date'])
   
2. Infer data types
   ↓ Date: datetime64[ns]
   ↓ Ticker: object (string)
   ↓ OHLCV: float64, int64
   
3. Load vào memory
   ✓ DataFrame với 2.5M rows
```

### Giai Đoạn 3: Validation

```
1. Check schema
   ✓ Columns: Date, Ticker, Open, High, Low, Close, Volume
   ✓ Types match PRICE_DATA_SCHEMA
   
2. Check nulls in critical columns
   ✓ Date, Ticker, Close: NO nulls
   
3. Add metadata
   ✓ ingested_at: datetime.now()
```

### Giai Đoạn 4: Save Output

```
1. Convert to Parquet
   ↓ df.to_parquet('./data/bronze/prices.parquet')
   ↓ Engine: pyarrow
   ↓ Compression: snappy
   
2. File được tạo
   ✓ Size: ~380MB (compressed)
   ✓ Format: Apache Parquet
   
3. Cleanup temp files
   ✓ Delete: ./temp/all_stock_data.csv
```

---

## 🎯 KAGGLE VS R2 - KHI NÀO DÙNG GÌ?

### KAGGLE (Khuyến nghị - Mặc định)

**Ưu điểm:**
✅ Đơn giản, không cần setup cloud
✅ Free, không giới hạn
✅ Tự động download + update

**Nhược điểm:**
⚠️ Cần internet để download
⚠️ Download lần đầu hơi lâu (5-10 phút)

**Khi nào dùng:**
- Đồ án cá nhân
- Lần đầu chạy
- Không có R2 credentials

### R2 (Alternative - Cho Production)

**Ưu điểm:**
✅ Nhanh hơn nếu data đã ở R2
✅ Có thể chia sẻ data trong team
✅ Versioning + backup

**Nhược điểm:**
⚠️ Phải setup credentials
⚠️ Phải upload data lần đầu
⚠️ Phức tạp hơn

**Khi nào dùng:**
- Team project
- Production deployment
- Cần data versioning

---

## ❓ TROUBLESHOOTING - CÁC LỖI THƯỜNG GẶP

### Lỗi 1: "Unauthorized" khi download Kaggle

**Nguyên nhân:** Kaggle API token không đúng

**Cách fix:**
```powershell
# 1. Check file có tồn tại
type %USERPROFILE%\.kaggle\kaggle.json

# 2. Nếu không có, làm lại BƯỚC 2
# 3. Nếu có, kiểm tra nội dung:
#    - username đúng chưa?
#    - key đúng chưa?

# 4. Thử lại
python bronze\ingest.py
```

### Lỗi 2: "ModuleNotFoundError: No module named 'kaggle'"

**Nguyên nhân:** Chưa cài package kaggle

**Cách fix:**
```powershell
pip install kaggle
python bronze\ingest.py
```

### Lỗi 3: "FileNotFoundError: No CSV files found"

**Nguyên nhân:** Kaggle dataset thay đổi tên file

**Cách fix:**
```powershell
# Check trong folder temp
dir temp

# Xem file nào được download
# Sửa trong bronze/ingest.py:
# Dòng: csv_files = [f for f in os.listdir(TEMP_DIR) if f.endswith('.csv')]
```

### Lỗi 4: "MemoryError"

**Nguyên nhân:** Không đủ RAM (cần ~8GB)

**Cách fix:**
- Đóng các chương trình khác
- Hoặc dùng máy khác có RAM lớn hơn
- Hoặc modify code để load chunks

### Lỗi 5: "Permission denied" khi tạo folder

**Nguyên nhân:** Windows bảo vệ folder

**Cách fix:**
```powershell
# Chạy PowerShell as Administrator
# Hoặc chọn folder khác (không phải C:\Program Files)
```

### Lỗi 6: "Script execution disabled" (PowerShell)

**Nguyên nhân:** Windows security policy

**Cách fix:**
```powershell
# Chạy lệnh này (as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sau đó kích hoạt lại venv
.\venv\Scripts\Activate.ps1
```

---

## 📋 CHECKLIST HOÀN THÀNH

### Setup
- [ ] Tạo folder structure
- [ ] Cài Python packages
- [ ] Lấy Kaggle API token
- [ ] Đặt kaggle.json đúng chỗ
- [ ] Tạo file config.py
- [ ] Tạo file bronze/ingest.py

### Execution
- [ ] Chạy `python bronze\ingest.py`
- [ ] Thấy log "INGESTION COMPLETED SUCCESSFULLY"
- [ ] File `data\bronze\prices.parquet` được tạo
- [ ] File size ~380MB
- [ ] Test đọc data OK

### Validation
- [ ] 2.5M rows
- [ ] 9,315 tickers
- [ ] No nulls trong Date, Ticker, Close
- [ ] Schema đúng

---

## 🎉 HOÀN THÀNH!

Nếu tất cả checklist đã ✅, bạn có:

```
✅ Bronze Layer hoàn thành
✅ File: ./data/bronze/prices.parquet (380MB)
✅ Data: 2.5M rows, 9315 tickers
✅ Date range: 1962 → 2025
```

**Bước tiếp theo:** Silver Layer

```powershell
# Coming soon...
python silver\clean.py
```

---

## 📚 TÀI LIỆU THAM KHẢO

### Files Liên Quan
- `config.py` - Configuration
- `bronze/ingest.py` - Main ingestion script
- `requirements.txt` - Python packages
- `docs/QUICK_START_v2.md` - Quick guide

### Kaggle Dataset
- URL: https://www.kaggle.com/datasets/hmingjungo/stock-price
- Size: ~3.5GB (uncompressed)
- Format: CSV
- Tickers: 9315 US stocks
- Date range: 1962-01-02 to 2025-04-02

### Context Document References
- Section 3.1: Price Data Schema
- Section 7.2: Logging Standards
- Section 7.3: Error Handling
- Section 8: R2 Configuration (optional)

---

**Có thắc mắc?** Check lại từng bước trong guide này!
