# 🚀 HƯỚNG DẪN NHANH - Bronze Layer (CẬP NHẬT)

## 🎯 MỤC TIÊU
Tải dữ liệu 9000+ mã chứng khoán từ **Kaggle** (KHÔNG cần R2) → Lưu vào `prices.parquet`

---

## ✨ THAY ĐỔI MỚI

### Trước (Version cũ):
```
❌ PHẢI có R2 → Phức tạp, tốn thời gian upload
```

### Giờ (Version mới):
```
✅ Đọc TRỰC TIẾP từ Kaggle → Đơn giản, nhanh chóng!
✅ Vẫn hỗ trợ R2 nếu bạn muốn dùng
```

---

## 📝 CÁCH SỬ DỤNG - 3 BƯỚC ĐƠN GIẢN

### **BƯỚC 1: Cài Thư Viện** (30 giây)

```bash
pip install pandas numpy pyarrow kaggle
```

**Giải thích:**
- `pandas`: Xử lý dữ liệu
- `pyarrow`: Đọc/ghi file parquet
- `kaggle`: Tải data từ Kaggle

---

### **BƯỚC 2: Cấu Hình Kaggle API** (2 phút)

#### 2.1 Lấy API Credentials

1. Vào https://www.kaggle.com/
2. Click vào avatar góc phải → **Settings**
3. Kéo xuống phần **API** → Click **Create New Token**
4. File `kaggle.json` sẽ được tải về

#### 2.2 Đặt File vào Đúng Chỗ

**Trên Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Trên Windows:**
```bash
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

#### 2.3 File kaggle.json Trông Như Thế Nào?

```json
{
  "username": "your_username",
  "key": "abc123def456ghi789"
}
```

---

### **BƯỚC 3: Chạy Bronze Ingestion** (5-10 phút)

```bash
# Đơn giản vậy thôi!
python bronze/ingest.py
```

Hoặc chỉ định nguồn rõ ràng:
```bash
# Tải từ Kaggle (mặc định)
python bronze/ingest.py kaggle

# Hoặc từ R2 (nếu bạn đã upload)
python bronze/ingest.py r2
```

---

## 📊 QUÁ TRÌNH CHẠY

Bạn sẽ thấy:

```
🚀 BRONZE LAYER INGESTION
📊 Data Source: auto

======================================================================
BRONZE LAYER INGESTION FROM KAGGLE
======================================================================
Downloading dataset: hmingjungo/stock-price
✓ Download completed
Loading file: all_stock_data.csv
Loading CSV file... (this may take a few minutes)
✓ Loaded 2,500,000 rows
✓ Unique tickers: 9,315
✓ Date range: 1962-01-02 to 2025-04-02
✓ Cleaned up temp files
======================================================================
KAGGLE INGESTION COMPLETED
Duration: 320.50 seconds
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

**Mất bao lâu:**
- Download từ Kaggle: ~2-3 phút (tùy mạng)
- Load CSV: ~2-3 phút
- Validation + Save: ~30 giây
- **Tổng:** ~5-10 phút

---

## ✅ KIỂM TRA KẾT QUẢ

```bash
# Test xem data đã đúng chưa
python test_bronze.py
```

Kết quả mong đợi:
```
✓ File exists: ./data/bronze/prices.parquet
✓ Data loaded: 2,500,000 rows
✓ Unique tickers: 9,315
✓ Date range: 1962-01-02 to 2025-04-02
✓✓✓ ALL TESTS PASSED ✓✓✓
```

---

## 🎉 XONG RỒI!

Bạn đã có file `./data/bronze/prices.parquet` với:
- ✅ 2.5 triệu dòng dữ liệu
- ✅ 9,315 mã chứng khoán
- ✅ Dữ liệu từ 1962 → 2025
- ✅ Schema đã validate

**Xem nhanh data:**
```bash
python -c "import pandas as pd; print(pd.read_parquet('./data/bronze/prices.parquet').head())"
```

---

## ❓ CÂU HỎI THƯỜNG GẶP

### Q1: Tôi có cần R2 không?

**A:** KHÔNG! Version mới tải trực tiếp từ Kaggle.

---

### Q2: Lỗi "Unauthorized" khi download Kaggle?

**A:** Kiểm tra:
1. File `kaggle.json` đã đúng chỗ chưa? (`~/.kaggle/kaggle.json`)
2. Permissions đúng chưa? (`chmod 600 ~/.kaggle/kaggle.json`)
3. Username/key trong file có đúng không?

---

### Q3: Dataset quá lớn, download lâu?

**A:** 
- File gốc ~3.5GB, nén lại ~300MB khi download
- Có thể mất 2-5 phút tùy tốc độ mạng
- Chỉ cần download 1 lần duy nhất!

---

### Q4: Tôi vẫn muốn dùng R2 thì sao?

**A:** Hoàn toàn được! Chạy:
```bash
python bronze/ingest.py r2
```
Nhớ config `.env` với R2 credentials.

---

### Q5: Auto-detect chọn nguồn nào?

**A:** Thứ tự ưu tiên:
1. ✅ Kaggle (nếu có `kaggle` package)
2. ✅ R2 (nếu có `boto3` + credentials trong .env)
3. ❌ Error nếu không có cả hai

---

## 🐛 XỬ LÝ LỖI

### Lỗi: "kaggle not installed"
```bash
pip install kaggle
```

### Lỗi: "Unauthorized"
Làm lại BƯỚC 2 (Cấu hình Kaggle API)

### Lỗi: "No CSV files found"
Kaggle dataset có thể thay đổi tên file. Check trong `./temp/` xem có file gì.

### Lỗi: "Memory Error"
Data quá lớn. Giải pháp:
1. Đóng các chương trình khác
2. Hoặc dùng máy có RAM lớn hơn (cần ~8GB)

---

## 🔄 BƯỚC TIẾP THEO

✅ Bronze Layer hoàn thành → Tiếp: **Silver Layer**

```bash
python silver/clean.py
```

Silver Layer sẽ:
- Xóa duplicate
- Lọc giá không hợp lệ
- Thêm sector metadata
- Tính daily return

---

## 📂 CẤU TRÚC FILE

```
./
├── bronze/
│   └── ingest.py          ← Chương trình chính (ĐÃ CẬP NHẬT)
├── data/
│   └── bronze/
│       └── prices.parquet ← Kết quả output
├── temp/                  ← Tự động tạo, chứa file download tạm
└── ~/.kaggle/
    └── kaggle.json        ← API credentials
```

---

## 💡 SO SÁNH VERSION CŨ VS MỚI

| Aspect | Version Cũ | Version Mới |
|--------|-----------|-------------|
| **Nguồn data** | R2 only | Kaggle (primary) + R2 (alternative) |
| **Setup** | Phức tạp (R2 creds) | Đơn giản (Kaggle API) |
| **Thời gian** | Upload lâu | Tải trực tiếp nhanh |
| **Flexibility** | Cố định R2 | Tự động detect hoặc chọn |

---

**Need help?** Đọc `README.md` để biết chi tiết hơn về cả 2 methods (Kaggle + R2)
