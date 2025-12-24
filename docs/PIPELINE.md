# Quy Trình Xử Lý Dữ Liệu

Tài liệu mô tả chi tiết các bước xử lý dữ liệu từ raw data đến kết quả phân tích cuối cùng.

---

## Tổng Quan

Hệ thống sử dụng kiến trúc Medallion với 3 layer:

```
Bronze (Raw) → Silver (Clean) → Gold (Analytics)
```

| Layer | Mục đích | Input | Output |
|-------|----------|-------|--------|
| Bronze | Lưu trữ raw data | Kaggle, API | Parquet files |
| Silver | Làm sạch, enrichment | Bronze data | Enriched data |
| Gold | Tính toán, phân tích | Silver data | Metrics, reports |

---

## Bước 1: Bronze Layer - Thu Thập Dữ Liệu

### Chạy lệnh

```bash
python bronze/ingest.py
# hoặc chạy tất cả loaders
python bronze/run_all_loaders.py
```

### Các script chính

| Script | Chức năng |
|--------|-----------|
| `ingest.py` | Download giá cổ phiếu từ Kaggle |
| `benchmark_loader.py` | Download SPY làm benchmark |
| `economic_loader.py` | Download chỉ số kinh tế |
| `metadata_loader.py` | Fetch thông tin sector từ yfinance |

### Xử lý trong Bronze

1. Validate schema (7 cột bắt buộc)
2. Kiểm tra null trong các cột quan trọng (date, ticker, close)
3. Thêm cột `ingested_at` để tracking thời gian
4. Lưu file Parquet (nén snappy)

### Output

- `data/bronze/all_stock_data.parquet` - 34.6 triệu dòng
- `data/bronze/prices_lakehouse/` - định dạng versioned

### Lưu ý quan trọng

Bronze layer không transform dữ liệu, chỉ validate và lưu nguyên bản gốc.

---

## Bước 2: Silver Layer - Làm Sạch Dữ Liệu

### Chạy lệnh

```bash
python silver/clean.py
# hoặc chạy tất cả processors
python silver/run_all_processors.py
```

### Các bước xử lý

#### 2.1 Chuẩn hóa tên cột

Chuyển từ PascalCase sang lowercase:

- `Date` → `date`
- `Ticker` → `ticker`
- `Close` → `close`

#### 2.2 Loại bỏ duplicate

- Xóa các dòng trùng lặp theo cặp (ticker, date)
- Giữ lại bản ghi mới nhất nếu có trùng

#### 2.3 Quality Gates

Loại bỏ dữ liệu không hợp lệ:

- Giá đóng cửa <= 0
- Giá cao < giá thấp  
- Volume < 0

Kết quả: loại bỏ ~1.2 triệu dòng (3.4% tổng số)

#### 2.4 Tính toán Daily Return

```
daily_return = (close_today - close_yesterday) / close_yesterday
```

Tính theo từng ticker, sắp xếp theo ngày.

#### 2.5 Join với Metadata

- Ghép thông tin sector, industry cho từng ticker
- Các ticker không có metadata sẽ gán sector = "Unknown"

### Output

- `data/silver/enriched_stocks.parquet` - 33.5 triệu dòng
- Các cột mới: `daily_return`, `sector`, `industry`, `enriched_at`

---

## Bước 3: Gold Layer - Tính Toán Phân Tích

### Chạy lệnh

```bash
python gold/run_all_strategies.py
```

### Các metrics được tính toán

#### 3.1 Risk Metrics cho từng ticker

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| Sharpe Ratio | (mean - rf) / std × √252 | Lợi nhuận trên mỗi đơn vị rủi ro |
| Sortino Ratio | (mean - rf) / downside_std × √252 | Chỉ xét rủi ro giảm giá |
| Max Drawdown | min((price - peak) / peak) | Mức sụt giảm lớn nhất từ đỉnh |
| Volatility | std × √252 | Độ biến động hàng năm |
| Beta | cov(stock, SPY) / var(SPY) | Độ nhạy với thị trường |
| VaR 95% | percentile(returns, 5) | Mức lỗ tối đa với xác suất 95% |

#### 3.2 Sector Metrics

Tổng hợp theo ngành:

- Số lượng cổ phiếu trong ngành
- Sharpe ratio trung bình
- Volatility trung bình
- Max drawdown trung bình

### Output

- `data/gold/ticker_metrics_lakehouse/` - Metrics cho 100 tickers
- `data/gold/sector_metrics_lakehouse/` - Metrics theo 11 sectors

---

## Bước 4: Dashboard - Trực Quan Hóa

### Chạy lệnh

```bash
streamlit run dashboard/app.py
```

### Các trang trong Dashboard

| Trang | Nội dung |
|-------|----------|
| Overview | KPI tổng quan, Top 10 cổ phiếu, biểu đồ sector |
| Risk Metrics | Scatter plot risk-return, phân phối volatility |
| Sector Analysis | So sánh giữa các ngành, drill-down chi tiết |
| Model Results | Kết quả backtest các chiến lược |
| Settings | Cấu hình đường dẫn, xóa cache |

### Truy cập

Mở browser tại: <http://localhost:8501>

---

## Bước 5: Testing - Kiểm Tra

### Chạy tests

```bash
pytest tests/ -v
```

### Các test case

| File | Số tests | Coverage |
|------|----------|----------|
| test_bronze.py | 14 | Schema, nulls, data types |
| test_silver.py | 13 | Quality gates, dedup, returns |
| test_risk_metrics.py | 21 | Sharpe, Beta, MDD, Sortino |

Tổng: 48 tests

---

## Thứ Tự Chạy Đúng

```bash
# 1. Thu thập dữ liệu
python bronze/ingest.py

# 2. Làm sạch dữ liệu
python silver/clean.py

# 3. Tính toán metrics
python gold/run_all_strategies.py

# 4. Chạy dashboard
streamlit run dashboard/app.py

# 5. Chạy tests (tùy chọn)
pytest tests/ -v
```

---

## Xử Lý Lỗi Thường Gặp

### Lỗi: FileNotFoundError data/bronze/

Nguyên nhân: Chưa chạy Bronze layer
Cách sửa: Chạy `python bronze/ingest.py` trước

### Lỗi: KeyError 'sector'

Nguyên nhân: Silver data chưa có metadata
Cách sửa: Chạy `python bronze/metadata_loader.py` rồi chạy lại Silver

### Lỗi: MemoryError

Nguyên nhân: Dữ liệu quá lớn (33 triệu dòng)
Cách sửa: Xử lý theo chunks hoặc filter theo khoảng thời gian

---

## Cấu Trúc Thư Mục

```
quant-data-platform/
├── bronze/               # Scripts thu thập dữ liệu
│   ├── ingest.py
│   └── run_all_loaders.py
│
├── silver/               # Scripts làm sạch dữ liệu
│   ├── clean.py
│   └── run_all_processors.py
│
├── gold/                 # Scripts phân tích
│   ├── risk_metrics.py
│   └── run_all_strategies.py
│
├── dashboard/            # Ứng dụng Streamlit
│   └── app.py
│
├── tests/                # Unit tests
│   ├── test_bronze.py
│   ├── test_silver.py
│   └── test_risk_metrics.py
│
├── utils/                # Tiện ích dùng chung
│   └── lakehouse_helper.py
│
├── data/                 # Dữ liệu (không commit lên Git)
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── config.py             # Cấu hình paths, constants
└── requirements.txt      # Dependencies
```

---

## Thông Số Kỹ Thuật

### Thời gian xử lý (ước tính)

- Bronze ingestion: 10-15 phút
- Silver cleaning: 5-10 phút
- Gold calculations: 2-5 phút

### Yêu cầu phần cứng

- RAM: tối thiểu 16GB (khuyến nghị 32GB)
- Disk: 5GB trống cho dữ liệu
- Python: 3.10 trở lên
