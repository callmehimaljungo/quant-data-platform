# Nguồn Dữ Liệu

Tài liệu mô tả các nguồn dữ liệu được sử dụng trong dự án, bao gồm thông tin về nơi thu thập và phương pháp lấy dữ liệu.

---

## 1. Dữ Liệu Giá Chứng Khoán

### Nguồn

- **Tên dataset:** US Stock Market Dataset
- **Nền tảng:** Kaggle
- **Link:** <https://www.kaggle.com/datasets/paultimothymooney/stock-market-data>

### Thông tin

| Thuộc tính | Giá trị |
|------------|---------|
| Số mã cổ phiếu | 9,315 tickers |
| Số dòng dữ liệu | 34.6 triệu dòng |
| Khoảng thời gian | 02/01/1962 - 04/11/2024 |
| Dung lượng file | 917 MB (Parquet, nén snappy) |

### Các cột dữ liệu

- `Date` - Ngày giao dịch
- `Ticker` - Mã cổ phiếu (VD: AAPL, MSFT)
- `Open` - Giá mở cửa (USD)
- `High` - Giá cao nhất trong ngày
- `Low` - Giá thấp nhất trong ngày
- `Close` - Giá đóng cửa
- `Volume` - Khối lượng giao dịch
- `Dividends` - Cổ tức (nếu có)
- `Stock Splits` - Chia tách cổ phiếu (nếu có)

### Cách thu thập

1. Đăng ký tài khoản Kaggle
2. Tạo API token tại kaggle.com/settings
3. Chạy lệnh:

```bash
python bronze/ingest.py
```

Hệ thống sẽ tự động download, validate schema và lưu vào thư mục `data/bronze/`.

---

## 2. Chỉ Số Kinh Tế

### Nguồn

- **Nguồn gốc:** Federal Reserve Economic Data (FRED)
- **API:** yfinance, fredapi

### Các chỉ số thu thập

| Chỉ số | Mô tả |
|--------|-------|
| VIX | Chỉ số biến động thị trường |
| Fed Funds Rate | Lãi suất Fed |
| CPI | Chỉ số giá tiêu dùng |
| GDP | Tổng sản phẩm quốc nội |
| Treasury 10Y | Lãi suất trái phiếu 10 năm |
| Unemployment Rate | Tỷ lệ thất nghiệp |
| Dollar Index | Chỉ số USD |

### Khoảng thời gian

- Từ 01/01/2024 đến hiện tại
- Cập nhật hàng ngày

### Cách thu thập

```bash
python bronze/economic_loader.py
```

---

## 3. Dữ Liệu Benchmark (SPY)

### Nguồn

- **Symbol:** SPY (SPDR S&P 500 ETF)
- **API:** yfinance

### Mục đích sử dụng

- Tính toán hệ số Beta của từng cổ phiếu
- So sánh hiệu suất danh mục với thị trường
- Đánh giá chiến lược đầu tư

### Cách thu thập

```bash
python bronze/benchmark_loader.py
```

---

## 4. Metadata Cổ Phiếu

### Nguồn

- **API:** yfinance
- **Thông tin:** Sector, Industry, Market Cap

### Các trường dữ liệu

- `sector` - Ngành theo phân loại GICS (11 ngành)
- `industry` - Lĩnh vực cụ thể
- `market_cap` - Vốn hóa thị trường
- `market_cap_category` - Large Cap / Mid Cap / Small Cap

### Phân loại GICS (11 ngành)

1. Information Technology
2. Health Care
3. Financials
4. Consumer Discretionary
5. Communication Services
6. Industrials
7. Consumer Staples
8. Energy
9. Utilities
10. Real Estate
11. Materials

---

## 5. Tin Tức Tài Chính (Tùy chọn)

### Nguồn

- **API:** NewsAPI, Finnhub
- **Loại dữ liệu:** Headlines tin tức

### Thông tin

- Tiêu đề tin liên quan đến từng mã cổ phiếu
- Điểm sentiment (tích cực/tiêu cực/trung lập)
- Số lượng tin trong ngày

### Lưu ý

Dữ liệu tin tức chỉ khả dụng trong khoảng thời gian gần đây (1-2 tháng) do giới hạn của free API tier.

---

## Tổ Chức Lưu Trữ

Dữ liệu được tổ chức theo kiến trúc Medallion:

```
data/
├── bronze/           # Dữ liệu thô, chưa xử lý
│   ├── all_stock_data.parquet
│   ├── prices_lakehouse/
│   ├── benchmarks_lakehouse/
│   └── economic_lakehouse/
│
├── silver/           # Dữ liệu đã làm sạch
│   └── enriched_stocks.parquet
│
└── gold/             # Dữ liệu phân tích
    ├── ticker_metrics_lakehouse/
    └── sector_metrics_lakehouse/
```

---

## Yêu Cầu Kỹ Thuật

### Credentials cần thiết

Tạo file `.env` với nội dung:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### Dependencies

- pandas >= 2.0.0
- yfinance >= 0.2.0
- kaggle >= 1.5.0
- pyarrow >= 14.0.0

---

## Ghi Chú

- Bronze layer giữ nguyên dữ liệu gốc, chỉ validate schema
- Dữ liệu Kaggle dùng PascalCase (Date, Ticker, Close)
- Silver layer chuyển sang lowercase (date, ticker, close)
- Timestamp `ingested_at` được thêm vào để tracking
