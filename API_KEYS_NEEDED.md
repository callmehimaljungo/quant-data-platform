# 🔑 Required API Keys Checklist

Based on full codebase scan: `quant-data-platform`

## 🚨 CRITICAL (Must Have)

### 1. Finnhub

- **Status:** ❌ Missing / Error (Key `d5cf...` returned 403)
- **Used In:**
  - `bronze/news_loader.py` (Market News)
  - `bronze/collectors/prices/finnhub_collector.py` (Price Backup)
- **Get Key:** [https://finnhub.io/register](https://finnhub.io/register) (Free Sandbox)

---

## ✅ INSTALLED (Ready)

### 2. FRED (Federal Reserve)

- **Status:** ✅ Ready (`c9038f...`)
- **Used In:** `bronze/collectors/economic/fred_collector.py`
- **Coverage:** 17 Economic Indicators (1962+)

### 3. Twelve Data

- **Status:** ✅ Ready (`ee3338...`)
- **Used In:** `bronze/collectors/prices/twelvedata_collector.py`
- **Coverage:** Price Backup (800 req/day)

### 4. YFinance

- **Status:** ✅ Ready (No Key Required)
- **Used In:** `bronze/collectors/prices/yfinance_collector.py`

---

## ⚠️ OPTIONAL (Nice to Have)

### 5. Polygon.io

- **Status:** ❌ Missing
- **Used In:** `bronze/collectors/prices/polygon_collector.py`
- **Note:** Existing code supports it. Good backup if you have a key.
- **Get Key:** [https://polygon.io/dashboard/signup](https://polygon.io/dashboard/signup)

### 6. NewsAPI

- **Status:** ❌ Missing (No collector code yet)
- **Note:** Useful for broader general news coverage.
- **Get Key:** [https://newsapi.org/register](https://newsapi.org/register)

---

## 📝 ACTION ITEMS

Please provide the **Finnhub Key** to restore News & Price capabilities:

```
Finnhub: [PASTE_KEY_HERE]
```
