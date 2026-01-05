
# Quant Data Platform

Python-based data pipeline for stock market analysis. Collects data from multiple sources (Finnhub, Polygon, NewsAPI), cleans it, and calculates risk metrics.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env
```

## Usage

### 1. Run Pipeline

To run the full ETL process (Ingestion -> Silver -> Gold):

```bash
python scripts/run_pipeline.py
```

### 2. Dashboard

Launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

### 3. Causal Analysis

Run the causal inference model:

```bash
python -m models.causal.main
```

## Architecture

- **Bronze**: Raw JSON data from APIs
- **Silver**: Cleaned Parquet files
- **Gold**: Feature-engineered datasets (Risk Metrics, Signals)

## Data Sources

- **Finnhub**: Real-time quotes
- **Polygon.io**: Historical aggregates
- **FRED**: Economic indicators (VIX, GDP, CPI)
- **NewsAPI**: Market sentiment

## Notes

- Dashboard loads from local cache by default for speed.
- See `TODO.md` for roadmap.
