"""Check Kaggle dataset coverage"""
import pandas as pd

df = pd.read_csv('data/temp/sp500_stocks.csv')

print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Tickers: {df['Symbol'].nunique()}")
print(f"\nDate range:")
print(f"  Min: {df['Date'].min()}")
print(f"  Max: {df['Date'].max()}")

# Check if we have Nov-Dec 2024 data
df['Date'] = pd.to_datetime(df['Date'])
nov_dec = df[(df['Date'] >= '2024-11-01') & (df['Date'] <= '2025-01-02')]
print(f"\nNov 2024 - Jan 2025 data:")
print(f"  Rows: {len(nov_dec):,}")
print(f"  Dates: {nov_dec['Date'].min().date()} to {nov_dec['Date'].max().date()}")
print(f"  Tickers: {nov_dec['Symbol'].nunique()}")
