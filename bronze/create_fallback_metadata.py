import pandas as pd
from pathlib import Path
import logging

# Config
DATA_DIR = Path('data')
METADATA_DIR = DATA_DIR / 'bronze' / 'metadata'
METADATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = METADATA_DIR / 'ticker_metadata.parquet'

# Hardcoded metadata for major tickers (Top holdings + ETFs)
FALLBACK_DATA = [
    # ETFs
    {'ticker': 'SPY', 'sector': 'Index ETF', 'industry': 'Index ETF'},
    {'ticker': 'QQQ', 'sector': 'Index ETF', 'industry': 'Index ETF'},
    {'ticker': 'IWM', 'sector': 'Index ETF', 'industry': 'Index ETF'},
    {'ticker': 'VXX', 'sector': 'Volatility', 'industry': 'Volatility'},
    
    # Technology
    {'ticker': 'AAPL', 'sector': 'Technology', 'industry': 'Consumer Electronics'},
    {'ticker': 'MSFT', 'sector': 'Technology', 'industry': 'Software - Infrastructure'},
    {'ticker': 'NVDA', 'sector': 'Technology', 'industry': 'Semiconductors'},
    {'ticker': 'AVGO', 'sector': 'Technology', 'industry': 'Semiconductors'},
    {'ticker': 'ORCL', 'sector': 'Technology', 'industry': 'Software - Infrastructure'},
    {'ticker': 'ADBE', 'sector': 'Technology', 'industry': 'Software - Infrastructure'},
    {'ticker': 'CRM', 'sector': 'Technology', 'industry': 'Software - Application'},
    {'ticker': 'AMD', 'sector': 'Technology', 'industry': 'Semiconductors'},
    
    # Communication Services
    {'ticker': 'GOOG', 'sector': 'Communication Services', 'industry': 'Internet Content & Information'},
    {'ticker': 'GOOGL', 'sector': 'Communication Services', 'industry': 'Internet Content & Information'},
    {'ticker': 'META', 'sector': 'Communication Services', 'industry': 'Internet Content & Information'},
    {'ticker': 'NFLX', 'sector': 'Communication Services', 'industry': 'Entertainment'},
    {'ticker': 'DIS', 'sector': 'Communication Services', 'industry': 'Entertainment'},
    
    # Consumer Cyclical
    {'ticker': 'AMZN', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail'},
    {'ticker': 'TSLA', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers'},
    {'ticker': 'HD', 'sector': 'Consumer Cyclical', 'industry': 'Home Improvement Retail'},
    {'ticker': 'MCD', 'sector': 'Consumer Cyclical', 'industry': 'Restaurants'},
    {'ticker': 'NKE', 'sector': 'Consumer Cyclical', 'industry': 'Footwear & Accessories'},
    
    # Financials
    {'ticker': 'JPM', 'sector': 'Financials', 'industry': 'Banks - Diversified'},
    {'ticker': 'V', 'sector': 'Financials', 'industry': 'Credit Services'},
    {'ticker': 'MA', 'sector': 'Financials', 'industry': 'Credit Services'},
    {'ticker': 'BAC', 'sector': 'Financials', 'industry': 'Banks - Diversified'},
    {'ticker': 'BRK.B', 'sector': 'Financials', 'industry': 'Insurance - Diversified'},
    
    # Healthcare
    {'ticker': 'LLY', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General'},
    {'ticker': 'UNH', 'sector': 'Healthcare', 'industry': 'Healthcare Plans'},
    {'ticker': 'JNJ', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General'},
    {'ticker': 'MRK', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General'},
    {'ticker': 'ABBV', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General'},
    
    # Consumer Defensive
    {'ticker': 'PG', 'sector': 'Consumer Defensive', 'industry': 'Household & Personal Products'},
    {'ticker': 'COST', 'sector': 'Consumer Defensive', 'industry': 'Discount Stores'},
    {'ticker': 'WMT', 'sector': 'Consumer Defensive', 'industry': 'Discount Stores'},
    {'ticker': 'KO', 'sector': 'Consumer Defensive', 'industry': 'Beverages - Non-Alcoholic'},
    {'ticker': 'PEP', 'sector': 'Consumer Defensive', 'industry': 'Beverages - Non-Alcoholic'},
    
    # Energy
    {'ticker': 'XOM', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated'},
    {'ticker': 'CVX', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated'},
    
    # Industrials
    {'ticker': 'CAT', 'sector': 'Industrials', 'industry': 'Farm & Heavy Construction Machinery'},
    {'ticker': 'UNP', 'sector': 'Industrials', 'industry': 'Railroads'},
    {'ticker': 'GE', 'sector': 'Industrials', 'industry': 'Specialty Industrial Machinery'},
]

def create_fallback_metadata():
    print(f"Creating fallback metadata with {len(FALLBACK_DATA)} known tickers...")
    
    df = pd.DataFrame(FALLBACK_DATA)
    # Add company_name as filler
    df['company_name'] = df['ticker'] + " Inc."
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print("✅ Fallback metadata created successfully.")
    print("Run your strategies/dashboard again to see accurate sectors for these tickers.")

if __name__ == "__main__":
    create_fallback_metadata()
