"""
FRED Economic Data Collector
Fetches historical economic indicators from Federal Reserve Economic Data

Free API: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
from typing import Dict, List
import pandas as pd
from datetime import datetime
import logging
import requests

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class FREDCollector:
    """Collector for FRED economic data"""
    
    # Key economic indicators
    SERIES = {
        # GDP & Growth
        'GDP': 'GDP',  # Gross Domestic Product (Quarterly, from 1947)
        'GDP_Growth': 'A191RL1Q225SBEA',  # Real GDP Growth Rate (Quarterly)
        
        # Employment
        'Unemployment_Rate': 'UNRATE',  # Unemployment Rate (Monthly, from 1948)
        'Nonfarm_Payrolls': 'PAYEMS',  # Total Nonfarm Payrolls (Monthly, from 1939)
        
        # Inflation
        'CPI': 'CPIAUCSL',  # Consumer Price Index (Monthly, from 1947)
        'Core_CPI': 'CPILFESL',  # CPI Less Food & Energy (Monthly, from 1957)
        'PCE': 'PCE',  # Personal Consumption Expenditures (Monthly, from 1959)
        
        # Interest Rates
        'Fed_Funds_Rate': 'DFF',  # Federal Funds Effective Rate (Daily, from 1954)
        'Treasury_10Y': 'GS10',  # 10-Year Treasury Rate (Daily, from 1962)
        'Treasury_2Y': 'GS2',  # 2-Year Treasury Rate (Daily, from 1976)
        'Treasury_3M': 'TB3MS',  # 3-Month Treasury Bill (Monthly, from 1934)
        
        # Market Indicators
        'SP500': 'SP500',  # S&P 500 Index (Daily, from 1927)
        'VIX': 'VIXCLS',  # CBOE Volatility Index (Daily, from 1990)
        
        # Housing
        'Housing_Starts': 'HOUST',  # Housing Starts (Monthly, from 1959)
        'Home_Sales': 'HSN1F',  # New Home Sales (Monthly, from 1963)
        
        # Consumer
        'Retail_Sales': 'RSXFS',  # Retail Sales (Monthly, from 1992)
        'Consumer_Sentiment': 'UMCSENT',  # U of Michigan Consumer Sentiment (Monthly, from 1978)
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key required. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    def fetch_series(self, series_id: str, start_date: str = '1962-01-01') -> pd.DataFrame:
        """Fetch a single economic series using direct HTTP request"""
        try:
            # Use FRED API directly without fredapi library
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'observation_start': start_date
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"FRED API error for {series_id}: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            observations = data.get('observations', [])
            
            if not observations:
                logger.warning(f"No observations for {series_id}")
                return pd.DataFrame()
            
            # Parse observations
            records = []
            for obs in observations:
                value_str = obs.get('value')
                # Skip missing values (marked as '.')
                if value_str == '.':
                    continue
                    
                try:
                    records.append({
                        'date': pd.to_datetime(obs.get('date')),
                        'value': float(value_str),
                        'series_id': series_id
                    })
                except (ValueError, TypeError):
                    continue
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_all(self, start_date: str = '1962-01-01') -> Dict[str, pd.DataFrame]:
        """Fetch all economic indicators"""
        logger.info(f"Fetching {len(self.SERIES)} economic series from FRED")
        logger.info(f"Start date: {start_date}")
        
        results = {}
        
        for name, series_id in self.SERIES.items():
            logger.info(f"  Fetching: {name} ({series_id})...")
            
            df = self.fetch_series(series_id, start_date)
            
            if not df.empty:
                df['indicator'] = name
                results[name] = df
                logger.info(f"    ✅ Got {len(df):,} observations")
            else:
                logger.warning(f"    ⚠️ No data")
        
        logger.info(f"\n✅ Fetched {len(results)}/{len(self.SERIES)} series")
        
        return results
    
    def save_to_bronze(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Save economic data to Bronze layer"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all series
        all_data = []
        for name, df in data.items():
            all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined['ingested_at'] = datetime.now()
            
            output_file = output_dir / 'economic_indicators.parquet'
            combined.to_parquet(output_file, index=False)
            
            logger.info(f"\n💾 Saved {len(combined):,} records to {output_file}")
            
            # Summary stats
            logger.info(f"\n📊 Summary:")
            logger.info(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")
            logger.info(f"   Indicators: {combined['indicator'].nunique()}")
            logger.info(f"   Total records: {len(combined):,}")
            
            return output_file
        
        return None


if __name__ == "__main__":
    # Test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True, help='FRED API key')
    parser.add_argument('--start-date', default='1962-01-01', help='Start date')
    args = parser.parse_args()
    
    from config import BRONZE_DIR
    
    collector = FREDCollector(api_key=args.api_key)
    data = collector.fetch_all(start_date=args.start_date)
    
    output_dir = BRONZE_DIR / 'economic_lakehouse'
    collector.save_to_bronze(data, output_dir)
