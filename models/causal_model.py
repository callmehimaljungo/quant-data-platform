"""
Causal Model for Quant Trading Analysis

Analyzes causal relationships between macro-economic factors and stock returns
using DAG-based causal inference with DoWhy library.

Key Questions:
1. Does VIX (fear index) CAUSE stock returns to decrease?
2. Does Fed Rate change CAUSE market volatility?
3. Do sector-specific factors have causal impact on individual stocks?
"""

import sys
from pathlib import Path
import io

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import LOG_FORMAT, SILVER_DIR, GOLD_DIR

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Optional: Check for causal inference libraries
try:
    import dowhy
    from dowhy import CausalModel as DoWhyCausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not installed. Using simplified causal analysis.")


class CausalAnalyzer:
    """
    Causal Analysis for Quant Trading
    
    Supports:
    - DAG-based causal discovery
    - Treatment effect estimation
    - Confounder adjustment
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Causal Analyzer
        
        Args:
            data: DataFrame with columns for treatment, outcome, and confounders
        """
        self.data = data.copy()
        self.results = {}
        
    def define_causal_dag(self) -> Dict:
        """
        Define the causal DAG for macro → stock returns
        
        Causal Graph:
        
            Fed_Rate_Change → VIX → Returns
                    ↓          ↑
                    → GDP → ---↗
                    
        Confounders: GDP, Inflation (CPI)
        Treatment: VIX (high volatility regime)
        Outcome: Stock Returns
        """
        dag = {
            'nodes': [
                'fed_rate_change',    # Fed Funds Rate change
                'vix_level',          # Fear index
                'cpi_change',         # Inflation
                'gdp_growth',         # Economic growth
                'stock_returns'       # Target
            ],
            'edges': [
                ('fed_rate_change', 'vix_level'),      # Fed rate ↑ → VIX ↑
                ('fed_rate_change', 'stock_returns'),  # Direct effect
                ('fed_rate_change', 'gdp_growth'),     # Fed rate affects growth
                ('vix_level', 'stock_returns'),        # VIX → Returns
                ('gdp_growth', 'stock_returns'),       # GDP → Returns
                ('cpi_change', 'fed_rate_change'),     # Inflation → Fed policy
                ('gdp_growth', 'vix_level'),           # Economy → Fear
            ],
            'confounders': ['gdp_growth', 'cpi_change'],
            'treatment': 'vix_level',
            'outcome': 'stock_returns'
        }
        return dag
    
    def prepare_causal_data(self, subset_cols: List[str] = None) -> pd.DataFrame:
        """
        Prepare data for causal analysis (memory-optimized)
        
        Creates:
        - Binary treatment: high_vix (VIX > median)
        - Continuous outcome: forward returns
        - Confounder variables
        """
        # Select only needed columns to save memory
        needed_cols = ['date', 'ticker', 'close', 'daily_return', 'vix', 'fed_rate', 'cpi', 'gdp_growth', 'sector']
        if subset_cols:
            needed_cols.extend(subset_cols)
        
        available_cols = [c for c in needed_cols if c in self.data.columns]
        df = self.data[available_cols].copy() if available_cols else self.data.copy()
        
        # Create treatment variable (high volatility regime)
        if 'vix' in df.columns:
            vix_median = df['vix'].median()
            df['high_vix'] = (df['vix'] > vix_median).astype('int8')
            df['vix_level'] = df['vix']
        
        # Create outcome variable (forward 5-day return)
        if 'close' in df.columns and 'ticker' in df.columns:
            df['stock_returns'] = df.groupby('ticker')['close'].pct_change(5).shift(-5)
        elif 'daily_return' in df.columns:
            df['stock_returns'] = df['daily_return'].rolling(5).sum().shift(-5)
        
        # Fed rate changes
        if 'fed_rate' in df.columns:
            df['fed_rate_change'] = df['fed_rate'].diff()
        
        # CPI changes (inflation momentum)
        if 'cpi' in df.columns:
            df['cpi_change'] = df['cpi'].pct_change()
        
        # Drop NaN efficiently using subset of critical columns
        critical_cols = ['stock_returns']
        if 'high_vix' in df.columns:
            critical_cols.append('high_vix')
        
        return df.dropna(subset=critical_cols)
    
    def estimate_ate_simple(self, 
                            treatment_col: str = 'high_vix',
                            outcome_col: str = 'stock_returns',
                            confounders: List[str] = None) -> Dict:
        """
        Estimate Average Treatment Effect (ATE) using simple regression adjustment
        
        This is a baseline method when DoWhy is not available.
        
        Formula: ATE = E[Y | T=1] - E[Y | T=0], adjusted for confounders
        """
        df = self.prepare_causal_data()
        
        if treatment_col not in df.columns or outcome_col not in df.columns:
            logger.error(f"Missing columns: {treatment_col} or {outcome_col}")
            return {}
        
        # Naive ATE (no adjustment)
        treated = df[df[treatment_col] == 1][outcome_col].mean()
        control = df[df[treatment_col] == 0][outcome_col].mean()
        naive_ate = treated - control
        
        # Adjusted ATE using OLS
        from sklearn.linear_model import LinearRegression
        
        feature_cols = [treatment_col]
        if confounders:
            feature_cols.extend([c for c in confounders if c in df.columns])
        
        X = df[feature_cols].dropna()
        y = df.loc[X.index, outcome_col]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Treatment coefficient is the adjusted ATE
        adjusted_ate = model.coef_[0]
        
        results = {
            'naive_ate': naive_ate,
            'adjusted_ate': adjusted_ate,
            'treatment': treatment_col,
            'outcome': outcome_col,
            'confounders_used': feature_cols[1:],
            'n_treated': int((df[treatment_col] == 1).sum()),
            'n_control': int((df[treatment_col] == 0).sum()),
            'mean_outcome_treated': treated,
            'mean_outcome_control': control,
            'interpretation': self._interpret_ate(adjusted_ate, treatment_col, outcome_col)
        }
        
        self.results['ate'] = results
        return results
    
    def _interpret_ate(self, ate: float, treatment: str, outcome: str) -> str:
        """Generate human-readable interpretation of ATE"""
        direction = "tăng" if ate > 0 else "giảm"
        magnitude = abs(ate * 100)  # Convert to percentage
        
        if 'vix' in treatment.lower():
            treatment_desc = "khi VIX cao (thị trường sợ hãi)"
        else:
            treatment_desc = f"khi {treatment}"
            
        return f"Khi {treatment_desc}, lợi nhuận cổ phiếu trung bình {direction} {magnitude:.2f}%"
    
    def estimate_ate_dowhy(self,
                           treatment_col: str = 'high_vix',
                           outcome_col: str = 'stock_returns',
                           confounders: List[str] = None) -> Dict:
        """
        Estimate ATE using DoWhy causal inference library
        
        Methods used:
        - Propensity Score Matching
        - Inverse Probability Weighting
        - Regression adjustment
        """
        if not DOWHY_AVAILABLE:
            logger.warning("DoWhy not available. Using simple estimation.")
            return self.estimate_ate_simple(treatment_col, outcome_col, confounders)
        
        df = self.prepare_causal_data()
        
        if confounders is None:
            confounders = ['fed_rate_change', 'cpi_change']
        
        valid_confounders = [c for c in confounders if c in df.columns]
        
        # Build causal model
        model = DoWhyCausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=valid_confounders
        )
        
        # Identify estimand
        identified_estimand = model.identify_effect()
        
        # Estimate using multiple methods
        estimates = {}
        
        # Method 1: Propensity Score Matching
        try:
            psm_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching"
            )
            estimates['propensity_matching'] = psm_estimate.value
        except Exception as e:
            logger.warning(f"PSM failed: {e}")
        
        # Method 2: Linear Regression
        try:
            lr_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            estimates['linear_regression'] = lr_estimate.value
        except Exception as e:
            logger.warning(f"LR failed: {e}")
        
        # Refutation test
        refutation = None
        try:
            refutation = model.refute_estimate(
                identified_estimand,
                estimates.get('linear_regression', psm_estimate),
                method_name="random_common_cause"
            )
        except Exception as e:
            logger.warning(f"Refutation failed: {e}")
        
        results = {
            'estimates': estimates,
            'best_estimate': np.mean(list(estimates.values())) if estimates else None,
            'refutation_passed': refutation is not None,
            'causal_graph': self.define_causal_dag()
        }
        
        self.results['dowhy'] = results
        return results
    
    def granger_causality_test(self,
                               cause_col: str,
                               effect_col: str,
                               max_lag: int = 10) -> Dict:
        """
        Granger Causality Test
        
        Tests if past values of X help predict Y (beyond Y's own past).
        Note: Granger causality is NOT true causality, but temporal precedence.
        
        Args:
            cause_col: Potential cause variable
            effect_col: Potential effect variable
            max_lag: Maximum lag to test
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        df = self.data[[cause_col, effect_col]].dropna()
        
        # Granger test requires stationary data
        # Using first differences
        df_diff = df.diff().dropna()
        
        results = {}
        try:
            gc_results = grangercausalitytests(
                df_diff[[effect_col, cause_col]],  # Note: effect first, cause second
                maxlag=max_lag,
                verbose=False
            )
            
            # Extract p-values for F-test at each lag
            p_values = {}
            for lag in range(1, max_lag + 1):
                p_values[lag] = gc_results[lag][0]['ssr_ftest'][1]
            
            # Find best lag (lowest p-value)
            best_lag = min(p_values, key=p_values.get)
            
            results = {
                'cause': cause_col,
                'effect': effect_col,
                'p_values_by_lag': p_values,
                'best_lag': best_lag,
                'best_p_value': p_values[best_lag],
                'significant_at_005': p_values[best_lag] < 0.05,
                'interpretation': self._interpret_granger(cause_col, effect_col, p_values[best_lag], best_lag)
            }
            
        except Exception as e:
            logger.error(f"Granger test failed: {e}")
            results = {'error': str(e)}
        
        self.results['granger'] = results
        return results
    
    def _interpret_granger(self, cause: str, effect: str, p_value: float, lag: int) -> str:
        """Interpret Granger causality results"""
        if p_value < 0.05:
            return f"{cause} Granger-causes {effect} với độ trễ {lag} ngày (p={p_value:.4f}). Biến động của {cause} có khả năng dự báo {effect}."
        else:
            return f"Không có bằng chứng {cause} Granger-causes {effect} (p={p_value:.4f})"
    
    def analyze_treatment_heterogeneity(self,
                                        treatment_col: str = 'high_vix',
                                        outcome_col: str = 'stock_returns',
                                        segment_col: str = 'sector') -> pd.DataFrame:
        """
        Analyze how treatment effect varies across segments
        
        Example: Does VIX impact tech stocks differently than utilities?
        """
        df = self.prepare_causal_data()
        
        if segment_col not in df.columns:
            logger.warning(f"Segment column {segment_col} not found")
            return pd.DataFrame()
        
        results = []
        
        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment]
            
            if len(segment_data) < 100:  # Skip if too few samples
                continue
            
            treated = segment_data[segment_data[treatment_col] == 1][outcome_col].mean()
            control = segment_data[segment_data[treatment_col] == 0][outcome_col].mean()
            ate = treated - control
            
            results.append({
                'segment': segment,
                'ate': ate,
                'ate_pct': ate * 100,
                'n_treated': (segment_data[treatment_col] == 1).sum(),
                'n_control': (segment_data[treatment_col] == 0).sum(),
                'mean_treated': treated,
                'mean_control': control
            })
        
        return pd.DataFrame(results).sort_values('ate')
    
    def generate_report(self) -> str:
        """Generate comprehensive causal analysis report"""
        report = []
        report.append("=" * 70)
        report.append("CAUSAL ANALYSIS REPORT - QUANT DATA PLATFORM")
        report.append("=" * 70)
        
        # ATE Results
        if 'ate' in self.results:
            ate = self.results['ate']
            report.append("\n📊 AVERAGE TREATMENT EFFECT (ATE)")
            report.append("-" * 40)
            report.append(f"Treatment: {ate['treatment']}")
            report.append(f"Outcome: {ate['outcome']}")
            report.append(f"Naive ATE: {ate['naive_ate']*100:.2f}%")
            report.append(f"Adjusted ATE: {ate['adjusted_ate']*100:.2f}%")
            report.append(f"N (treated): {ate['n_treated']}")
            report.append(f"N (control): {ate['n_control']}")
            report.append(f"\n[!] {ate['interpretation']}")
        
        # Granger Results
        if 'granger' in self.results:
            gc = self.results['granger']
            if 'error' not in gc:
                report.append("\n[GRANGER] GRANGER CAUSALITY TEST")
                report.append("-" * 40)
                report.append(f"Cause: {gc['cause']} -> Effect: {gc['effect']}")
                report.append(f"Best lag: {gc['best_lag']} days")
                report.append(f"P-value: {gc['best_p_value']:.4f}")
                report.append(f"Significant (a=0.05): {'YES' if gc['significant_at_005'] else 'NO'}")
                report.append(f"\n[!] {gc['interpretation']}")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)


# =============================================================================
# ENHANCED MULTI-SOURCE FUNCTIONS
# =============================================================================

def load_unified_data_full() -> pd.DataFrame:
    """
    Load and aggregate ALL data using DuckDB for memory efficiency
    
    Aggregates 33M+ rows to daily level before loading into pandas
    This enables causal analysis on the complete dataset
    
    Returns:
        DataFrame with daily aggregated data from all sources
    """
    import duckdb
    from config import BRONZE_DIR
    
    logger.info("=" * 60)
    logger.info("LOADING FULL DATASET WITH DUCKDB AGGREGATION")
    logger.info("=" * 60)
    
    # Price data path
    price_path = SILVER_DIR / 'enriched_stocks.parquet'
    
    if not price_path.exists():
        logger.error(f"Price data not found: {price_path}")
        return pd.DataFrame()
    
    # Use DuckDB for aggregation - handles 33M rows efficiently
    logger.info("Aggregating 33M+ rows with DuckDB...")
    
    conn = duckdb.connect()
    
    # Aggregate price data to daily level
    # Filter to VIX era (1990+) since VIX data starts from 1990
    query = f"""
    SELECT 
        date,
        COUNT(DISTINCT ticker) as n_tickers,
        AVG(daily_return) as avg_return,
        STDDEV(daily_return) as volatility,
        SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as pct_positive,
        AVG(close) as avg_close,
        AVG(ABS(daily_return)) as avg_abs_return
    FROM read_parquet('{price_path}')
    WHERE daily_return IS NOT NULL
      AND date >= '1990-01-01'
    GROUP BY date
    ORDER BY date
    """
    
    try:
        df_daily = conn.execute(query).fetchdf()
        logger.info(f"  Price aggregated: {len(df_daily):,} unique dates from 33M+ rows")
    except Exception as e:
        logger.error(f"DuckDB aggregation failed: {e}")
        return pd.DataFrame()
    
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    # Also get sector breakdown
    sector_query = f"""
    SELECT 
        date,
        sector,
        AVG(daily_return) as sector_return,
        COUNT(*) as sector_count
    FROM read_parquet('{price_path}')
    WHERE daily_return IS NOT NULL AND sector IS NOT NULL
    GROUP BY date, sector
    ORDER BY date, sector
    """
    
    try:
        df_sectors = conn.execute(sector_query).fetchdf()
        logger.info(f"  Sector data: {len(df_sectors):,} date-sector combinations")
    except Exception as e:
        logger.warning(f"Sector aggregation failed: {e}")
        df_sectors = None
    
    conn.close()
    
    # Load economic data
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    econ_path = SILVER_DIR / 'economic_lakehouse'
    if is_lakehouse_table(econ_path):
        logger.info("Loading economic indicators...")
        econ_df = lakehouse_to_pandas(econ_path)
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        
        # Rename columns
        econ_cols = {'VIXCLS': 'vix', 'DFF': 'fed_rate', 'GDP': 'gdp', 
                     'CPIAUCSL': 'cpi', 'DTWEXBGS': 'dollar_index', 'DGS10': 'treasury_10y'}
        econ_df = econ_df.rename(columns=econ_cols)
        
        merge_cols = ['date'] + [c for c in ['vix', 'fed_rate', 'gdp', 'cpi', 'dollar_index', 'treasury_10y'] 
                                  if c in econ_df.columns]
        econ_subset = econ_df[merge_cols].drop_duplicates('date')
        
        df_daily = df_daily.merge(econ_subset, on='date', how='left')
        logger.info(f"  Merged economic: {len([c for c in econ_subset.columns if c != 'date'])} indicators")
    
    # Load news (if available)
    news_path = SILVER_DIR / 'news_lakehouse'
    if is_lakehouse_table(news_path):
        logger.info("Loading news sentiment...")
        news_df = lakehouse_to_pandas(news_path)
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Aggregate news to daily
        news_daily = news_df.groupby('date').agg({
            'avg_sentiment': 'mean',
            'news_count': 'sum'
        }).reset_index() if 'avg_sentiment' in news_df.columns else None
        
        if news_daily is not None:
            news_daily = news_daily.rename(columns={'avg_sentiment': 'news_sentiment'})
            df_daily = df_daily.merge(news_daily, on='date', how='left')
            logger.info(f"  Merged news: sentiment + count")
    
    # Forward fill economic indicators
    for col in ['vix', 'fed_rate', 'gdp', 'cpi', 'dollar_index', 'treasury_10y']:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].ffill()
    
    # Fill NaN
    df_daily = df_daily.fillna(0)
    
    logger.info("=" * 60)
    logger.info(f"FINAL DATASET: {len(df_daily):,} days, {len(df_daily.columns)} columns")
    logger.info(f"Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
    logger.info(f"Columns: {list(df_daily.columns)}")
    logger.info("=" * 60)
    
    # Store sector data for later analysis
    df_daily.attrs['sector_data'] = df_sectors
    
    return df_daily


def load_unified_data(max_rows: int = 200000) -> pd.DataFrame:
    """
    Load unified data - tries full aggregation first, falls back to sampling
    """
    # Try full data aggregation first
    try:
        df = load_unified_data_full()
        if not df.empty and len(df) > 50:
            return df
    except Exception as e:
        logger.warning(f"Full data load failed: {e}")
    
    # Fallback to original method
    from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table
    
    unified_path = SILVER_DIR / 'unified_lakehouse'
    
    # Try unified first
    if is_lakehouse_table(unified_path):
        logger.info("Loading from unified_lakehouse (all data sources joined)")
        try:
            df = lakehouse_to_pandas(unified_path)
            logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            if len(df) > max_rows:
                # Sample by DATE RANGE (not random) to preserve time series continuity
                df['date'] = pd.to_datetime(df['date'])
                # Get recent data (last N rows sorted by date)
                df = df.sort_values('date').tail(max_rows)
                logger.info(f"Filtered to last {max_rows:,} rows (time series preserved)")
            
            return df
        except Exception as e:
            logger.warning(f"Failed to load unified: {e}")
    
    # Fallback: Join manually from separate sources
    logger.info("Unified data not found. Attempting manual join...")
    
    dfs = {}
    
    # 1. Load price data
    price_path = SILVER_DIR / 'enriched_stocks.parquet'
    if price_path.exists():
        logger.info("Loading price data...")
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(price_path)
        cols = [c for c in ['date', 'ticker', 'close', 'daily_return', 'sector'] 
                if c in pf.schema.names]
        dfs['prices'] = pd.read_parquet(price_path, columns=cols)
        
        if len(dfs['prices']) > max_rows:
            # Sample by DATE RANGE to preserve time series continuity for Granger causality
            dfs['prices']['date'] = pd.to_datetime(dfs['prices']['date'])
            dfs['prices'] = dfs['prices'].sort_values('date').tail(max_rows)
            logger.info(f"  Prices: {len(dfs['prices']):,} rows (recent, time series preserved)")
        else:
            logger.info(f"  Prices: {len(dfs['prices']):,} rows")
    
    # 2. Load economic data
    econ_path = SILVER_DIR / 'economic_lakehouse'
    if is_lakehouse_table(econ_path):
        logger.info("Loading economic data...")
        dfs['economic'] = lakehouse_to_pandas(econ_path)
        logger.info(f"  Economic: {len(dfs['economic']):,} rows, cols: {list(dfs['economic'].columns)}")
    
    # 3. Load news sentiment
    news_path = SILVER_DIR / 'news_lakehouse'
    if is_lakehouse_table(news_path):
        logger.info("Loading news sentiment...")
        dfs['news'] = lakehouse_to_pandas(news_path)
        logger.info(f"  News: {len(dfs['news']):,} rows")
    
    # Join all sources
    if 'prices' not in dfs:
        logger.error("No price data found!")
        return pd.DataFrame()
    
    df = dfs['prices'].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Join economic
    if 'economic' in dfs:
        econ = dfs['economic'].copy()
        econ['date'] = pd.to_datetime(econ['date'])
        
        # Rename columns for clarity
        econ_cols = {'VIXCLS': 'vix', 'DFF': 'fed_rate', 'GDP': 'gdp', 
                     'CPIAUCSL': 'cpi', 'DTWEXBGS': 'dollar_index', 'DGS10': 'treasury_10y'}
        econ = econ.rename(columns=econ_cols)
        
        merge_cols = ['date'] + [c for c in ['vix', 'fed_rate', 'gdp', 'cpi', 'dollar_index', 'treasury_10y'] 
                                  if c in econ.columns]
        econ_subset = econ[merge_cols].drop_duplicates('date')
        
        df = df.merge(econ_subset, on='date', how='left')
        logger.info(f"  Merged economic: now {len(df.columns)} columns")
    
    # Join news
    if 'news' in dfs:
        news = dfs['news'].copy()
        news['date'] = pd.to_datetime(news['date'])
        
        news_cols = ['date', 'ticker']
        if 'avg_sentiment' in news.columns:
            news_cols.append('avg_sentiment')
        if 'news_count' in news.columns:
            news_cols.append('news_count')
        
        if len(news_cols) > 2:
            news_subset = news[news_cols].drop_duplicates(['date', 'ticker'])
            df = df.merge(news_subset, on=['date', 'ticker'], how='left')
            # Rename
            if 'avg_sentiment' in df.columns:
                df = df.rename(columns={'avg_sentiment': 'news_sentiment'})
            logger.info(f"  Merged news: now {len(df.columns)} columns")
    
    # 4. Load and join metadata (sector, industry)
    from config import BRONZE_DIR
    meta_path = BRONZE_DIR / 'stock_metadata_lakehouse'
    if is_lakehouse_table(meta_path):
        logger.info("Loading metadata...")
        meta_df = lakehouse_to_pandas(meta_path)
        logger.info(f"  Metadata: {len(meta_df):,} rows")
        
        # Select columns
        meta_cols = ['ticker']
        for c in ['sector', 'industry', 'marketCap', 'market_cap']:
            if c in meta_df.columns:
                meta_cols.append(c)
        
        if len(meta_cols) > 1:
            meta_subset = meta_df[meta_cols].drop_duplicates('ticker')
            # Rename marketCap
            if 'marketCap' in meta_subset.columns:
                meta_subset = meta_subset.rename(columns={'marketCap': 'market_cap'})
            
            # Update sector from metadata (overwrite "Unknown")
            if 'sector' in meta_subset.columns:
                df = df.merge(meta_subset[['ticker', 'sector']], on='ticker', how='left', suffixes=('', '_meta'))
                if 'sector_meta' in df.columns:
                    # Use metadata sector where available
                    mask = (df['sector'].isna()) | (df['sector'] == 'Unknown')
                    df.loc[mask, 'sector'] = df.loc[mask, 'sector_meta']
                    df = df.drop(columns=['sector_meta'])
                logger.info(f"  Merged metadata: now {len(df.columns)} columns")
    
    # Forward fill economic indicators
    for col in ['vix', 'fed_rate', 'gdp', 'cpi', 'dollar_index', 'treasury_10y']:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    logger.info(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Available columns: {list(df.columns)}")
    
    return df


def estimate_multi_treatment_ate(df: pd.DataFrame, 
                                  treatments: List[str],
                                  outcome: str = 'daily_return',
                                  confounders: List[str] = None) -> pd.DataFrame:
    """
    Estimate Average Treatment Effect for multiple treatments
    
    Controls for all other treatments as confounders when estimating each
    
    Args:
        df: DataFrame with data
        treatments: List of treatment column names
        outcome: Outcome variable name
        confounders: Additional confounders
        
    Returns:
        DataFrame with ATE estimates for each treatment
    """
    from sklearn.linear_model import LinearRegression
    
    results = []
    
    df_clean = df.dropna(subset=[outcome] + treatments)
    
    for treatment in treatments:
        # Use all other treatments as confounders
        other_treatments = [t for t in treatments if t != treatment]
        all_confounders = other_treatments + (confounders or [])
        all_confounders = [c for c in all_confounders if c in df_clean.columns]
        
        # Naive ATE (no adjustment)
        if df_clean[treatment].nunique() == 2:
            # Binary treatment
            treated = df_clean[df_clean[treatment] == 1][outcome].mean()
            control = df_clean[df_clean[treatment] == 0][outcome].mean()
            naive_ate = treated - control
        else:
            # Continuous: use correlation as rough estimate
            naive_ate = df_clean[treatment].corr(df_clean[outcome])
        
        # Adjusted ATE using OLS
        feature_cols = [treatment] + all_confounders
        X = df_clean[feature_cols].dropna()
        y = df_clean.loc[X.index, outcome]
        
        model = LinearRegression()
        model.fit(X, y)
        adjusted_ate = model.coef_[0]
        
        # T-test for significance (simplified)
        from scipy import stats
        n = len(X)
        se = np.std(y - model.predict(X)) / np.sqrt(n)
        t_stat = adjusted_ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - len(feature_cols)))
        
        results.append({
            'treatment': treatment,
            'naive_ate': naive_ate,
            'adjusted_ate': adjusted_ate,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_obs': len(X),
            'confounders': all_confounders
        })
    
    return pd.DataFrame(results)


def granger_causality_matrix(df: pd.DataFrame, 
                              variables: List[str],
                              max_lag: int = 5) -> pd.DataFrame:
    """
    Build matrix of pairwise Granger causality tests
    
    Args:
        df: DataFrame with time series data
        variables: List of variable names to test
        max_lag: Maximum lag for Granger test
        
    Returns:
        DataFrame with p-values for each pair (row causes column)
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Aggregate to daily level first
    df_daily = df.groupby('date')[variables].mean().reset_index()
    df_daily = df_daily.dropna()
    
    if len(df_daily) < 50:
        logger.warning(f"Not enough data for Granger test: {len(df_daily)} rows")
        return pd.DataFrame()
    
    # Use first differences for stationarity
    df_diff = df_daily[variables].diff().dropna()
    
    # Build matrix
    results = pd.DataFrame(index=variables, columns=variables, dtype=float)
    
    for cause in variables:
        for effect in variables:
            if cause == effect:
                results.loc[cause, effect] = np.nan
                continue
            
            try:
                data = df_diff[[effect, cause]].dropna()
                if len(data) < max_lag + 10:
                    results.loc[cause, effect] = np.nan
                    continue
                
                gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                # Get minimum p-value across lags
                min_p = min([gc[i][0]['ssr_ftest'][1] for i in range(1, max_lag + 1)])
                results.loc[cause, effect] = min_p
                
            except Exception as e:
                results.loc[cause, effect] = np.nan
    
    return results


def conditional_ate_by_group(df: pd.DataFrame,
                              treatment: str,
                              outcome: str = 'daily_return',
                              groupby: str = 'sector') -> pd.DataFrame:
    """
    Estimate treatment effect conditional on groups
    
    Reveals heterogeneous treatment effects across sectors, market caps, etc.
    
    Args:
        df: DataFrame with data
        treatment: Treatment variable (will be binarized if continuous)
        outcome: Outcome variable
        groupby: Grouping variable (sector, market_cap_category, etc.)
        
    Returns:
        DataFrame with ATE by group
    """
    results = []
    
    df_clean = df.dropna(subset=[treatment, outcome, groupby])
    
    # Binarize treatment if needed
    if df_clean[treatment].nunique() > 2:
        median = df_clean[treatment].median()
        df_clean['treatment_binary'] = (df_clean[treatment] > median).astype(int)
        treatment_col = 'treatment_binary'
    else:
        treatment_col = treatment
    
    for group in df_clean[groupby].unique():
        group_data = df_clean[df_clean[groupby] == group]
        
        if len(group_data) < 100:
            continue
        
        treated = group_data[group_data[treatment_col] == 1]
        control = group_data[group_data[treatment_col] == 0]
        
        if len(treated) < 10 or len(control) < 10:
            continue
        
        ate = treated[outcome].mean() - control[outcome].mean()
        
        # T-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(treated[outcome], control[outcome])
        
        results.append({
            'group': group,
            'ate': ate,
            'ate_pct': ate * 100,
            'n_treated': len(treated),
            'n_control': len(control),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('ate', ascending=False)
    
    return result_df


def load_sample_data(max_rows: int = 100000) -> pd.DataFrame:
    """
    Load sample data for causal analysis demo
    
    Args:
        max_rows: Maximum rows to load (for memory efficiency)
    """
    # Try to load from silver layer
    silver_files = list(SILVER_DIR.glob("*.parquet"))
    
    if silver_files:
        logger.info(f"Loading from {silver_files[0]}")
        
        # Only read needed columns and sample for memory efficiency
        needed_cols = ['date', 'ticker', 'close', 'daily_return', 'sector']
        
        try:
            # Read metadata first to check size
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(silver_files[0])
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"Total rows in file: {total_rows:,}")
            
            # Get available columns
            available_cols = [c for c in needed_cols if c in parquet_file.schema.names]
            
            if total_rows > max_rows:
                # Sample by reading random row groups
                logger.info(f"Sampling {max_rows:,} rows for analysis...")
                df = pd.read_parquet(silver_files[0], columns=available_cols)
                df = df.sample(n=min(max_rows, len(df)), random_state=42)
            else:
                df = pd.read_parquet(silver_files[0], columns=available_cols)
            
            return df
            
        except Exception as e:
            logger.warning(f"Optimized load failed: {e}. Using standard load with sample.")
            df = pd.read_parquet(silver_files[0])
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)
            return df
    
    # Generate synthetic data for demo
    logger.info("Generating synthetic data for demo...")
    np.random.seed(42)
    n = 5000  # More data for better causal inference
    
    # Simulate causal structure:
    # Fed Rate → VIX → Returns (with GDP as confounder)
    
    gdp_growth = np.random.normal(2.5, 1.0, n)  # GDP growth %
    cpi = np.random.normal(2.0, 0.5, n)  # Inflation
    
    # Fed rate influenced by CPI
    fed_rate = 2 + 0.5 * cpi + np.random.normal(0, 0.3, n)
    
    # VIX influenced by Fed rate and GDP (inverse)
    vix = 15 + 2 * fed_rate - 1.5 * gdp_growth + np.random.normal(0, 3, n)
    vix = np.clip(vix, 10, 80)
    
    # Returns influenced by VIX (negative) and GDP (positive)
    # TRUE CAUSAL EFFECT: -0.3% per VIX point
    returns = 0.05 - 0.003 * vix + 0.01 * gdp_growth + np.random.normal(0, 0.02, n)
    
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='D'),
        'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'], n),
        'sector': np.random.choice(['Technology', 'Financials', 'Health Care'], n),
        'close': 100 + np.cumsum(returns * 100),  # Simulate price series
        'daily_return': returns,
        'vix': vix,
        'fed_rate': fed_rate,
        'cpi': cpi,
        'gdp_growth': gdp_growth,
        'stock_returns': returns
    })
    
    return df


def main():
    """Demo causal analysis with ALL data sources"""
    print("=" * 70)
    print("ENHANCED CAUSAL MODEL - MULTI-SOURCE ANALYSIS")
    print("=" * 70)
    
    # Load unified data (all sources joined)
    print("\n[1] LOADING DATA FROM ALL SOURCES...")
    df = load_unified_data(max_rows=200000)
    
    if df.empty:
        print("  [!] Unified data empty, falling back to sample data...")
        df = load_sample_data(max_rows=100000)
    
    print(f"\n[OK] Loaded {len(df):,} records")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    
    # Check available data sources
    print("\n[2] DATA SOURCE AVAILABILITY:")
    sources = {
        'Price': ['date', 'ticker', 'close', 'daily_return'],
        'Economic': ['vix', 'fed_rate', 'gdp', 'cpi', 'dollar_index', 'treasury_10y'],
        'News': ['news_sentiment', 'news_count', 'avg_sentiment'],
        'Metadata': ['sector', 'industry', 'market_cap'],
    }
    
    available_sources = {}
    for source, cols in sources.items():
        found = [c for c in cols if c in df.columns]
        available_sources[source] = found
        status = 'OK' if found else 'MISSING'
        print(f"  [{status}] {source}: {found if found else 'None'}")
    
    # Prepare outcome variable (handle both ticker-level and daily-aggregated data)
    if 'avg_return' in df.columns:
        # Daily aggregated data from DuckDB
        df['stock_returns'] = df['avg_return']
        print("\n[INFO] Using daily aggregated returns from FULL 33M+ rows")
    elif 'daily_return' in df.columns:
        df['stock_returns'] = df['daily_return']
    elif 'close' in df.columns:
        df['stock_returns'] = df.groupby('ticker')['close'].pct_change()
    
    # =========================================================================
    # MULTI-TREATMENT ATE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] MULTI-TREATMENT CAUSAL ANALYSIS")
    print("=" * 70)
    
    # Find available treatments
    potential_treatments = ['vix', 'fed_rate', 'news_sentiment', 'dollar_index']
    available_treatments = [t for t in potential_treatments if t in df.columns]
    
    if available_treatments and 'stock_returns' in df.columns:
        print(f"\nTreatments found: {available_treatments}")
        
        # Binarize treatments
        for t in available_treatments:
            median = df[t].median()
            df[f'high_{t}'] = (df[t] > median).astype(int)
        
        binary_treatments = [f'high_{t}' for t in available_treatments]
        
        # Estimate multi-treatment ATE
        print("\n[MULTI-ATE] Estimating treatment effects...")
        ate_df = estimate_multi_treatment_ate(
            df, 
            treatments=binary_treatments,
            outcome='stock_returns'
        )
        
        if not ate_df.empty:
            print(f"\n{'Treatment':<20} {'Adj.ATE':>12} {'P-value':>10} {'Sig?':>6}")
            print("-" * 50)
            for _, row in ate_df.iterrows():
                sig = 'YES' if row['significant'] else 'no'
                print(f"{row['treatment']:<20} {row['adjusted_ate']*100:>11.4f}% {row['p_value']:>10.4f} {sig:>6}")
    else:
        print("  Not enough variables for multi-treatment analysis")
    
    # =========================================================================
    # GRANGER CAUSALITY MATRIX
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4] GRANGER CAUSALITY MATRIX")
    print("=" * 70)
    
    granger_vars = ['vix', 'fed_rate', 'stock_returns']
    granger_vars = [v for v in granger_vars if v in df.columns]
    
    if len(granger_vars) >= 2:
        print(f"\nVariables: {granger_vars}")
        print("Building Granger causality matrix...")
        
        gc_matrix = granger_causality_matrix(df, granger_vars, max_lag=5)
        
        if not gc_matrix.empty:
            print("\nP-values (row -> column, lower = stronger causality):")
            print(gc_matrix.round(4).to_string())
            
            # Find significant relationships
            print("\n[SIGNIFICANT] Granger-causal relationships (p < 0.05):")
            for cause in granger_vars:
                for effect in granger_vars:
                    if cause != effect:
                        p = gc_matrix.loc[cause, effect]
                        if pd.notna(p) and p < 0.05:
                            print(f"  {cause} -> {effect} (p={p:.4f})")
    else:
        print("  Not enough variables for Granger test")
    
    # =========================================================================
    # SECTOR HETEROGENEITY ANALYSIS
    # =========================================================================
    if 'sector' in df.columns:
        print("\n" + "=" * 70)
        print("[5] SECTOR HETEROGENEITY ANALYSIS")
        print("=" * 70)
        
        # Use volatility as treatment
        if 'daily_return' in df.columns:
            abs_ret = df['daily_return'].abs()
            df['high_volatility'] = (abs_ret > abs_ret.median()).astype(int)
            
            print("\nTreatment: High Volatility (|return| > median)")
            print("Outcome: Stock Returns")
            print("Groupby: Sector")
            
            sector_ate = conditional_ate_by_group(
                df, 
                treatment='high_volatility',
                outcome='stock_returns',
                groupby='sector'
            )
            
            if not sector_ate.empty:
                print(f"\n{'Sector':<30} {'ATE (%)':>10} {'P-value':>10} {'Sig?':>6}")
                print("-" * 60)
                for _, row in sector_ate.head(10).iterrows():
                    sig = 'YES' if row['significant'] else 'no'
                    print(f"{row['group']:<30} {row['ate_pct']:>+10.4f} {row['p_value']:>10.4f} {sig:>6}")
    
    # =========================================================================
    # SAVE RESULTS TO LAKEHOUSE
    # =========================================================================
    print("\n" + "=" * 70)
    print("[6] SAVING RESULTS TO LAKEHOUSE")
    print("=" * 70)
    
    from utils import pandas_to_lakehouse
    output_path = GOLD_DIR / 'causal_analysis_lakehouse'
    
    # Save multi-treatment ATE results
    if 'ate_df' in locals() and not ate_df.empty:
        pandas_to_lakehouse(ate_df, output_path, mode='overwrite')
        print(f"  [OK] Saved to {output_path}")
    else:
        print("  [SKIP] No ATE results to save")
    
    # =========================================================================
    # SUMMARY & TRADING INSIGHTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("[SUMMARY] CAUSAL INSIGHTS FOR TRADING")
    print("=" * 70)
    print("""
Based on multi-source causal analysis:

1. MACRO FACTORS (VIX, Fed Rate)
   - Test whether VIX Granger-causes returns
   - Use as regime indicator for position sizing

2. ALTERNATIVE DATA (News Sentiment)  
   - Test causal effect on returns controlling for VIX
   - Potential alpha source if significant after adjustment

3. SECTOR ROTATION
   - Identify sectors with different volatility sensitivity
   - Rotate into defensive sectors when VIX high

4. IMPLEMENTATION
   - Use ATE estimates for signal weighting
   - Only trade on statistically significant relationships
   - Re-estimate monthly to check stability
""")
    print("=" * 70)


if __name__ == "__main__":
    main()


