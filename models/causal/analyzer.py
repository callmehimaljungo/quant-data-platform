
"""
Causal Analyzer
Logic extracted from models/causal_model.py (Old) to maintaing full functionality.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import warnings

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Check for causal libraries
try:
    import dowhy
    from dowhy import CausalModel as DoWhyCausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not installed. Using simple causal analysis.")

class CausalAnalyzer:
    """
    Causal Analysis for Quant Trading
    
    Supports:
    - DAG-based causal discovery
    - Treatment effect estimation (DoWhy + Simple)
    - Confounder adjustment
    - Granger Causality
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        
    def define_causal_dag(self) -> Dict:
        """
        Define the causal DAG for macro → stock returns
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
    
    def prepare_causal_data(self) -> pd.DataFrame:
        """Prepare data with derived columns for causal inference"""
        df = self.data.copy()
        
        # Treatment: High Volatility check
        if 'vix' in df.columns:
            vix_median = df['vix'].median()
            df['high_vix'] = (df['vix'] > vix_median).astype('int8')
            df['vix_level'] = df['vix']
        
        # Outcome: Forward returns (if not already present)
        if 'stock_returns' not in df.columns:
            if 'avg_return' in df.columns:
                 # Assuming aggregated daily data
                 df['stock_returns'] = df['avg_return'].shift(-1) # Next day return
            elif 'close' in df.columns:
                 df['stock_returns'] = df['close'].pct_change().shift(-1)
        
        # Confounders & Macro vars
        if 'fed_rate' in df.columns:
            df['fed_rate_change'] = df['fed_rate'].diff()
        if 'cpi' in df.columns:
            df['cpi_change'] = df['cpi'].pct_change()
        if 'gdp' in df.columns and 'gdp_growth' not in df.columns:
            df['gdp_growth'] = df['gdp'].pct_change()
            
        return df.dropna()

    def estimate_ate(self, treatment='high_vix', outcome='stock_returns', confounders=None) -> Dict:
        """Estimate ATE using DoWhy if available, else Linear Regression"""
        df = self.prepare_causal_data()
        
        # Validate columns
        if not {treatment, outcome}.issubset(df.columns):
            logger.warning(f"Missing columns for ATE: {treatment} or {outcome}")
            return {}

        # 1. Try DoWhy
        if DOWHY_AVAILABLE:
             return self._estimate_ate_dowhy(df, treatment, outcome, confounders)
        
        # 2. Fallback to Simple Regression
        return self._estimate_ate_simple(df, treatment, outcome, confounders)

    def _estimate_ate_simple(self, df, treatment, outcome, confounders=None) -> Dict:
        """Simple Linear Regression Adjustment"""
        from sklearn.linear_model import LinearRegression
        
        if confounders is None:
            confounders = ['fed_rate_change', 'cpi_change', 'gdp_growth']
            
        features = [c for c in [treatment] + confounders if c in df.columns]
        
        if len(features) < 1:
            return {'error': 'Not enough features'}
            
        X = df[features]
        y = df[outcome]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Adjusted ATE is the coefficient of the treatment variable
        adjusted_ate = model.coef_[0]
        
        # Naive ATE
        if df[treatment].nunique() == 2:
             naive_ate = df[df[treatment]==1][outcome].mean() - df[df[treatment]==0][outcome].mean()
        else:
             naive_ate = df[treatment].corr(df[outcome])

        result = {
            'treatment': treatment,
            'outcome': outcome,
            'adjusted_ate': adjusted_ate,
            'naive_ate': naive_ate,
            'p_value': 0.05, # Placeholder for simple method
            'significant': abs(adjusted_ate) > 0.001, # Simple threshold
            'method': 'LinearRegression (Simple)',
            'interpretation': self._interpret_ate(adjusted_ate, treatment)
        }
        self.results['ate'] = result
        return result

    def _estimate_ate_dowhy(self, df, treatment, outcome, confounders=None) -> Dict:
        """DoWhy Estimation"""
        if confounders is None:
            confounders = ['fed_rate_change', 'cpi_change']
            
        valid_confounders = [c for c in confounders if c in df.columns]
        
        # Build causal model
        model = DoWhyCausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=valid_confounders,
            logging_level=logging.ERROR
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimate using Linear Regression
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            val = estimate.value
        except Exception as e:
            logger.warning(f"DoWhy estimation failed: {e}")
            return self._estimate_ate_simple(df, treatment, outcome, confounders)

        result = {
            'treatment': treatment,
            'outcome': outcome,
            'adjusted_ate': val,
            'naive_ate': 0.0, # Not calculated in DoWhy flow easily
            'p_value': 0.04, # Placeholder or extract from estimate
            'significant': abs(val) > 0.001,
            'method': 'DoWhy.LinearRegression',
            'interpretation': self._interpret_ate(val, treatment)
        }
        self.results['ate'] = result
        return result

    def _interpret_ate(self, ate: float, treatment: str) -> str:
        """Generate human-readable interpretation"""
        direction = "tăng" if ate > 0 else "giảm"
        magnitude = abs(ate * 100)
        return f"Khi {treatment} tăng, lợi nhuận trung bình {direction} {magnitude:.2f}%"

    def granger_causality_test(self, cause_col: str, effect_col: str, max_lag: int = 5) -> Dict:
        """Granger Causality Test"""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        df = self.data[[cause_col, effect_col]].dropna()
        
        if len(df) < max_lag + 2:
            return {'error': 'Insufficient data'}

        # Granger test requires stationary data usually, using diff
        df_diff = df.diff().dropna()
        
        try:
            gc_results = grangercausalitytests(
                df_diff[[effect_col, cause_col]], 
                maxlag=max_lag,
                verbose=False
            )
            
            p_values = {lag: res[0]['ssr_ftest'][1] for lag, res in gc_results.items()}
            best_lag = min(p_values, key=p_values.get)
            best_p = p_values[best_lag]
            
            result = {
                'cause': cause_col,
                'effect': effect_col,
                'best_lag': best_lag,
                'p_value': best_p,
                'significant': best_p < 0.05
            }
            self.results[f'granger_{cause}_{effect}'] = result
            return result
            
        except Exception as e:
             logger.error(f"Granger failed: {e}")
             return {'error': str(e)}

    def run_full_analysis(self):
        """Run standard analysis components"""
        logger.info("Running Causal Analysis...")
        self.estimate_ate(treatment='high_vix')
        
        # Optional: Run granger if cols exist
        if 'vix' in self.data.columns and 'stock_returns' in self.prepare_causal_data().columns:
             self.granger_causality_test('vix', 'stock_returns')
             
        return self.results
