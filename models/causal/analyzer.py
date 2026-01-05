
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

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
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        
    def define_causal_dag(self) -> Dict:
        """Define the causal DAG for macro → stock returns"""
        dag = {
            'nodes': ['fed_rate_change', 'vix_level', 'cpi_change', 'gdp_growth', 'stock_returns'],
            'edges': [
                ('fed_rate_change', 'vix_level'),
                ('fed_rate_change', 'stock_returns'),
                ('vix_level', 'stock_returns'),
                ('gdp_growth', 'stock_returns'),
                ('cpi_change', 'fed_rate_change'),
            ],
            'confounders': ['gdp_growth', 'cpi_change'],
            'treatment': 'vix_level',
            'outcome': 'stock_returns'
        }
        return dag
    
    def prepare_causal_data(self) -> pd.DataFrame:
        """Prepare data columns"""
        df = self.data.copy()
        
        # Treatment: High Volatility check
        if 'vix' in df.columns:
            vix_median = df['vix'].median()
            df['high_vix'] = (df['vix'] > vix_median).astype('int8')
            df['vix_level'] = df['vix']
            
        # Outcome: Forward returns
        if 'avg_return' in df.columns:
             # Assuming aggregated data
             df['stock_returns'] = df['avg_return'].shift(-1) # Next day return
        
        # Confounders
        if 'fed_rate' in df.columns:
            df['fed_rate_change'] = df['fed_rate'].diff()
        if 'cpi' in df.columns:
            df['cpi_change'] = df['cpi'].pct_change()
            
        return df.dropna()

    def estimate_ate(self, treatment='high_vix', outcome='stock_returns') -> Dict:
        """Estimate ATE using available methods"""
        df = self.prepare_causal_data()
        
        # Simple Linear Regression Adjustment
        from sklearn.linear_model import LinearRegression
        
        confounders = ['fed_rate_change', 'cpi_change', 'gdp']
        features = [c for c in [treatment] + confounders if c in df.columns]
        
        if len(features) < 2:
            return {'error': 'Not enough features'}
            
        X = df[features]
        y = df[outcome]
        
        model = LinearRegression()
        model.fit(X, y)
        
        ate = model.coef_[0]
        
        result = {
            'treatment': treatment,
            'outcome': outcome,
            'adjusted_ate': ate,
            'naive_ate': df[df[treatment]==1][outcome].mean() - df[df[treatment]==0][outcome].mean(),
            'interpretation': f"ATE: {ate:.4f}"
        }
        self.results['ate'] = result
        return result
    
    def run_full_analysis(self):
        """Run all analysis steps"""
        logger.info("Running Causal Analysis...")
        self.estimate_ate()
        # Add Granger etc. here
        return self.results
