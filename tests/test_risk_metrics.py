"""
Unit Tests for Gold Layer Risk Metrics

Tests for core financial calculations:
- Sharpe Ratio
- Max Drawdown
- Beta calculation
- Sortino Ratio
- VaR 95%

Run with: pytest tests/test_risk_metrics.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# HELPER FUNCTIONS (copied from risk_metrics.py for testing)
# =============================================================================
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05, 
                           trading_days: int = 252) -> float:
    """Calculate annualized Sharpe Ratio"""
    if len(returns) < 30 or returns.std() == 0:
        return np.nan
    daily_rf = risk_free_rate / trading_days
    excess_return = returns.mean() - daily_rf
    return (excess_return / returns.std()) * np.sqrt(trading_days)


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate Maximum Drawdown from price series"""
    if len(prices) < 2:
        return np.nan
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate Beta relative to market"""
    if len(stock_returns) < 30 or len(market_returns) < 30:
        return np.nan
    
    # Align series
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    
    stock = aligned.iloc[:, 0]
    market = aligned.iloc[:, 1]
    
    cov = stock.cov(market)
    var = market.var()
    
    if var == 0:
        return np.nan
    
    return cov / var


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05,
                            trading_days: int = 252) -> float:
    """Calculate Sortino Ratio (using downside deviation)"""
    if len(returns) < 30:
        return np.nan
    
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    
    daily_rf = risk_free_rate / trading_days
    return ((returns.mean() - daily_rf) / downside.std()) * np.sqrt(trading_days)


def calculate_var_95(returns: pd.Series) -> float:
    """Calculate Value at Risk at 95% confidence"""
    if len(returns) < 30:
        return np.nan
    return returns.quantile(0.05)


# =============================================================================
# TEST CLASSES
# =============================================================================
class TestSharpeRatio:
    """Tests for Sharpe Ratio calculation"""
    
    def test_positive_returns(self):
        """Sharpe should be positive for consistent positive returns"""
        # Average daily return of 0.1%, low volatility
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.005, 100))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sharpe > 0, "Sharpe should be positive for positive excess returns"
    
    def test_negative_returns(self):
        """Sharpe should be negative for negative returns"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.002, 0.005, 100))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
        assert sharpe < 0, "Sharpe should be negative for negative excess returns"
    
    def test_zero_volatility_returns_nan(self):
        """Sharpe should return NaN for zero volatility"""
        returns = pd.Series([0.001] * 50)  # Constant returns = zero std
        sharpe = calculate_sharpe_ratio(returns)
        assert np.isnan(sharpe), "Sharpe should be NaN for zero volatility"
    
    def test_insufficient_data(self):
        """Sharpe should return NaN for < 30 data points"""
        returns = pd.Series([0.01, 0.02, -0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert np.isnan(sharpe), "Sharpe should be NaN for insufficient data"
    
    def test_reasonable_range(self):
        """Sharpe should be in reasonable range (-5 to 5)"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        sharpe = calculate_sharpe_ratio(returns)
        assert -5 < sharpe < 5, f"Sharpe {sharpe} outside reasonable range"


class TestMaxDrawdown:
    """Tests for Maximum Drawdown calculation"""
    
    def test_no_drawdown(self):
        """Monotonically increasing prices should have 0 drawdown"""
        prices = pd.Series([100, 110, 120, 130, 140, 150])
        mdd = calculate_max_drawdown(prices)
        assert mdd == 0, "Max drawdown should be 0 for always increasing prices"
    
    def test_50_percent_drawdown(self):
        """Test known 50% drawdown"""
        prices = pd.Series([100, 120, 60, 80])  # Peak 120, trough 60
        mdd = calculate_max_drawdown(prices)
        assert mdd == pytest.approx(-0.5, rel=0.01), f"Expected -0.5, got {mdd}"
    
    def test_recovery_doesnt_reset(self):
        """Drawdown should still reflect worst point even after recovery"""
        prices = pd.Series([100, 50, 200])  # 50% drop then recovery
        mdd = calculate_max_drawdown(prices)
        assert mdd == pytest.approx(-0.5, rel=0.01), "MDD should remember worst point"
    
    def test_multiple_drawdowns(self):
        """Should return the maximum (most negative) drawdown"""
        prices = pd.Series([100, 80, 90, 60, 100])  # 20% then 33% drawdowns
        mdd = calculate_max_drawdown(prices)
        # From peak 100 to 60 = -40%
        assert mdd == pytest.approx(-0.4, rel=0.01), f"Expected -0.4, got {mdd}"
    
    def test_negative_result(self):
        """Max drawdown should always be <= 0"""
        np.random.seed(42)
        prices = pd.Series(100 * (1 + np.random.normal(0, 0.01, 100)).cumprod())
        mdd = calculate_max_drawdown(prices)
        assert mdd <= 0, "Max drawdown should be <= 0"


class TestBeta:
    """Tests for Beta calculation"""
    
    def test_perfect_correlation(self):
        """Stock moving exactly with market should have beta ≈ 1"""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.001, 0.01, 100), 
                          index=pd.date_range('2020-01-01', periods=100))
        stock = market.copy()  # Perfect correlation
        beta = calculate_beta(stock, market)
        assert beta == pytest.approx(1.0, rel=0.01), f"Beta should be 1, got {beta}"
    
    def test_double_volatility(self):
        """Stock with 2x market moves should have beta ≈ 2"""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.001, 0.01, 100),
                          index=pd.date_range('2020-01-01', periods=100))
        stock = market * 2  # 2x moves
        beta = calculate_beta(stock, market)
        assert beta == pytest.approx(2.0, rel=0.1), f"Beta should be 2, got {beta}"
    
    def test_negative_beta(self):
        """Inverse relationship should have negative beta"""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.001, 0.01, 100),
                          index=pd.date_range('2020-01-01', periods=100))
        stock = -market  # Inverse
        beta = calculate_beta(stock, market)
        assert beta < 0, f"Beta should be negative for inverse correlation, got {beta}"
    
    def test_zero_beta(self):
        """Uncorrelated stock should have beta ≈ 0"""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0, 0.01, 100),
                          index=pd.date_range('2020-01-01', periods=100))
        np.random.seed(123)  # Different seed for independence
        stock = pd.Series(np.random.normal(0, 0.01, 100),
                         index=pd.date_range('2020-01-01', periods=100))
        beta = calculate_beta(stock, market)
        assert -0.5 < beta < 0.5, f"Beta should be near 0, got {beta}"
    
    def test_insufficient_data(self):
        """Beta should return NaN for insufficient data"""
        market = pd.Series([0.01, 0.02, -0.01])
        stock = pd.Series([0.02, 0.01, -0.02])
        beta = calculate_beta(stock, market)
        assert np.isnan(beta), "Beta should be NaN for insufficient data"


class TestSortinoRatio:
    """Tests for Sortino Ratio calculation"""
    
    def test_positive_sortino(self):
        """Sortino should be positive for positive returns"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.005, 100))
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        assert sortino > 0, "Sortino should be positive for positive returns"
    
    def test_sortino_with_mixed_returns(self):
        """Sortino should work with mixed positive/negative returns"""
        np.random.seed(42)
        # Mix of positive and negative returns
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        sortino = calculate_sortino_ratio(returns)
        # With mixed returns, Sortino should be calculable
        assert not np.isnan(sortino), "Sortino should work with mixed returns"
    
    def test_no_downside_returns_nan(self):
        """Sortino should return NaN if no negative returns"""
        returns = pd.Series([0.01, 0.02, 0.015] * 20)  # All positive
        sortino = calculate_sortino_ratio(returns)
        assert np.isnan(sortino), "Sortino should be NaN with no downside"


class TestVaR95:
    """Tests for Value at Risk calculation"""
    
    def test_var_is_negative(self):
        """VaR 95 should typically be negative (worst 5% of returns)"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        var = calculate_var_95(returns)
        assert var < 0, "VaR 95 should be negative for normal distribution"
    
    def test_var_percentile(self):
        """VaR should be close to 5th percentile"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))
        var = calculate_var_95(returns)
        expected = returns.quantile(0.05)
        assert var == pytest.approx(expected, rel=0.01)
    
    def test_insufficient_data(self):
        """VaR should return NaN for insufficient data"""
        returns = pd.Series([0.01, -0.02, 0.005])
        var = calculate_var_95(returns)
        assert np.isnan(var), "VaR should be NaN for insufficient data"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
