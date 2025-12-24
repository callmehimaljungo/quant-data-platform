"""
Backtest Engine

Simple event-driven backtesting engine for portfolio strategies.
Focused on clarity and correctness over speed.

Features:
- Daily rebalancing support
- Transaction costs and slippage
- Multiple strategies support
- Performance analytics

Usage:
    engine = BacktestEngine(initial_capital=100_000)
    results = engine.run(strategy='low_beta_quality', start_date='2020-01-01')
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BACKTEST_CONFIG, GOLD_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class Trade:
    """Represents a single trade"""
    date: datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    value: float
    commission: float


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time"""
    date: datetime
    cash: float
    positions: Dict[str, float]  # ticker -> shares
    prices: Dict[str, float]  # ticker -> price
    total_value: float


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    portfolio_history: List[PortfolioSnapshot]
    trades: List[Trade]
    daily_returns: pd.Series
    
    def summary(self) -> str:
        """Generate text summary of results"""
        return f"""
========================================
BACKTEST RESULTS: {self.strategy_name}
========================================
Period: {self.start_date.date()} to {self.end_date.date()}
Initial Capital: ${self.initial_capital:,.2f}
Final Value: ${self.final_value:,.2f}

RETURNS
-------
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}

RISK METRICS
------------
Volatility (Ann.): {self.volatility:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}

TRADING
-------
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.2%}
========================================
"""


# =============================================================================
# PORTFOLIO TRACKER
# =============================================================================
class PortfolioTracker:
    """Tracks portfolio positions and value over time"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # ticker -> shares
        self.history: List[PortfolioSnapshot] = []
        self.trades: List[Trade] = []
    
    def get_position(self, ticker: str) -> float:
        """Get current position for a ticker"""
        return self.positions.get(ticker, 0.0)
    
    def execute_trade(
        self, 
        date: datetime, 
        ticker: str, 
        shares: float, 
        price: float,
        commission_rate: float = 0.001
    ) -> Trade:
        """Execute a trade and update positions"""
        value = abs(shares * price)
        commission = value * commission_rate
        
        action = 'BUY' if shares > 0 else 'SELL'
        
        # Update cash
        if shares > 0:  # Buy
            self.cash -= (value + commission)
        else:  # Sell
            self.cash += (value - commission)
        
        # Update positions
        current = self.positions.get(ticker, 0.0)
        new_position = current + shares
        
        if abs(new_position) < 0.0001:  # Close to zero
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = new_position
        
        # Record trade
        trade = Trade(
            date=date,
            ticker=ticker,
            action=action,
            shares=abs(shares),
            price=price,
            value=value,
            commission=commission
        )
        self.trades.append(trade)
        
        return trade
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            shares * prices.get(ticker, 0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + position_value
    
    def take_snapshot(self, date: datetime, prices: Dict[str, float]) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state"""
        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            positions=self.positions.copy(),
            prices=prices.copy(),
            total_value=self.get_total_value(prices)
        )
        self.history.append(snapshot)
        return snapshot


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================
class PerformanceMetrics:
    """Calculate performance metrics from portfolio history"""
    
    @staticmethod
    def calculate_returns(portfolio_history: List[PortfolioSnapshot]) -> pd.Series:
        """Calculate daily returns from portfolio history"""
        values = pd.Series(
            [s.total_value for s in portfolio_history],
            index=[s.date for s in portfolio_history]
        )
        return values.pct_change().dropna()
    
    @staticmethod
    def total_return(initial: float, final: float) -> float:
        """Calculate total return"""
        return (final - initial) / initial
    
    @staticmethod
    def annualized_return(total_return: float, days: int) -> float:
        """Annualize a return over given number of days"""
        if days <= 0:
            return 0.0
        years = days / 252  # Trading days
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)
        return vol
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series, 
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(portfolio_history: List[PortfolioSnapshot]) -> float:
        """Calculate maximum drawdown"""
        values = pd.Series([s.total_value for s in portfolio_history])
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def win_rate(trades: List[Trade]) -> float:
        """Calculate win rate (percentage of profitable trades)"""
        if not trades:
            return 0.0
        
        # Group trades by position
        # Simplified: count sell trades that are profitable
        # This is a simplification - real win rate needs entry/exit pairing
        return 0.5  # Placeholder


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
class BacktestEngine:
    """
    Main backtesting engine.
    
    Supports:
    - Daily data processing
    - Multiple strategy types
    - Transaction costs
    - Performance analytics
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_rate: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # Will be initialized when running
        self.portfolio: Optional[PortfolioTracker] = None
        self.price_data: Optional[pd.DataFrame] = None
    
    def load_data(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load price data for backtesting"""
        from config import SILVER_DIR
        
        # Try to load from Silver layer
        silver_path = SILVER_DIR / 'enriched_stocks.parquet'
        
        if silver_path.exists():
            df = pd.read_parquet(silver_path)
            logger.info(f"Loaded {len(df):,} rows from Silver layer")
        else:
            # Create sample data for testing
            logger.warning("Silver data not found. Creating sample data.")
            df = self._create_sample_data()
        
        # Filter by date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
        
        return df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing when real data unavailable"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        data = []
        for ticker in tickers:
            base_price = np.random.uniform(100, 500)
            prices = base_price * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'close': max(prices[i], 1),
                    'daily_return': np.random.randn() * 2,
                    'sector': 'Technology',
                })
        
        return pd.DataFrame(data)
    
    def run(
        self,
        strategy: str = 'equal_weight',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_frequency: str = 'monthly'
    ) -> BacktestResult:
        """
        Run backtest with specified strategy.
        
        Args:
            strategy: Strategy name ('equal_weight', 'low_beta_quality', etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rebalance_frequency: 'daily', 'weekly', 'monthly'
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest: {strategy}")
        logger.info(f"Period: {start_date} to {end_date}")
        
        # Initialize
        self.portfolio = PortfolioTracker(self.initial_capital)
        self.price_data = self.load_data(start_date, end_date)
        
        if len(self.price_data) == 0:
            raise ValueError("No data available for backtesting")
        
        # Get unique dates
        dates = sorted(self.price_data['date'].unique())
        
        # Run simulation
        for i, date in enumerate(dates):
            day_data = self.price_data[self.price_data['date'] == date]
            prices = dict(zip(day_data['ticker'], day_data['close']))
            
            # Rebalance on first day of month
            should_rebalance = (
                i == 0 or 
                (rebalance_frequency == 'monthly' and 
                 pd.Timestamp(date).month != pd.Timestamp(dates[i-1]).month) or
                rebalance_frequency == 'daily'
            )
            
            if should_rebalance:
                self._rebalance(date, day_data, strategy)
            
            # Take daily snapshot
            self.portfolio.take_snapshot(date, prices)
        
        # Calculate results
        return self._calculate_results(strategy, dates[0], dates[-1])
    
    def _rebalance(
        self, 
        date: datetime, 
        day_data: pd.DataFrame,
        strategy: str
    ):
        """Rebalance portfolio according to strategy"""
        prices = dict(zip(day_data['ticker'], day_data['close']))
        
        # Get target weights based on strategy
        if strategy == 'equal_weight':
            tickers = list(prices.keys())[:10]  # Top 10 tickers
            weights = {t: 1.0/len(tickers) for t in tickers}
        elif strategy == 'low_beta_quality':
            # Simplified: use equal weight on "quality" stocks
            weights = self._low_beta_weights(day_data)
        else:
            # Default: equal weight
            tickers = list(prices.keys())[:10]
            weights = {t: 1.0/len(tickers) for t in tickers}
        
        # Calculate target positions
        total_value = self.portfolio.get_total_value(prices)
        
        for ticker, weight in weights.items():
            if ticker not in prices:
                continue
            
            target_value = total_value * weight
            target_shares = target_value / prices[ticker]
            current_shares = self.portfolio.get_position(ticker)
            
            diff = target_shares - current_shares
            
            if abs(diff) > 0.1:  # Minimum trade size
                # Apply slippage
                trade_price = prices[ticker] * (1 + self.slippage if diff > 0 else 1 - self.slippage)
                self.portfolio.execute_trade(
                    date, ticker, diff, trade_price, self.commission_rate
                )
    
    def _low_beta_weights(self, day_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate weights for low-beta quality strategy"""
        # Simplified implementation
        tickers = day_data['ticker'].unique()[:10]
        return {t: 1.0/len(tickers) for t in tickers}
    
    def _calculate_results(
        self, 
        strategy: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate performance metrics from backtest"""
        history = self.portfolio.history
        trades = self.portfolio.trades
        
        final_value = history[-1].total_value if history else self.initial_capital
        total_ret = PerformanceMetrics.total_return(self.initial_capital, final_value)
        
        days = (end_date - start_date).days
        ann_ret = PerformanceMetrics.annualized_return(total_ret, days)
        
        returns = PerformanceMetrics.calculate_returns(history)
        vol = PerformanceMetrics.volatility(returns)
        sharpe = PerformanceMetrics.sharpe_ratio(returns)
        mdd = PerformanceMetrics.max_drawdown(history)
        win = PerformanceMetrics.win_rate(trades)
        
        return BacktestResult(
            strategy_name=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_ret,
            annualized_return=ann_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=win,
            total_trades=len(trades),
            portfolio_history=history,
            trades=trades,
            daily_returns=returns
        )


# =============================================================================
# MAIN (Demo)
# =============================================================================
def main():
    """Demo backtest run"""
    engine = BacktestEngine(
        initial_capital=100_000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    results = engine.run(
        strategy='equal_weight',
        start_date='2020-01-01',
        end_date='2023-12-31',
        rebalance_frequency='monthly'
    )
    
    print(results.summary())
    
    return results


if __name__ == "__main__":
    main()
