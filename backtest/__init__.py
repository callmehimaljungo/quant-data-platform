"""
Backtest Module for Portfolio Strategy Evaluation

This module provides backtesting capabilities for:
- Low-Beta Quality Strategy
- Sector Rotation Strategy
- Sentiment-Based Allocation Strategy

Key Components:
- BacktestEngine: Event-driven backtesting engine
- PortfolioTracker: Tracks portfolio value over time
- PerformanceMetrics: Calculates returns, Sharpe, drawdown, etc.

Usage:
    from backtest import BacktestEngine
    
    engine = BacktestEngine(
        initial_capital=100_000,
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    results = engine.run(strategy='low_beta_quality')
    print(results.summary())
"""

from .engine import (
    BacktestEngine,
    PortfolioTracker,
    PerformanceMetrics,
    BacktestResult,
)

__all__ = [
    'BacktestEngine',
    'PortfolioTracker',
    'PerformanceMetrics',
    'BacktestResult',
]
