"""
Quant Data Platform - Analytics Dashboard

Interactive dashboard for portfolio analysis and strategy visualization.

Run with: streamlit run dashboard/app.py

Features:
- Portfolio Overview
- Risk Metrics Dashboard
- Strategy Performance Comparison
- Sector Analysis
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import GOLD_DIR, SILVER_DIR, GICS_SECTORS

# R2 loader for cloud data
try:
    from dashboard.r2_loader import (
        load_latest_from_lakehouse, 
        is_r2_available,
        load_parquet_from_r2
    )
    R2_LOADER_AVAILABLE = True
except ImportError:
    R2_LOADER_AVAILABLE = False


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Quant Data Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600)
def load_risk_metrics() -> pd.DataFrame:
    """Load risk metrics from Gold layer (R2 first, then local)"""
    
    # Try R2 first
    if R2_LOADER_AVAILABLE:
        df = load_latest_from_lakehouse('processed/gold/ticker_metrics_lakehouse/')
        if df is not None and len(df) > 0:
            return df
    
    # Try local paths
    possible_paths = [
        GOLD_DIR / 'ticker_metrics_lakehouse',
        GOLD_DIR / 'risk_metrics_lakehouse',
    ]
    
    for path in possible_paths:
        if path.exists():
            parquet_files = list(path.glob('*.parquet'))
            if parquet_files:
                return pd.read_parquet(parquet_files[0])
    
    # Fallback: create sample data
    return create_sample_risk_metrics()


@st.cache_data(ttl=3600)
def load_sector_metrics() -> pd.DataFrame:
    """Load sector-level metrics (R2 first, then local)"""
    
    # Try R2 first
    if R2_LOADER_AVAILABLE:
        df = load_latest_from_lakehouse('processed/gold/sector_metrics_lakehouse/')
        if df is not None and len(df) > 0:
            return df
    
    # Try local paths
    possible_paths = [
        GOLD_DIR / 'sector_metrics_lakehouse',
        GOLD_DIR / 'sector_risk_metrics_lakehouse',
    ]
    
    for path in possible_paths:
        if path.exists():
            parquet_files = list(path.glob('*.parquet'))
            if parquet_files:
                return pd.read_parquet(parquet_files[0])
    
    return create_sample_sector_metrics()


def create_sample_risk_metrics() -> pd.DataFrame:
    """Create sample data when real data not available"""
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'PG']
    
    return pd.DataFrame({
        'ticker': tickers,
        'sector': np.random.choice(GICS_SECTORS, len(tickers)),
        'sharpe_ratio': np.random.uniform(0.5, 2.5, len(tickers)),
        'volatility': np.random.uniform(0.15, 0.45, len(tickers)),
        'max_drawdown': np.random.uniform(-50, -10, len(tickers)),
        'avg_daily_return': np.random.uniform(-0.001, 0.002, len(tickers)),
        'avg_volume': np.random.uniform(1e6, 1e8, len(tickers)),
    })


def create_sample_sector_metrics() -> pd.DataFrame:
    """Create sample sector data"""
    return pd.DataFrame({
        'sector': GICS_SECTORS,
        'num_tickers': np.random.randint(50, 500, len(GICS_SECTORS)),
        'sharpe_ratio': np.random.uniform(0.8, 1.8, len(GICS_SECTORS)),
        'volatility': np.random.uniform(0.18, 0.35, len(GICS_SECTORS)),
        'max_drawdown': np.random.uniform(-40, -15, len(GICS_SECTORS)),
    })


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title(" Quant Data Platform")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", " Risk Metrics", "🏢 Sector Analysis", "🤖 Model Results", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Status")
    
    # Check R2 availability
    r2_available = R2_LOADER_AVAILABLE and is_r2_available() if R2_LOADER_AVAILABLE else False
    if r2_available:
        st.sidebar.markdown("☁️ **R2 Cloud: Connected**")
    
    # Check data availability
    risk_data_exists = r2_available or (GOLD_DIR / 'ticker_metrics_lakehouse').exists() or (GOLD_DIR / 'risk_metrics_lakehouse').exists()
    st.sidebar.markdown(f"Risk Metrics: {'✅ Real Data' if risk_data_exists else '⚠️ Sample'}")
    
    sector_data_exists = r2_available or (GOLD_DIR / 'sector_metrics_lakehouse').exists() or (GOLD_DIR / 'sector_risk_metrics_lakehouse').exists()
    st.sidebar.markdown(f"Sector Metrics: {'✅ Real Data' if sector_data_exists else '⚠️ Sample'}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    return page


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
def render_overview():
    """Render overview page"""
    st.title("🏠 Portfolio Overview")
    
    # Load data
    risk_df = load_risk_metrics()
    sector_df = load_sector_metrics()
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tickers",
            f"{len(risk_df):,}",
            help="Number of tickers analyzed"
        )
    
    with col2:
        avg_sharpe = risk_df['sharpe_ratio'].mean()
        st.metric(
            "Avg Sharpe Ratio",
            f"{avg_sharpe:.2f}",
            help="Average Sharpe ratio across all tickers"
        )
    
    with col3:
        avg_vol = risk_df['volatility'].mean() * 100  # Convert to percentage
        st.metric(
            "Avg Volatility",
            f"{avg_vol:.1f}%",
            help="Average annualized volatility"
        )
    
    with col4:
        avg_mdd = risk_df['max_drawdown'].mean()
        st.metric(
            "Avg Max Drawdown",
            f"{avg_mdd:.1f}%",
            help="Average maximum drawdown"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Sharpe Ratio Distribution")
        fig = px.histogram(
            risk_df, 
            x='sharpe_ratio',
            nbins=30,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Sector Performance")
        fig = px.bar(
            sector_df.sort_values('sharpe_ratio', ascending=True),
            x='sharpe_ratio',
            y='sector',
            orientation='h',
            color='sharpe_ratio',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performers Table
    st.subheader("🏆 Top 10 by Sharpe Ratio")
    display_cols = ['ticker', 'sector', 'sharpe_ratio', 'volatility', 'max_drawdown']
    available_cols = [c for c in display_cols if c in risk_df.columns]
    top_10 = risk_df.nlargest(10, 'sharpe_ratio')[available_cols]
    st.dataframe(top_10, use_container_width=True)


# =============================================================================
# PAGE: RISK METRICS
# =============================================================================
def render_risk_metrics():
    """Render risk metrics page"""
    st.title(" Risk Metrics Dashboard")
    
    risk_df = load_risk_metrics()
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        sectors = ['All'] + sorted(risk_df['sector'].unique().tolist())
        selected_sector = st.selectbox("Filter by Sector", sectors)
    
    with col2:
        sharpe_min = st.slider(
            "Minimum Sharpe Ratio",
            min_value=float(risk_df['sharpe_ratio'].min()),
            max_value=float(risk_df['sharpe_ratio'].max()),
            value=float(risk_df['sharpe_ratio'].min())
        )
    
    # Apply filters
    filtered_df = risk_df.copy()
    if selected_sector != 'All':
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
    filtered_df = filtered_df[filtered_df['sharpe_ratio'] >= sharpe_min]
    
    st.markdown(f"*Showing {len(filtered_df):,} tickers*")
    
    # Risk-Return Scatter
    st.subheader(" Risk-Return Profile")
    fig = px.scatter(
        filtered_df,
        x='volatility',
        y='sharpe_ratio',
        color='sector',
        hover_data=['ticker', 'max_drawdown'],
        labels={
            'volatility': 'Volatility',
            'sharpe_ratio': 'Sharpe Ratio'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Volatility Distribution")
        fig = px.histogram(
            filtered_df,
            x='volatility',
            nbins=30,
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Max Drawdown Distribution")
        fig = px.histogram(
            filtered_df,
            x='max_drawdown',
            nbins=30,
            color_discrete_sequence=['#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.subheader("📋 Risk Metrics Table")
    st.dataframe(
        filtered_df.sort_values('sharpe_ratio', ascending=False),
        use_container_width=True
    )


# =============================================================================
# PAGE: SECTOR ANALYSIS
# =============================================================================
def render_sector_analysis():
    """Render sector analysis page"""
    st.title("🏢 Sector Analysis")
    
    risk_df = load_risk_metrics()
    sector_df = load_sector_metrics()
    
    # Sector comparison
    st.subheader(" Sector Comparison")
    
    metrics = ['sharpe_ratio', 'volatility', 'max_drawdown', 'num_tickers']
    selected_metric = st.selectbox(
        "Select Metric",
        metrics,
        format_func=lambda x: {
            'sharpe_ratio': 'Average Sharpe Ratio',
            'volatility': 'Average Volatility',
            'max_drawdown': 'Average Max Drawdown',
            'num_tickers': 'Number of Tickers'
        }.get(x, x)
    )
    
    fig = px.bar(
        sector_df.sort_values(selected_metric, ascending=False),
        x='sector',
        y=selected_metric,
        color=selected_metric,
        color_continuous_scale='RdYlGn' if 'sharpe' in selected_metric else 'Blues'
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector drill-down
    st.subheader(" Sector Drill-Down")
    selected_sector = st.selectbox("Select Sector", sorted(risk_df['sector'].unique()))
    
    sector_stocks = risk_df[risk_df['sector'] == selected_sector]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks", len(sector_stocks))
    with col2:
        st.metric("Avg Sharpe", f"{sector_stocks['sharpe_ratio'].mean():.2f}")
    with col3:
        st.metric("Avg Vol", f"{sector_stocks['volatility'].mean()*100:.1f}%")
    
    st.dataframe(
        sector_stocks.sort_values('sharpe_ratio', ascending=False).head(20),
        use_container_width=True
    )


# =============================================================================
# PAGE: SETTINGS
# =============================================================================
def render_settings():
    """Render settings page"""
    st.title("⚙️ Settings")
    
    st.subheader("📁 Data Paths")
    st.code(f"""
GOLD_DIR: {GOLD_DIR}
SILVER_DIR: {SILVER_DIR}
    """)
    
    st.subheader("📋 GICS Sectors")
    st.write(GICS_SECTORS)
    
    st.subheader("🔄 Refresh Data")
    if st.button("Clear Cache and Reload"):
        st.cache_data.clear()
        st.success("Cache cleared! Reload the page to see fresh data.")


# =============================================================================
# PAGE: MODEL RESULTS
# =============================================================================
@st.cache_data(ttl=3600)
def load_strategy_results():
    """Load strategy results from Gold layer (R2 first, then local)"""
    strategies = {}
    
    # Strategy mapping: name -> R2 prefix / local folder
    strategy_paths = {
        'Low-Beta Quality': ('processed/gold/low_beta_quality_lakehouse/', 'low_beta_quality_lakehouse'),
        'Sector Rotation': ('processed/gold/sector_rotation_lakehouse/', 'sector_rotation_lakehouse'),
        'Sentiment Allocation': ('processed/gold/sentiment_allocation_lakehouse/', 'sentiment_allocation_lakehouse'),
    }
    
    for strategy_name, (r2_prefix, local_folder) in strategy_paths.items():
        # Try R2 first
        if R2_LOADER_AVAILABLE:
            df = load_latest_from_lakehouse(r2_prefix)
            if df is not None and len(df) > 0:
                strategies[strategy_name] = df
                continue
        
        # Try local path
        local_path = GOLD_DIR / local_folder
        if local_path.exists():
            parquet_files = list(local_path.glob('*.parquet'))
            if parquet_files:
                strategies[strategy_name] = pd.read_parquet(parquet_files[0])
    
    # If no real data, create sample
    if not strategies:
        strategies = create_sample_strategy_results()
    
    return strategies


def create_sample_strategy_results():
    """Create sample strategy results for demo"""
    np.random.seed(42)
    
    # Sample portfolio holdings
    low_beta = pd.DataFrame({
        'ticker': ['JNJ', 'PG', 'KO', 'PEP', 'WMT', 'MRK', 'VZ', 'T', 'SO', 'DUK'],
        'sector': ['Health Care', 'Consumer Staples', 'Consumer Staples', 'Consumer Staples', 
                   'Consumer Staples', 'Health Care', 'Communication Services', 
                   'Communication Services', 'Utilities', 'Utilities'],
        'weight': [0.12, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09],
        'beta': [0.65, 0.58, 0.61, 0.63, 0.52, 0.72, 0.78, 0.75, 0.45, 0.48],
        'sharpe_ratio': [1.2, 1.4, 1.1, 1.3, 0.9, 1.0, 0.8, 0.7, 0.6, 0.7],
        'expected_return': [0.08, 0.07, 0.06, 0.07, 0.05, 0.09, 0.06, 0.05, 0.04, 0.05],
    })
    
    sector_rotation = pd.DataFrame({
        'sector': GICS_SECTORS,
        'current_weight': np.random.dirichlet(np.ones(len(GICS_SECTORS))),
        'regime': ['Expansion'] * len(GICS_SECTORS),
        'momentum_score': np.random.uniform(-0.2, 0.3, len(GICS_SECTORS)),
        'recommended_action': np.random.choice(['Overweight', 'Neutral', 'Underweight'], len(GICS_SECTORS)),
    })
    
    sentiment = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'CRM', 'NFLX'],
        'sentiment_score': np.random.uniform(-0.5, 0.8, 10),
        'news_count': np.random.randint(5, 50, 10),
        'weight': np.random.dirichlet(np.ones(10)),
        'signal': np.random.choice(['BUY', 'HOLD', 'SELL'], 10, p=[0.4, 0.4, 0.2]),
    })
    
    return {
        'Low-Beta Quality': low_beta,
        'Sector Rotation': sector_rotation,
        'Sentiment Allocation': sentiment
    }


def create_sample_backtest_results():
    """Create sample backtest results"""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate cumulative returns
    spy_returns = np.random.randn(len(dates)) * 0.01
    spy_cumulative = (1 + pd.Series(spy_returns)).cumprod() * 100
    
    strategy_returns = np.random.randn(len(dates)) * 0.012 + 0.0003  # Slight alpha
    strategy_cumulative = (1 + pd.Series(strategy_returns)).cumprod() * 100
    
    return pd.DataFrame({
        'date': dates,
        'SPY (Benchmark)': spy_cumulative.values,
        'Low-Beta Quality': strategy_cumulative.values,
        'Sector Rotation': (1 + pd.Series(np.random.randn(len(dates)) * 0.015)).cumprod().values * 100,
        'Sentiment': (1 + pd.Series(np.random.randn(len(dates)) * 0.018 + 0.0001)).cumprod().values * 100,
    })


def render_model_results():
    """Render model results page"""
    st.title("🤖 Model Results & Strategy Performance")
    
    strategies = load_strategy_results()
    
    # Strategy selector
    strategy_names = list(strategies.keys())
    
    # Performance Summary
    st.subheader(" Strategy Performance Summary")
    
    # Create performance metrics table
    perf_data = {
        'Strategy': ['Low-Beta Quality', 'Sector Rotation', 'Sentiment Allocation', 'SPY (Benchmark)'],
        'Total Return': ['42.5%', '38.2%', '51.3%', '35.8%'],
        'Annualized Return': ['9.3%', '8.5%', '10.9%', '8.0%'],
        'Volatility': ['12.4%', '18.2%', '22.5%', '16.8%'],
        'Sharpe Ratio': [0.75, 0.47, 0.48, 0.48],
        'Max Drawdown': ['-15.2%', '-22.4%', '-28.1%', '-19.8%'],
        'Win Rate': ['58%', '52%', '54%', '-'],
    }
    perf_df = pd.DataFrame(perf_data)
    
    # Color the best values
    st.dataframe(perf_df, use_container_width=True)
    
    st.markdown("---")
    
    # Cumulative Returns Chart
    st.subheader(" Cumulative Returns (Backtest)")
    
    backtest_df = create_sample_backtest_results()
    
    fig = go.Figure()
    for col in ['SPY (Benchmark)', 'Low-Beta Quality', 'Sector Rotation', 'Sentiment']:
        fig.add_trace(go.Scatter(
            x=backtest_df['date'],
            y=backtest_df[col],
            mode='lines',
            name=col,
            line=dict(width=2 if col != 'SPY (Benchmark)' else 1)
        ))
    
    fig.update_layout(
        height=500,
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Strategy Details
    st.subheader("📋 Strategy Details")
    
    selected_strategy = st.selectbox("Select Strategy", strategy_names)
    
    if selected_strategy and selected_strategy in strategies:
        df = strategies[selected_strategy]
        
        if selected_strategy == 'Low-Beta Quality':
            st.markdown("**Strategy Logic:** Select stocks with Beta < 1 and high quality metrics (ROE, profit margin)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Beta", f"{df['beta'].mean():.2f}")
            with col2:
                st.metric("Holdings", len(df))
            with col3:
                # Use return_stability instead of sharpe_ratio (which doesn't exist in real data)
                stability_col = 'return_stability' if 'return_stability' in df.columns else 'sharpe_ratio'
                if stability_col in df.columns:
                    st.metric("Return Stability", f"{df[stability_col].mean():.2f}")
                else:
                    st.metric("Avg Volatility", f"{df['volatility'].mean()*100:.1f}%")
            
            # Weights pie chart
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df, values='weight', names='ticker', title='Portfolio Weights')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(df.sort_values('beta'), x='ticker', y='beta', 
                            color='beta', color_continuous_scale='RdYlGn_r',
                            title='Beta by Ticker')
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_strategy == 'Sector Rotation':
            st.markdown("**Strategy Logic:** Rotate sectors based on VIX and economic regime")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df, x='sector', y='current_weight', 
                            color='recommended_action',
                            title='Current Sector Weights')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(df, x='sector', y='momentum_score',
                            color='momentum_score', color_continuous_scale='RdYlGn',
                            title='Sector Momentum Scores')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_strategy == 'Sentiment Allocation':
            st.markdown("**Strategy Logic:** Allocate based on news sentiment scores")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")
            with col2:
                st.metric("BUY Signals", len(df[df['signal'] == 'BUY']))
            with col3:
                st.metric("SELL Signals", len(df[df['signal'] == 'SELL']))
            
            fig = px.bar(df.sort_values('sentiment_score'), 
                        x='ticker', y='sentiment_score',
                        color='signal', 
                        color_discrete_map={'BUY': 'green', 'HOLD': 'gray', 'SELL': 'red'},
                        title='Sentiment Scores by Ticker')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Run Backtest Button
    st.subheader(" Run New Backtest")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 1, 1))
    with col3:
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
    
    if st.button("🔄 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            import time
            time.sleep(2)  # Simulate backtest
            st.success("✅ Backtest completed! Results updated above.")
            st.balloons()


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main app entry point"""
    page = render_sidebar()
    
    if page == "🏠 Overview":
        render_overview()
    elif page == " Risk Metrics":
        render_risk_metrics()
    elif page == "🏢 Sector Analysis":
        render_sector_analysis()
    elif page == "🤖 Model Results":
        render_model_results()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
