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


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Quant Data Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600)
def load_risk_metrics() -> pd.DataFrame:
    """Load risk metrics from Gold layer"""
    path = GOLD_DIR / 'risk_metrics_lakehouse'
    parquet_files = list(path.glob('*.parquet')) if path.exists() else []
    
    if parquet_files:
        return pd.read_parquet(parquet_files[0])
    
    # Fallback: create sample data
    return create_sample_risk_metrics()


@st.cache_data(ttl=3600)
def load_sector_metrics() -> pd.DataFrame:
    """Load sector-level metrics"""
    path = GOLD_DIR / 'sector_risk_metrics_lakehouse'
    parquet_files = list(path.glob('*.parquet')) if path.exists() else []
    
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
        'volatility_pct': np.random.uniform(15, 45, len(tickers)),
        'max_drawdown_pct': np.random.uniform(-50, -10, len(tickers)),
        'beta': np.random.uniform(0.5, 1.5, len(tickers)),
        'alpha': np.random.uniform(-0.1, 0.2, len(tickers)),
    })


def create_sample_sector_metrics() -> pd.DataFrame:
    """Create sample sector data"""
    return pd.DataFrame({
        'sector': GICS_SECTORS,
        'num_tickers': np.random.randint(50, 500, len(GICS_SECTORS)),
        'avg_sharpe': np.random.uniform(0.8, 1.8, len(GICS_SECTORS)),
        'avg_volatility_pct': np.random.uniform(18, 35, len(GICS_SECTORS)),
        'avg_max_drawdown_pct': np.random.uniform(-40, -15, len(GICS_SECTORS)),
    })


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("📊 Quant Data Platform")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📈 Risk Metrics", "🏢 Sector Analysis", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Status")
    
    # Check data availability
    risk_data_exists = (GOLD_DIR / 'risk_metrics_lakehouse').exists()
    st.sidebar.markdown(f"Risk Metrics: {'✅' if risk_data_exists else '⚠️ Sample'}")
    
    sector_data_exists = (GOLD_DIR / 'sector_risk_metrics_lakehouse').exists()
    st.sidebar.markdown(f"Sector Metrics: {'✅' if sector_data_exists else '⚠️ Sample'}")
    
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
        avg_vol = risk_df['volatility_pct'].mean()
        st.metric(
            "Avg Volatility",
            f"{avg_vol:.1f}%",
            help="Average annualized volatility"
        )
    
    with col4:
        avg_mdd = risk_df['max_drawdown_pct'].mean()
        st.metric(
            "Avg Max Drawdown",
            f"{avg_mdd:.1f}%",
            help="Average maximum drawdown"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sharpe Ratio Distribution")
        fig = px.histogram(
            risk_df, 
            x='sharpe_ratio',
            nbins=30,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sector Performance")
        fig = px.bar(
            sector_df.sort_values('avg_sharpe', ascending=True),
            x='avg_sharpe',
            y='sector',
            orientation='h',
            color='avg_sharpe',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performers Table
    st.subheader("🏆 Top 10 by Sharpe Ratio")
    top_10 = risk_df.nlargest(10, 'sharpe_ratio')[
        ['ticker', 'sector', 'sharpe_ratio', 'volatility_pct', 'beta']
    ]
    st.dataframe(top_10, use_container_width=True)


# =============================================================================
# PAGE: RISK METRICS
# =============================================================================
def render_risk_metrics():
    """Render risk metrics page"""
    st.title("📈 Risk Metrics Dashboard")
    
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
    st.subheader("📈 Risk-Return Profile")
    fig = px.scatter(
        filtered_df,
        x='volatility_pct',
        y='sharpe_ratio',
        color='sector',
        hover_data=['ticker', 'beta', 'max_drawdown_pct'],
        labels={
            'volatility_pct': 'Volatility (%)',
            'sharpe_ratio': 'Sharpe Ratio'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Beta Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Beta Distribution")
        fig = px.histogram(
            filtered_df,
            x='beta',
            nbins=30,
            color_discrete_sequence=['#2ecc71']
        )
        fig.add_vline(x=1, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Max Drawdown Distribution")
        fig = px.histogram(
            filtered_df,
            x='max_drawdown_pct',
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
    st.subheader("📊 Sector Comparison")
    
    metrics = ['avg_sharpe', 'avg_volatility_pct', 'avg_max_drawdown_pct', 'num_tickers']
    selected_metric = st.selectbox(
        "Select Metric",
        metrics,
        format_func=lambda x: {
            'avg_sharpe': 'Average Sharpe Ratio',
            'avg_volatility_pct': 'Average Volatility (%)',
            'avg_max_drawdown_pct': 'Average Max Drawdown (%)',
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
    st.subheader("🔍 Sector Drill-Down")
    selected_sector = st.selectbox("Select Sector", sorted(risk_df['sector'].unique()))
    
    sector_stocks = risk_df[risk_df['sector'] == selected_sector]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks", len(sector_stocks))
    with col2:
        st.metric("Avg Sharpe", f"{sector_stocks['sharpe_ratio'].mean():.2f}")
    with col3:
        st.metric("Avg Vol", f"{sector_stocks['volatility_pct'].mean():.1f}%")
    
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
# MAIN
# =============================================================================
def main():
    """Main app entry point"""
    page = render_sidebar()
    
    if page == "🏠 Overview":
        render_overview()
    elif page == "📈 Risk Metrics":
        render_risk_metrics()
    elif page == "🏢 Sector Analysis":
        render_sector_analysis()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
