
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd

# =============================================================================
# SETUP & CONFIG
# =============================================================================

# Setup Path First
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Env
load_dotenv(PROJECT_ROOT / '.env')

# Page configuration
st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports (must be after path setup)
from dashboard.utils.data_loader import (
    load_risk_metrics, 
    get_cache_key, 
    load_sector_metrics,
    create_sample_sector_metrics
)
from dashboard.utils.formatting import format_dataframe, get_column_config
from dashboard.components.sidebar import render_sidebar

# Render Sidebar
render_sidebar()

# Page Title
st.title("🏠 Quant Dashboard")

# =============================================================================
# DATA LOADING
# =============================================================================
cache_key = get_cache_key()
# This function loads from R2 if configured and available
risk_df = load_risk_metrics(cache_key)

if risk_df is None:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra kết nối R2 hoặc chạy pipeline Local.")
    st.stop()

# =============================================================================
# KPI Cards
# =============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Tổng số Ticker",
        f"{len(risk_df):,}",
        help="Số lượng cổ phiếu được phân tích"
    )

with col2:
    avg_sharpe = risk_df['sharpe_ratio'].median()
    st.metric(
        "Sharpe Ratio (Median)",
        f"{avg_sharpe:.2f}",
        help="Sharpe Ratio (dùng Median để loại bỏ outlier)"
    )

with col3:
    # Filter only extreme outliers (> 500%)
    valid_vol = risk_df[risk_df['volatility'] < 500]['volatility']
    avg_vol = valid_vol.median() if len(valid_vol) > 0 else risk_df['volatility'].median()
        
    st.metric(
        "Volatility (Median)",
        f"{avg_vol:.1f}%",
        help="Độ biến động hàng năm (Annualized Volatility)"
    )

with col4:
    avg_mdd = risk_df['max_drawdown'].median()
    st.metric(
        "Max Drawdown (Median)",
        f"{avg_mdd:.1f}%",
        help="Mức sụt giảm tối đa từ đỉnh"
    )

st.markdown("---")

# =============================================================================
# Charts
# =============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Phân phối Sharpe Ratio")
    fig = px.histogram(
        risk_df, 
        x='sharpe_ratio',
        nbins=30,
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("📈 Hiệu suất theo Sector")
    # Derive sector metrics
    if 'sector' in risk_df.columns:
        sector_agg = risk_df.groupby('sector').agg({
            'sharpe_ratio': 'median',
            'ticker': 'count'
        }).reset_index()
        sector_agg.columns = ['sector', 'sharpe_ratio', 'num_tickers']
        sector_agg = sector_agg.sort_values('sharpe_ratio', ascending=True)
    else:
        sector_agg = create_sample_sector_metrics().sort_values('sharpe_ratio', ascending=True)
        
    fig = px.bar(
        sector_agg,
        x='sharpe_ratio',
        y='sector',
        orientation='h',
        color='sharpe_ratio',
        color_continuous_scale='RdYlGn',
        hover_data=['num_tickers'] if 'num_tickers' in sector_agg.columns else None
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, width='stretch')

# =============================================================================
# Top Performers Table
# =============================================================================
st.subheader("🏆 Top 10 theo Sharpe Ratio")
display_cols = ['ticker', 'sector', 'sharpe_ratio', 'volatility', 'max_drawdown']
available_cols = [c for c in display_cols if c in risk_df.columns]
top_10 = risk_df.nlargest(10, 'sharpe_ratio')[available_cols]

st.dataframe(
    format_dataframe(top_10), 
    width='stretch', 
    column_config=get_column_config()
)
