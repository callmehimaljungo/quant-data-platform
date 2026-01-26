
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
risk_df = load_risk_metrics(cache_key)

if risk_df is None:
    st.error("❌ Không thể tải dữ liệu. Vui lòng kiểm tra kết nối R2 hoặc chạy pipeline Local.")
    st.stop()

# Derive Market Stats
median_sharpe = risk_df['sharpe_ratio'].median()
market_regime = "🟢 Bull Market" if median_sharpe > 1.0 else ("🔴 Bear Market" if median_sharpe < 0.5 else "🟡 Neutral")

# Quality Signals (thay cho Volatility & Max Drawdown vô nghĩa)
high_sharpe_count = len(risk_df[risk_df['sharpe_ratio'] > 2.0])  # Opportunities
quality_stocks = len(risk_df[risk_df['sharpe_ratio'] > 1.0])    # Mã chất lượng (Sharpe > 1)
# Median Drawdown (lọc bỏ mã phá sản < -95%)
valid_dd = risk_df[risk_df['max_drawdown'] > -95]['max_drawdown']
median_dd = valid_dd.median() if len(valid_dd) > 0 else -50.0

# =============================================================================
# MARKET PULSE HEADER
# =============================================================================
st.markdown(f"### ⚡ Nhịp Thị Trường")
col1, col2, col_search = st.columns([1, 1, 2])

with col1:
    st.metric("Tổng số Mã", f"{len(risk_df):,}")

with col2:
    st.metric("Sharpe Thị Trường (TB)", f"{median_sharpe:.2f}", delta_color="normal")

with col_search:
    # Ticker Search Feature
    search_ticker = st.selectbox(
        "🔍 Tra cứu mã nhanh",
        options=[""] + sorted(risk_df['ticker'].tolist()),
        index=0,
        placeholder="Nhập mã (VD: AAPL, TSLA...)",
        label_visibility="collapsed"
    )
    
    if search_ticker:
        t_data = risk_df[risk_df['ticker'] == search_ticker].iloc[0]
        # Modern display for searched ticker
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 10px; border-radius: 8px; border: 1px solid #333;">
            <span style="font-size: 1.2rem; font-weight: bold; color: #2ecc71;">{search_ticker}</span> | 
            <b>Sharpe:</b> {t_data['sharpe_ratio']:.2f} | 
            <b>Biến động:</b> {t_data['volatility']:.1f}% | 
            <b>Sụt giảm (MaxDD):</b> {t_data['max_drawdown']:.1f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Tìm kiếm mã để xem nhanh các chỉ số rủi ro.")

st.markdown("---")

# =============================================================================
# STRATEGY SIGNALS (TABS)
# =============================================================================
st.subheader("🎯 Strategy Signals")
tab1, tab2, tab3 = st.tabs(["🏆 High Sharpe Alpha", "🛡️ Low Volatility Defense", "🔥 Momentum Movers"])

with tab1:
    st.markdown("**Top cổ phiếu có hiệu suất điều chỉnh rủi ro tốt nhất (Sharpe > 2.0)**")
    top_sharpe = risk_df.nlargest(20, 'sharpe_ratio')[['ticker', 'sector', 'sharpe_ratio', 'volatility', 'max_drawdown']]
    st.dataframe(format_dataframe(top_sharpe), use_container_width=True, column_config=get_column_config())

with tab2:
    st.markdown("**Top cổ phiếu biến động thấp nhất (Volatility < 20%)**")
    low_vol = risk_df.nsmallest(20, 'volatility')[['ticker', 'sector', 'sharpe_ratio', 'volatility', 'max_drawdown']]
    st.dataframe(format_dataframe(low_vol), use_container_width=True, column_config=get_column_config())
    
with tab3:
    st.markdown("**Top cổ phiếu tăng trưởng mạnh nhất (Daily Return High)**")
    # Proxy momentum by avg_ret if available
    if 'avg_ret' in risk_df.columns:
        risk_df['est_annual_ret'] = risk_df['avg_ret'] * 252 * 100
        top_mom = risk_df.nlargest(20, 'est_annual_ret')[['ticker', 'sector', 'est_annual_ret', 'volatility', 'sharpe_ratio']]
        st.dataframe(format_dataframe(top_mom), use_container_width=True, column_config=get_column_config())
    else:
        st.info("Momentum data not available yet.")

# =============================================================================
# SECTOR OVERVIEW
# =============================================================================
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Phân phối Sharpe Ratio")
    fig_hist = px.histogram(risk_df, x='sharpe_ratio', nbins=50, title="Market Breadth (Sharpe Distribution)", color_discrete_sequence=['#2ecc71'])
    fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("🏢 Ngành dẫn sóng")
    if 'sector' in risk_df.columns:
        sector_perf = risk_df.groupby('sector')['sharpe_ratio'].median().sort_values(ascending=False).head(10)
        fig_bar = px.bar(sector_perf, orientation='h', title="Top Sectors vs Median Sharpe", color=sector_perf.values, color_continuous_scale='Viridis')
        fig_bar.update_layout(height=350, showlegend=False, xaxis_title="Median Sharpe")
        st.plotly_chart(fig_bar, use_container_width=True)
