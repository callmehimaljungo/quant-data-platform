
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.utils.data_loader import load_risk_metrics, get_cache_key
from dashboard.utils.formatting import format_dataframe, get_column_config
from dashboard.components.sidebar import render_sidebar

render_sidebar()

st.title("📈 Bảng Chỉ số Rủi ro (Risk Metrics)")

# Load Data
cache_key = get_cache_key()
risk_df = load_risk_metrics(cache_key)

if risk_df is None or risk_df.empty:
    st.warning("Chưa có dữ liệu.")
    st.stop()

# =============================================================================
# Filters
# =============================================================================
col1, col2 = st.columns(2)

with col1:
    sectors = ['Tất cả'] + sorted(risk_df['sector'].unique().tolist()) if 'sector' in risk_df.columns else ['Tất cả']
    selected_sector = st.selectbox("Lọc theo Ngành (Sector)", sectors)

with col2:
    min_s = float(risk_df['sharpe_ratio'].min())
    max_s = float(risk_df['sharpe_ratio'].max())
    sharpe_min = st.slider(
        "Sharpe Ratio tối thiểu",
        min_value=min_s,
        max_value=max_s,
        value=min_s
    )

# Apply filters
filtered_df = risk_df.copy()
if selected_sector != 'Tất cả':
    filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
filtered_df = filtered_df[filtered_df['sharpe_ratio'] >= sharpe_min]

st.markdown(f"*Hiển thị {len(filtered_df):,} ticker*")
st.markdown("---")

# =============================================================================
# Visualizations
# =============================================================================

# Risk-Return Scatter
st.subheader("📈 Hồ sơ Rủi ro - Lợi nhuận")
fig = px.scatter(
    filtered_df,
    x='volatility',
    y='sharpe_ratio',
    color='sector' if 'sector' in filtered_df.columns else None,
    hover_data=['ticker', 'max_drawdown'],
    labels={
        'volatility': 'Biến động (Volatility) %',
        'sharpe_ratio': 'Sharpe Ratio'
    }
)
fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(height=500, hovermode='closest')
st.plotly_chart(fig, width='stretch')

# Distributions
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Phân phối Biến động")
    fig = px.histogram(
        filtered_df,
        x='volatility',
        nbins=30,
        color_discrete_sequence=['#2ecc71'],
        labels={'volatility': 'Volatility (%)'}
    )
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("📉 Phân phối Sụt giảm (MaxDD)")
    fig = px.histogram(
        filtered_df,
        x='max_drawdown',
        nbins=30,
        color_discrete_sequence=['#e74c3c'],
        labels={'max_drawdown': 'Max Drawdown (%)'}
    )
    st.plotly_chart(fig, width='stretch')

# =============================================================================
# Data Table
# =============================================================================
st.subheader("📋 Chi tiết Chỉ số Rủi ro")

display_df = format_dataframe(filtered_df.sort_values('sharpe_ratio', ascending=False))
st.dataframe(
    display_df,
    width='stretch',
    column_config=get_column_config()
)
