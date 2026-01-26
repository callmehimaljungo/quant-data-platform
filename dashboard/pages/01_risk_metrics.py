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
f_col1, f_col2 = st.columns(2)

with f_col1:
    # DEFINITIVE FIX: Filter out empty/null sectors to remove the "ghost/blank" option
    if 'sector' in risk_df.columns:
        valid_sectors = risk_df['sector'].dropna().unique().tolist()
        valid_sectors = [s for s in valid_sectors if str(s).strip() != ""]
        sector_list = ['Tất cả'] + sorted(valid_sectors)
    else:
        sector_list = ['Tất cả']
        
    selected_sector = st.selectbox(
        label="Lọc theo Ngành (Sector)",
        options=sector_list,
        key="sb_sector_filter_v4", # Increment key to force reset
        label_visibility="visible"
    )

with f_col2:
    min_s = float(risk_df['sharpe_ratio'].min())
    max_s = float(risk_df['sharpe_ratio'].max())
    sharpe_min = st.slider(
        label="Sharpe Ratio tối thiểu",
        min_value=min_s,
        max_value=max_s,
        value=min_s,
        key="sl_sharpe_min_v4"
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
fig_scatter = px.scatter(
    filtered_df,
    x='volatility',
    y='sharpe_ratio',
    color='sector' if 'sector' in filtered_df.columns else None,
    hover_data=['ticker', 'max_drawdown'],
    labels={'volatility': 'Biến động (Volatility) %', 'sharpe_ratio': 'Sharpe Ratio'}
)
fig_scatter.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
fig_scatter.update_layout(height=500, hovermode='closest')
st.plotly_chart(fig_scatter, use_container_width=True)

# Distributions
d_col1, d_col2 = st.columns(2)

with d_col1:
    st.subheader("📊 Phân phối Biến động")
    fig_vol = px.histogram(
        filtered_df,
        x='volatility',
        nbins=30,
        color_discrete_sequence=['#2ecc71'],
        labels={'volatility': 'Volatility (%)'}
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with d_col2:
    st.subheader("📉 Phân phối Sụt giảm (MaxDD)")
    fig_dd = px.histogram(
        filtered_df,
        x='max_drawdown',
        nbins=30,
        color_discrete_sequence=['#e74c3c'],
        labels={'max_drawdown': 'Max Drawdown (%)'}
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# =============================================================================
# Data Table
# =============================================================================
st.subheader("📋 Chi tiết Chỉ số Rủi ro")
final_df = format_dataframe(filtered_df.sort_values('sharpe_ratio', ascending=False))
st.dataframe(
    final_df,
    use_container_width=True,
    column_config=get_column_config()
)
