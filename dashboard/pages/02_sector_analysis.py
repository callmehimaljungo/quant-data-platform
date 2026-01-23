
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.express as px
from dashboard.utils.data_loader import load_risk_metrics, load_sector_metrics, get_cache_key
from dashboard.utils.formatting import format_dataframe, get_column_config
from dashboard.components.sidebar import render_sidebar

render_sidebar()

st.title("🏢 Phân tích Sector")

cache_key = get_cache_key()
risk_df = load_risk_metrics(cache_key)
sector_df = load_sector_metrics(cache_key)

# Check for Rate Limit / Missing Metadata indication
if 'sector' in risk_df.columns:
    unknown_count = len(risk_df[risk_df['sector'] == 'Unknown'])
    total_count = len(risk_df)
    if total_count > 0 and (unknown_count / total_count) > 0.2:
         st.warning(
             f"⚠️ **Thông báo Hệ thống**: {unknown_count}/{total_count} mã ticker đang hiển thị Sector 'Unknown'.\n\n"
             "**Nguyên nhân**: API Yahoo Finance đang bị giới hạn (Rate Limit/Quota Exceeded) nên chưa thể tải metadata.\n"
             "**Khắc phục**: Hệ thống sẽ tự động thử lại sau. Hiện tại đang sử dụng dữ liệu dự phòng cho các mã lớn."
         )

# Sector comparison
st.subheader("📊 So sánh Ngành (Sector)")

metrics = ['sharpe_ratio', 'volatility', 'max_drawdown', 'num_tickers']
selected_metric = st.selectbox(
    "Chọn chỉ số",
    metrics,
    format_func=lambda x: {
        'sharpe_ratio': 'Sharpe Ratio trung bình',
        'volatility': 'Volatility trung bình',
        'max_drawdown': 'Max Drawdown trung bình',
        'num_tickers': 'Số lượng Ticker'
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
st.plotly_chart(fig, width='stretch')

# Sector drill-down
st.subheader("🔍 Chi tiết Sector")
# Handle case where sector column missing
if 'sector' in risk_df.columns:
    sectors = sorted(risk_df['sector'].unique())
else:
    sectors = []
    
if sectors:
    # Add "All Sectors" option
    sectors_with_all = ["— Tất cả Ngành —"] + sectors
    selected_sector = st.selectbox("Chọn Sector", sectors_with_all)
    
    if selected_sector == "— Tất cả Ngành —":
        sector_stocks = risk_df
    else:
        sector_stocks = risk_df[risk_df['sector'] == selected_sector]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số lượng Cổ phiếu", len(sector_stocks))
    with col2:
        st.metric("Sharpe TB", f"{sector_stocks['sharpe_ratio'].mean():.2f}")
    with col3:
        # Volatility is already in percent format
        st.metric("Biến động TB (Vol)", f"{sector_stocks['volatility'].mean():.1f}%")
    
    st.dataframe(
        format_dataframe(sector_stocks.sort_values('sharpe_ratio', ascending=False).head(20)),
        width='stretch',
        column_config=get_column_config()
    )
else:
    st.info("Không có dữ liệu ngành.")
