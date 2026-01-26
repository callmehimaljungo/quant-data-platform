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

if risk_df is not None and 'sector' in risk_df.columns:
    unknown_count = len(risk_df[risk_df['sector'] == 'Unknown'])
    total_count = len(risk_df)
    if total_count > 0 and (unknown_count / total_count) > 0.2:
         st.warning(f"⚠️ **Thông báo Hệ thống**: {unknown_count}/{total_count} mã ticker đang hiển thị Sector 'Unknown'.")

st.subheader("📊 So sánh Ngành (Sector)")

metrics_options = ['sharpe_ratio', 'volatility', 'max_drawdown', 'num_tickers']
selected_metric = st.selectbox(
    label="Chọn chỉ số",
    options=metrics_options,
    format_func=lambda x: {
        'sharpe_ratio': 'Sharpe Ratio trung bình',
        'volatility': 'Volatility trung bình',
        'max_drawdown': 'Max Drawdown trung bình',
        'num_tickers': 'Số lượng Ticker'
    }.get(x, x),
    key="sb_selector_metric_v4"
)

fig_bar = px.bar(
    sector_df.sort_values(selected_metric, ascending=False),
    x='sector',
    y=selected_metric,
    color=selected_metric,
    color_continuous_scale='RdYlGn' if 'sharpe' in selected_metric else 'Blues'
)
fig_bar.update_layout(height=400, xaxis_tickangle=-45)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("🔍 Chi tiết Sector")

if risk_df is not None and 'sector' in risk_df.columns:
    # DEFINITIVE FIX: Filter out empty/null sectors to remove the "ghost/blank" option
    valid_sector_names = risk_df['sector'].dropna().unique().tolist()
    valid_sector_names = [s for s in valid_sector_names if str(s).strip() != ""]
    sector_options = ["— Tất cả Ngành —"] + sorted(valid_sector_names)
    
    final_selected_sector = st.selectbox(
        label="Chọn Sector để xem chi tiết",
        options=sector_options,
        key="sb_sector_drilldown_v4", # Increment key
        label_visibility="visible"
    )
    
    if final_selected_sector == "— Tất cả Ngành —":
        sector_stocks = risk_df
    else:
        sector_stocks = risk_df[risk_df['sector'] == final_selected_sector]
    
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.metric("Số lượng Cổ phiếu", len(sector_stocks))
    with s_col2:
        st.metric("Sharpe TB", f"{sector_stocks['sharpe_ratio'].mean():.2f}")
    with s_col3:
        st.metric("Biến động TB (Vol)", f"{sector_stocks['volatility'].mean():.1f}%")
    
    st.dataframe(
        format_dataframe(sector_stocks.sort_values('sharpe_ratio', ascending=False).head(20)),
        use_container_width=True,
        column_config=get_column_config()
    )
else:
    st.info("Không có dữ liệu ngành.")
