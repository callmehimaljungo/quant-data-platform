
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.r2_loader import load_latest_from_lakehouse, is_r2_available
from dashboard.components.sidebar import render_sidebar

render_sidebar()

# Try import causal model logic if needed, or just display results
# For dashboard, we typically just LOAD results.

st.title("🔬 Mô hình Machine Learning")

st.markdown("""
Trang này hiển thị kết quả từ các mô hình ML bao gồm:
- **Causal Analysis**: Phân tích nhân quả VIX → Returns
- **Sector Rotation**: Phân bổ danh mục theo chu kỳ kinh tế
""")

tab1, tab2 = st.tabs(["📊 Causal Analysis", "🔄 Sector Rotation"])

with tab1:
    st.subheader("Phân tích Nhân quả (Causal Analysis)")
    st.markdown("""
    **Câu hỏi nghiên cứu**: Yếu tố nào thực sự **GÂY RA** thay đổi lợi nhuận cổ phiếu?
    """)
    
    # Check R2 availability
    r2_ready = False
    try: r2_ready = is_r2_available()
    except: pass
    
    df = None
    if r2_ready:
        df = load_latest_from_lakehouse('processed/gold/causal_analysis_lakehouse/')
    if df is None:
        # Fallback to local
        try:
            local_path = Path("data/gold/causal_analysis_lakehouse/latest_causal_metrics.parquet")
            if local_path.exists():
                df = pd.read_parquet(local_path)
        except Exception:
            pass
    
    if df is not None and len(df) > 0:
            df['treatment_clean'] = df['treatment'].str.replace('high_', '').str.replace('_', ' ').str.title()
            df['ate_pct'] = df['adjusted_ate'] * 100
            
            st.markdown("### 📊 Tác động Nhân quả (ATE)")
            # Color by Sign of ATE (Positive = Green, Negative = Red)
            df['color_type'] = df['ate_pct'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
            
            fig = px.bar(df.sort_values('ate_pct'), 
                        x='ate_pct', y='treatment_clean',
                        orientation='h',
                        color='color_type',
                        color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B'},
                        labels={'ate_pct': 'Average Treatment Effect (%)', 
                               'treatment_clean': 'Treatment'},
                        title='Tác động của các yếu tố lên lợi nhuận cổ phiếu')
            fig.update_layout(height=max(400, len(df) * 50), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, width='stretch')
            
            st.markdown("### 💡 Giải thích kết quả")
            for _, row in df.iterrows():
                treatment = row['treatment_clean']
                ate = row['ate_pct']
                p_val = row['p_value']
                sig = row['significant']
                
                if sig:
                    st.success(f"**{treatment}** có tác động **có ý nghĩa** (p={p_val:.4f}). ATE = {ate:+.2f}%")
                else:
                    st.info(f"**{treatment}** không có tác động rõ ràng (p={p_val:.4f}). ATE = {ate:+.2f}%")
            
            with st.expander("View Raw Data"):
                st.dataframe(df, width='stretch')
    else:
        st.info("💡 Chưa có kết quả Causal Analysis. Vui lòng chạy pipeline.")
        st.code("python -m models.causal.main")
        
        st.markdown("**Kết quả mẫu (Sample):**")
        sample_causal = pd.DataFrame({
            'Treatment': ['News Sentiment', 'VIX', 'Dollar Index'],
            'ATE (%)': [21.47, 3.33, 2.07],
            'Significant': ['YES ✓', 'no', 'no']
        })
        st.dataframe(sample_causal, width='stretch')

with tab2:
    st.subheader("Phân bổ theo Chu kỳ Kinh tế (Sector Rotation)")
    st.markdown("""
    **Chiến lược**: Sector Rotation theo Business Cycle (Fidelity Research)
    - Xác định giai đoạn kinh tế hiện tại dựa trên chỉ số VIX (Fear Index)
    - Phân bổ vốn vào các ngành phù hợp với từng giai đoạn:
      - **Recovery**: Technology, Consumer Cyclical (kinh tế hồi phục)
      - **Expansion**: Energy, Industrials (kinh tế tăng trưởng)
      - **Recession**: Healthcare, Utilities, Consumer Defensive (phòng thủ)
    """)
    
    sr_df = None
    if r2_ready:
        sr_df = load_latest_from_lakehouse('processed/gold/sector_rotation_lakehouse/')
    
    if sr_df is None:
        # Fallback to local
        try:
            local_sr_path = Path("data/gold/sector_rotation_lakehouse")
            if local_sr_path.exists():
                parquet_files = sorted(local_sr_path.glob("*.parquet"))
                if parquet_files:
                    sr_df = pd.read_parquet(parquet_files[-1])
        except Exception:
            pass
            
    if sr_df is not None and not sr_df.empty:
        # Display current regime
        current_regime = sr_df['regime'].iloc[0] if 'regime' in sr_df.columns else "Unknown"
        regime_colors = {'expansion': '🟢', 'peak': '🟡', 'recession': '🔴', 'recovery': '🔵'}
        regime_icon = regime_colors.get(current_regime.lower(), '⚪')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Giai đoạn Kinh tế", f"{regime_icon} {current_regime.title()}")
        with col2:
            st.metric("Số cổ phiếu", len(sr_df))
        with col3:
            n_sectors = sr_df['sector'].nunique() if 'sector' in sr_df.columns else 0
            st.metric("Số ngành", n_sectors)
        
        # Sector allocation chart
        if 'sector' in sr_df.columns:
            sector_weights = sr_df.groupby('sector').size().reset_index(name='count')
            fig_sr = px.pie(
                sector_weights,
                values='count',
                names='sector',
                title=f'Phân bổ Danh mục theo Ngành (Regime: {current_regime.title()})',
                hole=0.4
            )
            fig_sr.update_layout(height=500)
            st.plotly_chart(fig_sr, use_container_width=True)
        
        # Explanation
        st.info(f"""
        **Giải thích**: Với giai đoạn **{current_regime.title()}** hiện tại, hệ thống khuyến nghị 
        tập trung vào các ngành có hiệu suất tốt trong chu kỳ này theo nghiên cứu của Fidelity và NBER.
        """)
        
        # Full table
        with st.expander("Xem chi tiết danh mục"):
            st.dataframe(sr_df, use_container_width=True)
    else:
        st.warning("💡 Chưa có kết quả Sector Rotation. Vui lòng chạy pipeline.")
        st.code("python -m gold.sector_rotation")
