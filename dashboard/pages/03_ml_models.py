
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
- **Feature Importance**: Xếp hạng các yếu tố quan trọng dự báo giá
""")

tab1, tab2 = st.tabs(["📊 Causal Analysis", "🌲 Feature Importance"])

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
            fig = px.bar(df.sort_values('ate_pct'), 
                        x='ate_pct', y='treatment_clean',
                        orientation='h',
                        color='significant',
                        color_discrete_map={True: '#00CC96', False: '#EF553B'},
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
    st.subheader("Tầm quan trọng của yếu tố (Feature Importance)")
    st.markdown("""
    **Mô hình**: Random Forest Classifier (Dự báo xu hướng giá cổ phiếu).
    Bảng dưới đây hiển thị các yếu tố kỹ thuật và vĩ mô có ảnh hưởng lớn nhất đến biến động giá.
    """)
    
    fi_df = None
    if r2_ready:
        fi_df = load_latest_from_lakehouse('processed/gold/feature_importance_lakehouse/')
    
    if fi_df is None:
        # Fallback to local
        try:
            local_fi_path = Path("data/gold/feature_importance_lakehouse/latest_feature_importance.parquet")
            if local_fi_path.exists():
                fi_df = pd.read_parquet(local_fi_path)
        except Exception:
            pass
            
    if fi_df is not None and not fi_df.empty:
        # Style the feature names
        fi_df['feature_display'] = fi_df['feature'].str.replace('_', ' ').str.title()
        
        fig_fi = px.bar(
            fi_df.head(15).sort_values('importance', ascending=True),
            x='importance',
            y='feature_display',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues',
            labels={'importance': 'Độ quan trọng (Importance)', 'feature_display': 'Yếu tố'},
            title='Top 15 Yếu tố quan trọng nhất'
        )
        fig_fi.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_fi, width='stretch')
        
        with st.expander("Ghi chú về các yếu tố"):
            st.info("""
            - **RSI/MACD**: Chỉ số kỹ thuật cho biết trạng thái quá mua/quá bán.
            - **VIX**: Chỉ số đo lường sự sợ hãi của thị trường.
            - **Sentiment**: Tâm lý thị trường từ tin tức và mạng xã hội.
            - **Returns_L1**: Lợi nhuận của ngày hôm trước (tính quán tính).
            """)
    else:
        st.warning("💡 Chưa có kết quả Feature Importance. Hệ thống đang hiển thị dữ liệu mẫu.")
        sample_fi = pd.DataFrame({
            'feature': ['RSI_14', 'VIX_Level', 'News_Sentiment', 'EMA_50', 'Daily_Returns_L1', 'Volume_MA_10', 'Dollar_Index', 'MACD_Signal'],
            'importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
        }).sort_values('importance', ascending=False)
        
        fig_sample = px.bar(
            sample_fi, x='importance', y='feature', orientation='h', 
            color='importance', color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_sample, width='stretch')
        st.markdown("Chạy lệnh sau để tính toán lại: `python -m models.causal.main`")
