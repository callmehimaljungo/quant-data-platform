
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
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
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
                st.dataframe(df, use_container_width=True)
    else:
        st.info("💡 Chưa có kết quả Causal Analysis. Vui lòng chạy pipeline.")
        st.code("python models/causal_model.py")
        
        st.markdown("**Kết quả mẫu (Sample):**")
        sample_causal = pd.DataFrame({
            'Treatment': ['News Sentiment', 'VIX', 'Dollar Index'],
            'ATE (%)': [21.47, 3.33, 2.07],
            'Significant': ['YES ✓', 'no', 'no']
        })
        st.dataframe(sample_causal, use_container_width=True)

with tab2:
    st.write("Feature Importance visualization coming soon.")
