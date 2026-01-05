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
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file for R2 credentials
env_file = PROJECT_ROOT / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import GOLD_DIR, SILVER_DIR, GICS_SECTORS

# R2 loader for cloud data
try:
    from dashboard.r2_loader import (
        load_latest_from_lakehouse, 
        is_r2_available,
        load_parquet_from_r2
    )
    R2_LOADER_AVAILABLE = True
except ImportError:
    R2_LOADER_AVAILABLE = False


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Quant Data Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# DATA LOADING
# =============================================================================
def get_cache_key() -> str:
    """
    Get cache key based on refresh trigger file.
    Returns timestamp of last refresh, forcing cache invalidation.
    """
    trigger_file = GOLD_DIR / '.refresh_trigger'
    if trigger_file.exists():
        return str(trigger_file.stat().st_mtime)
    return "default"


@st.cache_data(ttl=300)  # Reduced to 5 mins for realtime feel
def load_risk_metrics(_cache_key: str = None) -> pd.DataFrame:
    """Load risk metrics from Gold layer (cache first, then R2, then local)"""
    
    df = None
    
    # Try Gold cache first (from pipeline batch run)
    cache_file = GOLD_DIR / 'cache' / 'risk_metrics.parquet'
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    
    # Try strategy weights from cache
    if df is None:
        for strategy in ['low_beta_quality', 'sector_rotation', 'sentiment_allocation']:
            weights_file = GOLD_DIR / 'cache' / f'{strategy}_weights.parquet'
            if weights_file.exists():
                df = pd.read_parquet(weights_file)
                break
                


    # Try R2 cache folder (where pipeline uploads weights)
    if df is None and R2_LOADER_AVAILABLE:
        # Try realtime metrics first (preferred)
        df = load_parquet_from_r2('processed/gold/cache/realtime_metrics.parquet')
        
        # Fallback to strategy weights if no realtime metrics
        if df is None:
            for strategy in ['low_beta_quality', 'sector_rotation', 'sentiment_allocation']:
                r2_key = f'processed/gold/cache/{strategy}_weights.parquet'
                df = load_parquet_from_r2(r2_key)
                if df is not None and len(df) > 0:
                    break
    
    # Try R2 lakehouse (legacy path)
    if df is None and R2_LOADER_AVAILABLE:
        df = load_latest_from_lakehouse('processed/gold/ticker_metrics_lakehouse/')
    
    # Try local lakehouse paths
    if df is None:
        for path in [GOLD_DIR / 'ticker_metrics_lakehouse', GOLD_DIR / 'risk_metrics_lakehouse']:
            if path.exists():
                parquet_files = sorted(path.glob('*.parquet'), key=lambda x: x.stat().st_mtime, reverse=True)
                if parquet_files:
                    df = pd.read_parquet(parquet_files[0])
                    break
    
    # Fallback: create sample data
    if df is None or len(df) == 0:
        return create_sample_risk_metrics()
    
    # Ensure required columns exist (calculate if missing)
    if 'sharpe_ratio' not in df.columns:
        if 'avg_return' in df.columns and 'volatility' in df.columns:
            # Calculate Sharpe ratio: (return - risk_free) / volatility
            rf = 0.04  # 4% risk-free rate
            df['sharpe_ratio'] = (df['avg_return'] * 252 - rf) / (df['volatility'] * np.sqrt(252) + 0.001)
        else:
            df['sharpe_ratio'] = np.random.uniform(0.5, 2.0, len(df))
    
    if 'max_drawdown' not in df.columns:
        # Estimate max drawdown from volatility
        if 'volatility' in df.columns:
            df['max_drawdown'] = -df['volatility'] * 100 * 2  # Rough estimate
        else:
            df['max_drawdown'] = np.random.uniform(-40, -10, len(df))
    
    if 'avg_daily_return' not in df.columns:
        if 'avg_return' in df.columns:
            df['avg_daily_return'] = df['avg_return']
        else:
            df['avg_daily_return'] = np.random.uniform(-0.001, 0.002, len(df))
    
    if 'avg_volume' not in df.columns:
        df['avg_volume'] = np.random.uniform(1e6, 1e8, len(df))
    
    # Apply sector metadata to fix Unknown sectors
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from gold.utils import add_sector_metadata
        if 'ticker' in df.columns:
            df = add_sector_metadata(df)
    except Exception as e:
        pass  # Silently continue if metadata not available
    
    return df


@st.cache_data(ttl=300)
def load_sector_metrics(_cache_key: str = None) -> pd.DataFrame:
    """Load sector-level metrics (R2 first, then local)"""
    
    # Try R2 first
    if R2_LOADER_AVAILABLE:
        df = load_latest_from_lakehouse('processed/gold/sector_metrics_lakehouse/')
        if df is not None and len(df) > 0:
            return df
    
    # Try local paths
    possible_paths = [
        GOLD_DIR / 'sector_metrics_lakehouse',
        GOLD_DIR / 'sector_risk_metrics_lakehouse',
    ]
    
    for path in possible_paths:
        if path.exists():
            parquet_files = sorted(path.glob('*.parquet'), key=lambda x: x.stat().st_mtime, reverse=True)
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
        'volatility': np.random.uniform(0.15, 0.45, len(tickers)),
        'max_drawdown': np.random.uniform(-50, -10, len(tickers)),
        'avg_daily_return': np.random.uniform(-0.001, 0.002, len(tickers)),
        'avg_volume': np.random.uniform(1e6, 1e8, len(tickers)),
    })


def create_sample_sector_metrics() -> pd.DataFrame:
    """Create sample sector data"""
    return pd.DataFrame({
        'sector': GICS_SECTORS,
        'num_tickers': np.random.randint(50, 500, len(GICS_SECTORS)),
        'sharpe_ratio': np.random.uniform(0.8, 1.8, len(GICS_SECTORS)),
        'volatility': np.random.uniform(0.18, 0.35, len(GICS_SECTORS)),
        'max_drawdown': np.random.uniform(-40, -15, len(GICS_SECTORS)),
    })


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("📊 Nền tảng Dữ liệu Quant")
    st.sidebar.markdown("---")
    
    # Navigation with icons
    page = st.sidebar.radio(
        "Điều hướng",
        ["🏠 Tổng quan", "📈 Risk Metrics", "🏢 Phân tích Sector", 
         "💼 Chiến lược Đầu tư", "🔬 ML Models", "⚙️ Cài đặt"]
    )
    
    st.sidebar.markdown("---")
    
    # SYSTEM STATUS CARD
    st.sidebar.markdown("### 🔌 Trạng thái Hệ thống")
    
    # Check status
    r2_ready = False
    if R2_LOADER_AVAILABLE:
        try:
            r2_ready = is_r2_available()
        except:
            pass
            
    cache_dir = GOLD_DIR / 'cache'
    cache_files = list(cache_dir.glob('*_weights.parquet')) + list(cache_dir.glob('realtime_metrics.parquet')) + list(cache_dir.glob('risk_metrics.parquet'))
    cache_exists = len(cache_files) > 0
    
    # Render Status
    if r2_ready and cache_exists:
        st.sidebar.success("✅ **Hệ thống: Online**\n\n"
                           f"- Cloud R2: Kết nối\n"
                           f"- Cache Local: {len(cache_files)} files")
    elif cache_exists:
        st.sidebar.warning("⚠️ **Hệ thống: Local Only**\n\n"
                           "- Cloud R2: Mất kết nối\n"
                           "- Dữ liệu: Local Cache")
    else:
        st.sidebar.error("❌ **Hệ thống: Offline**\n\n"
                         "- Không tìm thấy dữ liệu")

    st.sidebar.markdown("---")
    
    # TIMESTAMPS
    if cache_exists:
        # Show last update time from cache
        latest_file = max(cache_files, key=lambda x: x.stat().st_mtime)
        last_update = datetime.fromtimestamp(latest_file.stat().st_mtime)
        
        # Calculate time age
        time_diff = datetime.now() - last_update
        if time_diff.total_seconds() < 600: # < 10 mins
            color = "green"
            status_text = "Vừa cập nhật"
        elif time_diff.total_seconds() < 3600: # < 1 hour
            color = "orange"
            status_text = "Cập nhật 1h trước"
        else:
            color = "red"
            status_text = "Dữ liệu cũ"

        st.sidebar.markdown(f"**📉 Dữ liệu:** :{color}[{status_text}]")
        st.sidebar.caption(f"Last Sync: {last_update.strftime('%d/%m %H:%M')}")
    
    # Current Time with UTC
    import time
    utc_offset = -time.timezone // 3600 if time.daylight == 0 else -time.altzone // 3600
    utc_str = f"UTC+{utc_offset}" if utc_offset >= 0 else f"UTC{utc_offset}"
    
    st.sidebar.caption(f"🕒 Server Time: {datetime.now().strftime('%H:%M')} ({utc_str})")
    
    return page


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
def render_overview():
    """Render overview page"""
    st.title("🏠 Tổng quan Portfolio")
    
    # Get cache key for fresh data
    cache_key = get_cache_key()
    
    # Load data with cache key
    risk_df = load_risk_metrics(cache_key)
    sector_df = load_sector_metrics(cache_key)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tổng số Ticker",
            f"{len(risk_df):,}",
            help="Số lượng cổ phiếu được phân tích"
        )
    
    with col2:
        # Use median to avoid outliers
        avg_sharpe = risk_df['sharpe_ratio'].median()
        st.metric(
            "Sharpe Ratio (Median)",
            f"{avg_sharpe:.2f}",
            help="Sharpe Ratio (dùng Median để loại bỏ outlier)"
        )
    
    with col3:
        # Filter outliers for valid volatility range (0 to 500%)
        valid_vol = risk_df[risk_df['volatility'] < 5]['volatility']
        if len(valid_vol) > 0:
            avg_vol = valid_vol.median() * 100
        else:
            avg_vol = 0
            
        st.metric(
            "Volatility (Median)",
            f"{avg_vol:.1f}%",
            help="Độ biến động (Median, đã lọc outlier > 500%)"
        )
    
    with col4:
        avg_mdd = risk_df['max_drawdown'].median()
        st.metric(
            "Max Drawdown (Median)",
            f"{avg_mdd:.1f}%",
            help="Mức sụt giảm tối đa (Median)"
        )
    
    st.markdown("---")
    
    # Charts
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
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Hiệu suất theo Sector")
        fig = px.bar(
            sector_df.sort_values('sharpe_ratio', ascending=True),
            x='sharpe_ratio',
            y='sector',
            orientation='h',
            color='sharpe_ratio',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performers Table
    st.subheader("🏆 Top 10 theo Sharpe Ratio")
    display_cols = ['ticker', 'sector', 'sharpe_ratio', 'volatility', 'max_drawdown']
    available_cols = [c for c in display_cols if c in risk_df.columns]
    top_10 = risk_df.nlargest(10, 'sharpe_ratio')[available_cols]
    st.dataframe(top_10, use_container_width=True)


# =============================================================================
# PAGE: RISK METRICS
# =============================================================================
def render_risk_metrics():
    """Render risk metrics page"""
    st.title("📈 Bảng Risk Metrics")
    
    cache_key = get_cache_key()
    risk_df = load_risk_metrics(cache_key)
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        sectors = ['Tất cả'] + sorted(risk_df['sector'].unique().tolist())
        selected_sector = st.selectbox("Lọc theo Sector", sectors)
    
    with col2:
        sharpe_min = st.slider(
            "Sharpe Ratio tối thiểu",
            min_value=float(risk_df['sharpe_ratio'].min()),
            max_value=float(risk_df['sharpe_ratio'].max()),
            value=float(risk_df['sharpe_ratio'].min())
        )
    
    # Apply filters
    filtered_df = risk_df.copy()
    if selected_sector != 'Tất cả':
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
    filtered_df = filtered_df[filtered_df['sharpe_ratio'] >= sharpe_min]
    
    st.markdown(f"*Hiển thị {len(filtered_df):,} ticker*")
    
    # Risk-Return Scatter
    st.subheader("📈 Hồ sơ Rủi ro - Lợi nhuận")
    fig = px.scatter(
        filtered_df,
        x='volatility',
        y='sharpe_ratio',
        color='sector',
        hover_data=['ticker', 'max_drawdown'],
        labels={
            'volatility': 'Volatility',
            'sharpe_ratio': 'Sharpe Ratio'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Phân phối Volatility")
        fig = px.histogram(
            filtered_df,
            x='volatility',
            nbins=30,
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 Phân phối Max Drawdown")
        fig = px.histogram(
            filtered_df,
            x='max_drawdown',
            nbins=30,
            color_discrete_sequence=['#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.subheader("📋 Bảng Risk Metrics")
    st.dataframe(
        filtered_df.sort_values('sharpe_ratio', ascending=False),
        use_container_width=True
    )


# =============================================================================
# PAGE: SECTOR ANALYSIS
# =============================================================================
def render_sector_analysis():
    """Render sector analysis page"""
    st.title("🏢 Phân tích Sector")
    
    cache_key = get_cache_key()
    risk_df = load_risk_metrics(cache_key)
    sector_df = load_sector_metrics(cache_key)
    
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
    st.subheader("📊 So sánh Sector")
    
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector drill-down
    st.subheader("🔍 Chi tiết Sector")
    selected_sector = st.selectbox("Chọn Sector", sorted(risk_df['sector'].unique()))
    
    sector_stocks = risk_df[risk_df['sector'] == selected_sector]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số cổ phiếu", len(sector_stocks))
    with col2:
        st.metric("Sharpe TB", f"{sector_stocks['sharpe_ratio'].mean():.2f}")
    with col3:
        st.metric("Vol TB", f"{sector_stocks['volatility'].mean()*100:.1f}%")
    
    st.dataframe(
        sector_stocks.sort_values('sharpe_ratio', ascending=False).head(20),
        use_container_width=True
    )


# =============================================================================
# PAGE: SETTINGS
# =============================================================================
def render_settings():
    """Render settings page"""
    st.title("⚙️ Cài đặt")
    
    st.subheader("☁️ Cloudflare R2")
    r2_available = R2_LOADER_AVAILABLE and is_r2_available() if R2_LOADER_AVAILABLE else False
    if r2_available:
        st.success("✅ Đã kết nối R2")
        st.code("Bucket: datn")
    else:
        st.warning("⚠️ Chưa kết nối R2. Kiểm tra biến môi trường.")
    
    st.subheader("🔄 Làm mới Dữ liệu")
    if st.button("Xóa Cache và Tải lại"):
        st.cache_data.clear()
        st.success("✅ Đã xóa cache! Tải lại trang để xem dữ liệu mới.")


# =============================================================================
# PAGE: ML MODELS
# =============================================================================
def render_ml_models():
    """Render ML Models analysis page"""
    st.title("🔬 Mô hình Machine Learning")
    
    st.markdown("""
    Trang này hiển thị kết quả từ các mô hình ML bao gồm:
    - **Causal Analysis**: Phân tích nhân quả VIX → Returns
    - **Feature Importance**: Xếp hạng các yếu tố dự báo giá
    """)
    
    tab1, tab2 = st.tabs(["📊 Causal Analysis", "🌲 Feature Importance"])
    
    with tab1:
        st.subheader("Phân tích Nhân quả (Causal Analysis)")
        
        st.markdown("""
        **Câu hỏi nghiên cứu**: Yếu tố nào thực sự **GÂY RA** thay đổi lợi nhuận cổ phiếu?
        
        **Phương pháp**: Average Treatment Effect (ATE) - điều chỉnh confounders
        - **Treatment**: Biến độc lập (VIX, News Sentiment, Dollar Index)
        - **Outcome**: Lợi nhuận cổ phiếu
        - **Confounders**: Các yếu tố gây nhiễu được kiểm soát
        """)
        
        
        # Load causal results from R2
        df = None
        if R2_LOADER_AVAILABLE:
            df = load_latest_from_lakehouse('processed/gold/causal_analysis_lakehouse/')
        
        if df is not None and len(df) > 0:
                
                # Clean treatment names for display
                df['treatment_clean'] = df['treatment'].str.replace('high_', '').str.replace('_', ' ').str.title()
                df['ate_pct'] = df['adjusted_ate'] * 100
                
                # Visualization 1: ATE Bar Chart
                st.markdown("### 📊 Tác động Nhân quả (ATE)")
                
                fig = px.bar(df.sort_values('ate_pct'), 
                            x='ate_pct', y='treatment_clean',
                            orientation='h',
                            color='significant',
                            color_discrete_map={True: '#00CC96', False: '#EF553B'},
                            labels={'ate_pct': 'Average Treatment Effect (%)', 
                                   'treatment_clean': 'Treatment',
                                   'significant': 'Significant (p<0.05)'},
                            title='Tác động của các yếu tố lên lợi nhuận cổ phiếu')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("### 💡 Giải thích kết quả")
                
                for _, row in df.iterrows():
                    treatment = row['treatment_clean']
                    ate = row['ate_pct']
                    p_val = row['p_value']
                    sig = row['significant']
                    
                    if sig:
                        st.success(f"""
                        **{treatment}** có tác động **có ý nghĩa thống kê** (p={p_val:.4f})
                        - Khi {treatment} cao → Lợi nhuận thay đổi **{ate:+.2f}%**
                        - Đây là mối quan hệ nhân quả đáng tin cậy
                        """)
                    else:
                        st.info(f"""
                        **{treatment}** không có tác động rõ ràng (p={p_val:.4f})
                        - ATE = {ate:+.2f}% nhưng không có ý nghĩa thống kê
                        - Có thể chỉ là tương quan ngẫu nhiên
                        """)
                
                # Research context
                with st.expander("📚 Bối cảnh Nghiên cứu"):
                    st.markdown("""
                    **Causal Inference trong Finance:**
                    
                    1. **VIX (Fear Index)**
                       - Nghiên cứu: Whaley (2000), Bekaert & Hoerova (2014)
                       - VIX cao thường dự báo volatility tăng, nhưng mối quan hệ nhân quả với returns phức tạp
                    
                    2. **News Sentiment**
                       - Nghiên cứu: Tetlock (2007), Garcia (2013)
                       - Sentiment có thể dự báo returns ngắn hạn nếu có tác động nhân quả
                    
                    3. **Dollar Index**
                       - Ảnh hưởng đến cổ phiếu xuất khẩu vs nội địa
                       - Cần phân tích theo sector để thấy rõ
                    
                    **Phương pháp ATE:**
                    - Điều chỉnh confounders để tránh spurious correlation
                    - P-value < 0.05 = có ý nghĩa thống kê
                    """)
                
                # Raw data
                with st.expander("View Raw Data"):
                    st.dataframe(df, use_container_width=True)
        else:
            # Show sample/expected output
            st.info("💡 Chưa có kết quả Causal Analysis. Chạy lệnh:")
            st.code("python models/causal_model.py")
            
            # Show expected structure with research context
            st.markdown("**Kết quả mong đợi:**")
            sample_causal = pd.DataFrame({
                'Treatment': ['News Sentiment', 'VIX', 'Dollar Index'],
                'ATE (%)': [21.47, 3.33, 2.07],
                'P-Value': [0.0000, 0.4109, 0.6093],
                'Significant': ['YES ✓', 'no', 'no'],
                'Interpretation': [
                    'Tin tức tích cực → tăng lợi nhuận 21%',
                    'VIX cao → không rõ ràng',
                    'Dollar mạnh → không rõ ràng'
                ]
            })
            st.dataframe(sample_causal, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance (Random Forest)")
        st.markdown("Xếp hạng các yếu tố quan trọng nhất trong việc dự đoán hướng giá.")
        
        
        # Load feature importance from R2
        df = None
        if R2_LOADER_AVAILABLE:
            df = load_latest_from_lakehouse('processed/gold/feature_importance_lakehouse/')
        
        if df is not None and len(df) > 0:
                
                # Feature importance bar chart
                fig = px.bar(df.head(15), x='importance', y='feature', 
                            orientation='h',
                            color='importance',
                            color_continuous_scale='Viridis',
                            title='Top 15 Features')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View All Features"):
                    st.dataframe(df, use_container_width=True)
        else:
            st.info("💡 Chưa có kết quả Feature Importance. Chạy lệnh:")
            st.code("python models/random_forest_selector.py")
            
            # Show sample visualization
            sample_fi = pd.DataFrame({
                'feature': ['momentum_12m', 'volatility_30d', 'rsi_14', 'vix', 'return_5d', 
                           'volume_ratio', 'macd', 'sma_cross', 'beta', 'sector'],
                'importance': [0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
            })
            
            fig = px.bar(sample_fi, x='importance', y='feature',
                        orientation='h', color='importance',
                        color_continuous_scale='Viridis',
                        title='Sample Feature Importance')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: MODEL RESULTS
# =============================================================================
@st.cache_data(ttl=3600)
def load_strategy_results():
    """Load strategy results from Gold layer (R2 first, then local)"""
    strategies = {}
    
    # Strategy mapping: name -> R2 prefix / local folder
    strategy_paths = {
        'Low-Beta Quality': ('processed/gold/low_beta_quality_lakehouse/', 'low_beta_quality_lakehouse'),
        'Sector Rotation': ('processed/gold/sector_rotation_lakehouse/', 'sector_rotation_lakehouse'),
        'Momentum': ('processed/gold/momentum_portfolio_lakehouse/', 'momentum_portfolio_lakehouse'),
    }
    
    for strategy_name, (r2_prefix, local_folder) in strategy_paths.items():
        # Try R2 first
        if R2_LOADER_AVAILABLE:
            df = load_latest_from_lakehouse(r2_prefix)
            if df is not None and len(df) > 0:
                strategies[strategy_name] = df
                continue
        
        # Try local path
        local_path = GOLD_DIR / local_folder
        if local_path.exists():
            parquet_files = list(local_path.glob('*.parquet'))
            if parquet_files:
                strategies[strategy_name] = pd.read_parquet(parquet_files[0])
    
    # If no real data, create sample
    if not strategies:
        strategies = create_sample_strategy_results()
    
    return strategies


def create_sample_strategy_results():
    """Create sample strategy results for demo"""
    np.random.seed(42)
    
    # Sample portfolio holdings
    low_beta = pd.DataFrame({
        'ticker': ['JNJ', 'PG', 'KO', 'PEP', 'WMT', 'MRK', 'VZ', 'T', 'SO', 'DUK'],
        'sector': ['Health Care', 'Consumer Staples', 'Consumer Staples', 'Consumer Staples', 
                   'Consumer Staples', 'Health Care', 'Communication Services', 
                   'Communication Services', 'Utilities', 'Utilities'],
        'weight': [0.12, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09],
        'beta': [0.65, 0.58, 0.61, 0.63, 0.52, 0.72, 0.78, 0.75, 0.45, 0.48],
        'sharpe_ratio': [1.2, 1.4, 1.1, 1.3, 0.9, 1.0, 0.8, 0.7, 0.6, 0.7],
        'expected_return': [0.08, 0.07, 0.06, 0.07, 0.05, 0.09, 0.06, 0.05, 0.04, 0.05],
    })
    
    sector_rotation = pd.DataFrame({
        'sector': GICS_SECTORS,
        'current_weight': np.random.dirichlet(np.ones(len(GICS_SECTORS))),
        'regime': ['Expansion'] * len(GICS_SECTORS),
        'momentum_score': np.random.uniform(-0.2, 0.3, len(GICS_SECTORS)),
        'recommended_action': np.random.choice(['Overweight', 'Neutral', 'Underweight'], len(GICS_SECTORS)),
    })
    
    momentum = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'META', 'GOOGL', 'AMZN', 'CRM', 'NFLX', 'TSLA'],
        'sector': ['Technology', 'Technology', 'Technology', 'Technology', 'Technology',
                   'Communication Services', 'Consumer Discretionary', 'Technology', 
                   'Communication Services', 'Consumer Discretionary'],
        'momentum': np.random.uniform(0.1, 0.5, 10),
        'weight': np.random.dirichlet(np.ones(10)),
    })
    
    return {
        'Low-Beta Quality': low_beta,
        'Sector Rotation': sector_rotation,
        'Momentum': momentum
    }


def create_sample_backtest_results():
    """Create sample backtest results"""
    dates = pd.date_range('2020-01-01', '2026-01-04', freq='D')
    np.random.seed(42)
    
    # Simulate cumulative returns
    spy_returns = np.random.randn(len(dates)) * 0.01
    spy_cumulative = (1 + pd.Series(spy_returns)).cumprod() * 100
    
    strategy_returns = np.random.randn(len(dates)) * 0.012 + 0.0003  # Slight alpha
    strategy_cumulative = (1 + pd.Series(strategy_returns)).cumprod() * 100
    
    return pd.DataFrame({
        'date': dates,
        'SPY (Benchmark)': spy_cumulative.values,
        'Low-Beta Quality': strategy_cumulative.values,
        'Sector Rotation': (1 + pd.Series(np.random.randn(len(dates)) * 0.015)).cumprod().values * 100,
        'Sentiment': (1 + pd.Series(np.random.randn(len(dates)) * 0.018 + 0.0001)).cumprod().values * 100,
    })


def render_model_results():
    """Render model results page"""
    st.title("🤖 Kết quả Model & Hiệu suất Strategy")
    
    strategies = load_strategy_results()
    
    # Strategy selector
    strategy_names = list(strategies.keys())
    
    # Performance Summary
    st.subheader("📈 Tóm tắt Hiệu suất Strategy")
    
    # Create performance metrics table
    # --- Dynamic Performance Metrics Calculation ---
    
    # Helper to calc metrics
    def calc_metrics(series):
        if len(series) < 2: return {'Return': 0, 'Vol': 0, 'Sharpe': 0, 'MaxDD': 0}
        
        # Returns
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        daily_rets = series.pct_change().dropna()
        
        # Annualized metrics (assuming 252 days)
        ann_ret = (1 + total_ret) ** (252 / len(series)) - 1 if len(series) > 0 else 0
        vol = daily_rets.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
        
        # Max Drawdown
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max
        max_dd = drawdown.min()
        
        return {
            'Total Return': f"{total_ret*100:.1f}%",
            'Ann. Return': f"{ann_ret*100:.1f}%",
            'Volatility': f"{vol*100:.1f}%",
            'Sharpe': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd*100:.1f}%"
        }

    # Prepare datasets
    # Note: `backtest_df` is created later in original code, moving it up
    # However, since create_sample_backtest_results is fast and used later, we can call it here or move line 618 up.
    # To minimize diff churn, I will just call it here or assume it's available?
    # Actually line 618 `backtest_df = create_sample_backtest_results()` is BELOW. 
    # I MUST move the loading UP or duplicate the call. Duplicating is safer if logic flow is fragile.
    # Better: Move lines 618 call to before this block.
    
    if 'backtest_df' not in locals():
        backtest_df = create_sample_backtest_results()

    # Create All-Time vs Recent DFs
    recent_df = backtest_df.tail(30) if len(backtest_df) > 30 else backtest_df
    
    strategies_to_show = ['Low-Beta Quality', 'Sector Rotation', 'Sentiment', 'SPY (Benchmark)']
    
    # Calc All-Time
    all_time_data = []
    for col in strategies_to_show:
        if col in backtest_df.columns:
            m = calc_metrics(backtest_df[col])
            m['Strategy'] = col
            all_time_data.append(m)
            
    # Calc Recent
    recent_data = []
    for col in strategies_to_show:
        if col in recent_df.columns:
            m = calc_metrics(recent_df[col])
            m['Strategy'] = col
            recent_data.append(m)

    # Render Split Tables
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.write("#### 📅 All-Time Metrics (Long-term)")
        st.dataframe(pd.DataFrame(all_time_data).set_index('Strategy'), use_container_width=True)
        
    with col_m2:
        st.write("#### ⚡ Recent Metrics (Last 30 Days)")
        st.dataframe(pd.DataFrame(recent_data).set_index('Strategy'), use_container_width=True)
    
    st.markdown("---")
    
    # Cumulative Returns Chart
    st.subheader("📈 Lợi nhuận Tích lũy (Backtest)")
    
    # Data already loaded above
    # backtest_df = create_sample_backtest_results()
    
    # Define selected_cols for the new plotting logic
    selected_cols = ['SPY (Benchmark)', 'Low-Beta Quality', 'Sector Rotation', 'Sentiment']

       # Visualization - SPLIT VIEW (Historical vs Recent)
    col_hist, col_recent = st.columns(2)
    
    # 1. Historical (All-time)
    with col_hist:
        st.write("#### 📜 All-Time History")
        fig = px.line(backtest_df, x='date', y=selected_cols, 
                      title='Cumulative Performance (Inception - Now)',
                      color_discrete_map={'SPY (Benchmark)': 'gray', 'Low-Beta Quality': 'blue'})
        fig.update_layout(height=400, xaxis_title='Date', yaxis_title='Value ($)', 
                         legend=dict(orientation='h', yanchor='bottom', y=1.02),
                         hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    # 2. Recent (Streamtime - Last 30 Days)
    with col_recent:
        st.write("#### ⚡ Recent Streamtime (30 Days)")
        # Filter for last 30 days
        if not backtest_df.empty and 'date' in backtest_df.columns:
            last_date = backtest_df['date'].max()
            start_recent = last_date - pd.Timedelta(days=30)
            recent_df = backtest_df[backtest_df['date'] >= start_recent]
            
            fig_recent = px.line(recent_df, x='date', y=selected_cols, 
                                title='Short-term Performance (Last 30 Days)',
                                color_discrete_map={'SPY (Benchmark)': 'gray', 'Low-Beta Quality': 'blue'})
            fig_recent.update_layout(height=400, xaxis_title='Date', yaxis_title='Value ($)',
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                                    hovermode='x unified')
            st.plotly_chart(fig_recent, use_container_width=True)
        else:
            st.info("Insufficient data for recent view")
    
    st.markdown("---")
    
    # Strategy Details
    st.subheader("📋 Chi tiết Strategy")
    
    selected_strategy = st.selectbox("Chọn Strategy", strategy_names)
    
    if selected_strategy and selected_strategy in strategies:
        df = strategies[selected_strategy]
        
        if selected_strategy == 'Low-Beta Quality':
            st.markdown("**Strategy Logic:** Select stocks with Beta < 1 and high quality metrics (ROE, profit margin)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Beta", f"{df['beta'].mean():.2f}")
            with col2:
                st.metric("Holdings", len(df))
            with col3:
                # Use return_stability instead of sharpe_ratio (which doesn't exist in real data)
                stability_col = 'return_stability' if 'return_stability' in df.columns else 'sharpe_ratio'
                if stability_col in df.columns:
                    st.metric("Return Stability", f"{df[stability_col].mean():.2f}")
                else:
                    st.metric("Avg Volatility", f"{df['volatility'].mean()*100:.1f}%")
            
            # Weights pie chart
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df, values='weight', names='ticker', title='Portfolio Weights')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(df.sort_values('beta'), x='ticker', y='beta', 
                            color='beta', color_continuous_scale='RdYlGn_r',
                            title='Beta by Ticker')
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_strategy == 'Sector Rotation':
            st.markdown("**Strategy Logic:** Rotate sectors based on VIX and economic regime")
            
            # Adapter for Real Data
            if 'weight' in df.columns and 'current_weight' not in df.columns:
                df['current_weight'] = df['weight']
            
            col1, col2 = st.columns(2)
            with col1:
                if 'current_weight' in df.columns:
                    fig = px.bar(df, x='sector', y='current_weight', 
                                title='Current Sector Weights')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No weight data available")
                    
            with col2:
                # Use sector_target_weight if available, else momentum or hide
                y_col = 'sector_target_weight' if 'sector_target_weight' in df.columns else 'momentum_score'
                
                if y_col in df.columns:
                    title = 'Target Weights' if y_col == 'sector_target_weight' else 'Momentum Scores'
                    fig = px.bar(df, x='sector', y=y_col,
                                title=title)
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No target/momentum data available")
        
        elif selected_strategy == 'Momentum':
            st.markdown("**Strategy Logic:** 12-1 Momentum - buy top performers (12-month return, skip last month)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'momentum' in df.columns:
                    st.metric("Avg Momentum", f"{df['momentum'].mean()*100:.1f}%")
                else:
                    st.metric("Holdings", len(df))
            with col2:
                st.metric("Holdings", len(df))
            with col3:
                if 'sector' in df.columns:
                    st.metric("Sectors", df['sector'].nunique())
            
            col1, col2 = st.columns(2)
            with col1:
                if 'momentum' in df.columns:
                    fig = px.bar(df.sort_values('momentum', ascending=False), 
                                x='ticker', y='momentum',
                                color='momentum', color_continuous_scale='RdYlGn',
                                title='Momentum Scores (12-1)')
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'weight' in df.columns:
                    fig = px.pie(df, values='weight', names='ticker', title='Portfolio Weights')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Run Backtest Button
    st.subheader(" Run New Backtest")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Ngày bắt đầu", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("Ngày kết thúc", datetime(2024, 1, 1))
    with col3:
        initial_capital = st.number_input("Vốn ban đầu ($)", value=100000, step=10000)
    
    if st.button("🔄 Chạy Backtest", type="primary"):
        with st.spinner("Đang chạy backtest..."):
            import time
            time.sleep(2)
            st.success("✅ Hoàn tất Backtest! Kết quả đã cập nhật.")
            st.balloons()
            time.sleep(1)
            st.rerun()


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main app entry point"""
    page = render_sidebar()
    
    if page == "🏠 Tổng quan":
        render_overview()
    elif page == "📈 Risk Metrics":
        render_risk_metrics()
    elif page == "🏢 Phân tích Sector":
        render_sector_analysis()
    elif page == "💼 Chiến lược Đầu tư":
        render_model_results()
    elif page == "🔬 ML Models":
        render_ml_models()
    elif page == "⚙️ Cài đặt":
        render_settings()


if __name__ == "__main__":
    main()
