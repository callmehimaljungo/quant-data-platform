
import streamlit as st
import datetime
import time
from pathlib import Path
from dashboard.utils.data_loader import load_risk_metrics, get_cache_key
from dashboard.r2_loader import is_r2_available, get_r2_object_last_modified
from utils.quant_validators import FinancialDataValidator
from config import GOLD_DIR

def render_sidebar():
    """
    Render common sidebar elements (System Status, Health Check).
    Should be called at the start of each page.
    """
    st.sidebar.title("📊 Quant Platform")
    
    # SYSTEM STATUS CARD
    st.sidebar.markdown("### 🔌 Trạng thái Hệ thống")
    
    # Check R2
    r2_ready = False
    try:
        r2_ready = is_r2_available()
    except:
        pass
        
    cache_dir = GOLD_DIR / 'cache'
    # Simplified existence check
    cache_exists = (cache_dir / 'realtime_metrics.parquet').exists() or (cache_dir / 'risk_metrics.parquet').exists()
    
    last_update = None
    
    # Render Status
    if r2_ready:
        st.sidebar.success("✅ **Hệ thống: Online**\n\nCloud R2: Kết nối")
        # Try to get last update from R2 - Check risk_metrics.parquet (Main Data)
        r2_time = get_r2_object_last_modified('processed/gold/cache/risk_metrics.parquet')
        
        # Fallback to realtime if risk_metrics not found
        if not r2_time:
            r2_time = get_r2_object_last_modified('processed/gold/cache/realtime_metrics.parquet')
            
        if r2_time:
            last_update = r2_time
            
    elif cache_exists:
        st.sidebar.warning("⚠️ **Hệ thống: Local Only**\n\nLocal Cache")
        # Get from local
        if (cache_dir / 'realtime_metrics.parquet').exists():
            f = (cache_dir / 'realtime_metrics.parquet')
        elif (cache_dir / 'risk_metrics.parquet').exists():
            f = (cache_dir / 'risk_metrics.parquet')
        else:
            f = None
            
        if f:
            last_update = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        
    else:
        st.sidebar.error("❌ **Hệ thống: Offline**")

    st.sidebar.markdown("---")
    
    # TIMESTAMPS DISPLAY
    if last_update:
        # Calculate time age
        # Note: timezone handling is tricky, simplistic approach here
        if last_update.tzinfo:
            now = datetime.datetime.now(last_update.tzinfo)
            display_time = last_update.strftime('%d/%m %H:%M UTC')
        else:
            now = datetime.datetime.now()
            display_time = last_update.strftime('%d/%m %H:%M')
            
        time_diff = now - last_update
        seconds = time_diff.total_seconds()
        
        if seconds < 1800: # < 30 mins
            color = "green"
            status_text = "Vừa cập nhật"
        elif seconds < 7200: # < 2 hours
            color = "orange"
            status_text = "Cập nhật gần đây"
        else:
            color = "red"
            status_text = "Dữ liệu cũ"

        st.sidebar.markdown(f"**📉 Dữ liệu:** :{color}[{status_text}]")
        st.sidebar.caption(f"Last Sync: {display_time}")
    
    # Current Time
    utc_offset = -time.timezone // 3600 if time.daylight == 0 else -time.altzone // 3600
    utc_str = f"UTC+{utc_offset}" if utc_offset >= 0 else f"UTC{utc_offset}"
    st.sidebar.caption(f"🕒 Server Time: {datetime.datetime.now().strftime('%H:%M')} ({utc_str})")
    
    # --- HEALTH CHECK ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Data Health")
    
    # Load data for validation (cached)
    risk_df = load_risk_metrics(get_cache_key())
    
    if risk_df is not None:
        validator = FinancialDataValidator()
        val_res = validator.validate(risk_df)
        
        if val_res['is_valid']:
            st.sidebar.success(f"✅ Hợp lệ ({len(risk_df)} mã)")
        else:
            err_count = val_res['summary']['error_count']
            warn_count = val_res['summary']['warning_count']
            
            if err_count > 0:
                st.sidebar.error(f"❌ {err_count} Lỗi nghiêm trọng")
            elif warn_count > 0:
                st.sidebar.warning(f"⚠️ {warn_count} Cảnh báo")
                
            with st.sidebar.expander("Chi tiết vấn đề"):
                for err in val_res['errors']:
                    st.write(f"🔴 **{err.column}**: {err.message}")
                for warn in val_res['warnings']:
                    st.write(f"🟡 **{warn.column}**: {warn.message}")
