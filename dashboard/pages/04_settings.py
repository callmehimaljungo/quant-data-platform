
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.r2_loader import is_r2_available
from dashboard.components.sidebar import render_sidebar

render_sidebar()

st.title("⚙️ Cài đặt")

st.subheader("☁️ Cloudflare R2")
try:
    r2_ready = is_r2_available()
except:
    r2_ready = False

if r2_ready:
    st.success("✅ Đã kết nối R2")
    st.code("Bucket: datn")
else:
    st.warning("⚠️ Chưa kết nối R2. Kiểm tra biến môi trường.")

st.subheader("🔄 Làm mới Dữ liệu")
if st.button("Xóa Cache và Tải lại"):
    st.cache_data.clear()
    st.success("✅ Đã xóa cache! Tải lại trang để xem dữ liệu mới.")
