# 🚀 Hướng dẫn Deploy Streamlit Cloud với R2

Để chạy app trên Streamlit Cloud và kết nối với dữ liệu R2, bạn cần làm các bước sau:

## 1. Cập nhật Code lên GitHub

Đảm bảo bạn đã push code mới nhất (bao gồm logic R2 sync) lên repository GitHub của bạn.

```bash
git add .
git commit -m "Update dashboard with R2 sync"
git push origin main
```

## 2. Cấu hình Secrets trên Streamlit Cloud

Streamlit Cloud không đọc file `.env` vì lý do bảo mật. Bạn cần nhập key vào phần Secrets.

1. Truy cập [share.streamlit.io](https://share.streamlit.io/)
2. Tìm app của bạn (VD: `dubaochungkhoan`)
3. Bấm dấu `⋮` (Settings) -> **Settings**
4. Chọn tab **Secrets**
5. Dán nội dung sau vào ô soạn thảo:

```toml
R2_ENDPOINT = "https://2e1dfe3165f058a398ee7cac430e8301.r2.cloudflarestorage.com"
R2_ACCESS_KEY = "d551ffe8ce25a9803db48c6624009f54"
R2_SECRET_KEY = "d492ee78cb6e23cc42942aa79ba7816879f387f9cf777d3e67f0366a2be2fd2a"
R2_BUCKET = "datn"
```

1. Bấm **Save**.

## 3. Wake Up / Reboot App

- Nếu app đang ngủ (Zzzz), bấm **"Yes, get this app back up!"**.
- Nếu app đang chạy lỗi, bấm **Reboot** trong menu góc phải trên cùng.

## 4. Kiểm tra

- Mở App trên trình duyệt.
- Vào menu **Settings** bên trái.
- Nếu thấy **"✅ Đã kết nối R2"** nghĩa là thành công!

---
**Lưu ý:**
Do Streamlit Cloud là server nhỏ (Community Cloud), việc xử lý data quá lớn có thể bị giới hạn RAM. Pipeline hiện tại đã tối ưu (Streaming Mode) nên sẽ chạy ổn.
