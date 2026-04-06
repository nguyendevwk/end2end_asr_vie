# Vietnamese Voice Agent Documentation

## 📚 Tài liệu hướng dẫn

Hệ thống Voice Agent thế hệ mới cho tiếng Việt với kiến trúc tối ưu real-time.

### Tài liệu chính

1. **[Kiến trúc hệ thống](ARCHITECTURE.md)** - Thiết kế tổng quan và luồng xử lý
2. **[Hướng dẫn cài đặt](INSTALLATION.md)** - Cài đặt và cấu hình môi trường
3. **[Hướng dẫn sử dụng](USAGE.md)** - Chạy và test hệ thống
4. **[Tối ưu hóa hiệu năng](OPTIMIZATION.md)** - Các kỹ thuật tối ưu
5. **[API Reference](API.md)** - Chi tiết API endpoints và WebSocket

### Tài liệu kỹ thuật

- **[qwenasr/README.md](qwenasr/README.md)** - Hướng dẫn Qwen3-ASR model
- **[Troubleshooting](TROUBLESHOOTING.md)** - Xử lý lỗi thường gặp

## 🎯 Mục tiêu dự án

Voice Agent demo cho phỏng vấn xin việc:

- ✅ **Clean code** - Dễ đọc, dễ mở rộng
- ✅ **Real-time** - Latency < 2s end-to-end
- ✅ **Lightweight** - Chạy trên GPU 4GB
- ✅ **Production-ready** - Monitoring, logging, error handling đầy đủ

## 🚀 Quick Start

```bash
# Clone và cài đặt
cd src
uv sync

# Cấu hình
cp .env.example .env
# Chỉnh GROQ_API_KEY trong .env

# Chạy server
uv run voice-agent

# Chạy web client (terminal khác)
uv run voice-agent-client
```

## 📖 Đọc thêm

- Xem [../src/README.md](../src/README.md) cho quick reference
- Xem [ARCHITECTURE.md](ARCHITECTURE.md) để hiểu kiến trúc
- Xem [OPTIMIZATION.md](OPTIMIZATION.md) để tối ưu hiệu năng
