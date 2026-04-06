# Models Directory

Thư mục chứa các model AI sau khi download.

## Cấu trúc

```
models/
├── asr/           # Qwen3-ASR models
│   └── Qwen3-ASR-0.6B/
├── tts/           # Gwen-TTS models
│   └── gwen-tts-0.6B/
└── vad/           # Silero VAD
    └── silero_vad.jit
```

## Download Models

### Tự động (khi chạy lần đầu)

Models sẽ tự động download khi khởi động service.

### Thủ công (offline environment)

```bash
# ASR - Qwen3-ASR
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./models/asr/Qwen3-ASR-0.6B

# TTS - Gwen-TTS
huggingface-cli download g-group-ai-lab/gwen-tts-0.6B --local-dir ./models/tts/gwen-tts-0.6B

# VAD - Silero (tự động từ torch.hub)
```

## Environment Variables

```bash
# Sử dụng local models
ASR_MODEL=./models/asr/Qwen3-ASR-0.6B
TTS_MODEL=./models/tts/gwen-tts-0.6B

# Hoặc từ HuggingFace Hub (mặc định)
ASR_MODEL=Qwen/Qwen3-ASR-0.6B
TTS_MODEL=g-group-ai-lab/gwen-tts-0.6B
```

## Disk Space

| Model | Size |
|-------|------|
| Qwen3-ASR-0.6B | ~1.2 GB |
| Gwen-TTS-0.6B | ~1.5 GB |
| Silero VAD | ~2 MB |
| **Total** | ~2.7 GB |
