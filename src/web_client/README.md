# Web Test Client

Web-based test client for Voice Agent.

## Usage

```bash
# Start Voice Agent server first
uv run python -m voice_agent.main

# Then start test client (in another terminal)
uv run python -m web_client.server --port 8080
```

Open <http://localhost:8080> in your browser.

## Features

- 🎤 Real-time audio capture from microphone
- 📊 Audio visualizer
- 💬 Live transcript display
- 📈 Latency metrics
- 🔊 Audio playback of bot responses

## Browser Requirements

- Modern browser (Chrome, Firefox, Edge)
- Microphone permission
- WebSocket support
- Web Audio API support

## Notes

- Uses ScriptProcessor for audio capture (AudioWorklet requires HTTPS)
- Audio is captured at 16kHz mono
- PCM S16LE format for both input and output
