"""
Standalone Web Test Client for Voice Agent.

A simple Flask-based web server with HTML/JS frontend for testing the voice agent.

Usage:
    python -m web_client.server --port 8080

Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, render_template_string, send_from_directory

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent Test Client</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2rem;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 16px;
            color: #00d4ff;
        }
        
        .config-group {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        
        .config-item {
            flex: 1;
            min-width: 200px;
        }
        
        .config-item label {
            display: block;
            margin-bottom: 6px;
            font-size: 0.85rem;
            color: #aaa;
        }
        
        .config-item input {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
            font-size: 0.95rem;
        }
        
        .config-item input:focus {
            outline: none;
            border-color: #00d4ff;
        }
        
        .btn {
            padding: 14px 32px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            color: #fff;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 212, 255, 0.3);
        }
        
        .btn-danger {
            background: #e74c3c;
            color: #fff;
        }
        
        .btn-danger:hover {
            background: #c0392b;
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .controls {
            display: flex;
            gap: 16px;
            justify-content: center;
            margin: 24px 0;
        }
        
        .status {
            text-align: center;
            padding: 16px;
            border-radius: 12px;
            font-size: 1.1rem;
            margin-bottom: 20px;
        }
        
        .status.idle {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .status.connecting {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }
        
        .status.connected {
            background: rgba(40, 167, 69, 0.2);
            color: #28a745;
        }
        
        .status.listening {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            animation: pulse 1.5s infinite;
        }
        
        .status.speaking {
            background: rgba(123, 47, 247, 0.2);
            color: #b57aff;
        }
        
        .status.error {
            background: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .visualizer {
            height: 80px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        
        .visualizer canvas {
            width: 100%;
            height: 100%;
        }
        
        .transcript {
            min-height: 120px;
            max-height: 300px;
            overflow-y: auto;
            padding: 16px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .transcript .user {
            color: #00d4ff;
            margin-bottom: 8px;
        }
        
        .transcript .bot {
            color: #b57aff;
            margin-bottom: 8px;
        }
        
        .transcript .system {
            color: #888;
            font-size: 0.85rem;
            font-style: italic;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
        }
        
        .metric {
            text-align: center;
            padding: 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: #888;
            margin-top: 4px;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice Agent Test Client</h1>
        
        <!-- Config -->
        <div class="card">
            <h2>⚙️ Configuration</h2>
            <div class="config-group">
                <div class="config-item">
                    <label>WebSocket URL</label>
                    <input type="text" id="wsUrl" value="ws://localhost:8000/ws/agent">
                </div>
                <div class="config-item">
                    <label>Sample Rate</label>
                    <input type="number" id="sampleRate" value="16000" disabled>
                </div>
            </div>
        </div>
        
        <!-- Status -->
        <div class="status idle" id="status">
            🔌 Disconnected
        </div>
        
        <!-- Visualizer -->
        <div class="visualizer">
            <canvas id="visualizer"></canvas>
        </div>
        
        <!-- Controls -->
        <div class="controls">
            <button class="btn btn-primary" id="connectBtn" onclick="toggleConnection()">
                🔗 Connect
            </button>
            <button class="btn btn-danger" id="stopBtn" onclick="stop()" disabled>
                ⏹️ Stop
            </button>
        </div>
        
        <!-- Transcript -->
        <div class="card">
            <h2>💬 Conversation</h2>
            <div class="transcript" id="transcript">
                <div class="system">Click "Connect" to start...</div>
            </div>
        </div>
        
        <!-- Metrics -->
        <div class="card">
            <h2>📊 Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="metricVad">-</div>
                    <div class="metric-label">VAD (ms)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metricAsr">-</div>
                    <div class="metric-label">ASR (ms)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metricTts">-</div>
                    <div class="metric-label">TTS (ms)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metricTtfa">-</div>
                    <div class="metric-label">TTFA (ms)</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Voice Agent Test Client v1.0 | WebSocket + Web Audio API
        </div>
    </div>

    <script>
        // === State ===
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let processor = null;
        let isConnected = false;
        let isRecording = false;
        
        // Audio playback
        let audioQueue = [];
        let isPlaying = false;
        
        // Visualizer
        let analyser = null;
        let visualizerCtx = null;
        let animationId = null;
        
        // === DOM Elements ===
        const statusEl = document.getElementById('status');
        const transcriptEl = document.getElementById('transcript');
        const connectBtn = document.getElementById('connectBtn');
        const stopBtn = document.getElementById('stopBtn');
        const wsUrlInput = document.getElementById('wsUrl');
        const canvas = document.getElementById('visualizer');
        
        // === Initialize ===
        function init() {
            visualizerCtx = canvas.getContext('2d');
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
        }
        
        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }
        
        // === Connection ===
        async function toggleConnection() {
            if (isConnected) {
                disconnect();
            } else {
                await connect();
            }
        }
        
        async function connect() {
            try {
                setStatus('connecting', '🔄 Connecting...');
                
                // Initialize audio context
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });
                
                // Get microphone
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });
                
                // Setup analyser for visualization
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                const source = audioContext.createMediaStreamSource(mediaStream);
                source.connect(analyser);
                
                // Setup audio processor
                await setupAudioProcessor(source);
                
                // Connect WebSocket
                const wsUrl = wsUrlInput.value;
                ws = new WebSocket(wsUrl);
                ws.binaryType = 'arraybuffer';
                
                ws.onopen = () => {
                    isConnected = true;
                    isRecording = true;
                    setStatus('connected', '✅ Connected - Speak now!');
                    connectBtn.textContent = '🔌 Disconnect';
                    stopBtn.disabled = false;
                    startVisualization();
                    addTranscript('system', 'Connected to voice agent');
                };
                
                ws.onmessage = (event) => {
                    if (event.data instanceof ArrayBuffer) {
                        // Audio response
                        playAudio(event.data);
                    } else {
                        // Text event
                        handleEvent(event.data);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    setStatus('error', '❌ Connection error');
                };
                
                ws.onclose = () => {
                    disconnect();
                };
                
            } catch (error) {
                console.error('Connection failed:', error);
                setStatus('error', '❌ ' + error.message);
            }
        }
        
        function disconnect() {
            isConnected = false;
            isRecording = false;
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            stopVisualization();
            
            setStatus('idle', '🔌 Disconnected');
            connectBtn.textContent = '🔗 Connect';
            stopBtn.disabled = true;
            addTranscript('system', 'Disconnected');
        }
        
        function stop() {
            disconnect();
        }
        
        // === Audio Processing ===
        async function setupAudioProcessor(source) {
            // Use ScriptProcessor for compatibility
            // (AudioWorklet would be better but requires HTTPS)
            // Buffer size must be power of 2: 256, 512, 1024, 2048, 4096, 8192, 16384
            const bufferSize = 2048; // ~128ms at 16kHz (closest valid size to 100ms)
            processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            processor.onaudioprocess = (e) => {
                if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) {
                    return;
                }
                
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Convert float32 to int16 PCM
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                // Send to server
                ws.send(pcmData.buffer);
            };
            
            source.connect(processor);
            processor.connect(audioContext.destination);
        }
        
        // === Audio Playback ===
        function playAudio(arrayBuffer) {
            audioQueue.push(arrayBuffer);
            if (!isPlaying) {
                playNextAudio();
            }
        }
        
        async function playNextAudio() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }
            
            isPlaying = true;
            const arrayBuffer = audioQueue.shift();
            
            try {
                // Convert PCM S16LE to Float32
                const int16Data = new Int16Array(arrayBuffer);
                const float32Data = new Float32Array(int16Data.length);
                for (let i = 0; i < int16Data.length; i++) {
                    float32Data[i] = int16Data[i] / 32768.0;
                }
                
                // Create audio buffer
                const audioBuffer = audioContext.createBuffer(1, float32Data.length, 16000);
                audioBuffer.getChannelData(0).set(float32Data);
                
                // Play
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.onended = playNextAudio;
                source.start();
                
            } catch (error) {
                console.error('Playback error:', error);
                playNextAudio();
            }
        }
        
        // === Event Handling ===
        function handleEvent(data) {
            console.log('Event:', data);
            
            if (data.startsWith('TRANSCRIPT:')) {
                const text = data.substring(11);
                addTranscript('user', text);
            } else if (data === 'LISTENING') {
                setStatus('listening', '🎤 Listening...');
            } else if (data === 'PROCESSING') {
                setStatus('connected', '⏳ Processing...');
            } else if (data === 'SPEAKING') {
                setStatus('speaking', '🔊 Speaking...');
            } else if (data === 'IDLE') {
                setStatus('connected', '✅ Ready - Speak now!');
            } else if (data.startsWith('ERROR:')) {
                const error = data.substring(6);
                addTranscript('system', 'Error: ' + error);
                setStatus('error', '❌ ' + error);
            }
        }
        
        // === UI Helpers ===
        function setStatus(state, text) {
            statusEl.className = 'status ' + state;
            statusEl.textContent = text;
        }
        
        function addTranscript(type, text) {
            const div = document.createElement('div');
            div.className = type;
            
            if (type === 'user') {
                div.innerHTML = '👤 <strong>You:</strong> ' + text;
            } else if (type === 'bot') {
                div.innerHTML = '🤖 <strong>Bot:</strong> ' + text;
            } else {
                div.textContent = text;
            }
            
            transcriptEl.appendChild(div);
            transcriptEl.scrollTop = transcriptEl.scrollHeight;
        }
        
        // === Visualization ===
        function startVisualization() {
            if (!analyser) return;
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            function draw() {
                animationId = requestAnimationFrame(draw);
                
                analyser.getByteFrequencyData(dataArray);
                
                visualizerCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                visualizerCtx.fillRect(0, 0, canvas.width, canvas.height);
                
                const barWidth = (canvas.width / bufferLength) * 2.5;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const barHeight = (dataArray[i] / 255) * canvas.height;
                    
                    const gradient = visualizerCtx.createLinearGradient(0, canvas.height, 0, 0);
                    gradient.addColorStop(0, '#00d4ff');
                    gradient.addColorStop(1, '#7b2ff7');
                    
                    visualizerCtx.fillStyle = gradient;
                    visualizerCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                    
                    x += barWidth + 1;
                }
            }
            
            draw();
        }
        
        function stopVisualization() {
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
            
            if (visualizerCtx) {
                visualizerCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                visualizerCtx.fillRect(0, 0, canvas.width, canvas.height);
            }
        }
        
        // === Init ===
        init();
    </script>
</body>
</html>
"""

# Create Flask app
app = Flask(__name__)


@app.route("/")
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/static/<path:filename>")
def static_files(filename: str):
    """Serve static files."""
    static_dir = Path(__file__).parent / "static"
    return send_from_directory(static_dir, filename)


def main():
    """Run the test client server."""
    parser = argparse.ArgumentParser(description="Voice Agent Test Client")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          🎙️  Voice Agent Test Client                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Open in browser: http://localhost:{args.port}                    ║
║                                                              ║
║  Make sure Voice Agent is running at ws://localhost:8000     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
