# PrivateWebSpeech

A standalone private speech interface for web applications that provides speech-to-text (STT) and text-to-speech (TTS) capabilities through a simple API.

## Overview

PrivateWebSpeech serves as a self-hosted bridge between web applications and speech processing models, allowing you to add speech functionality to your web applications without relying on third-party services.

### Features

- Real-time speech-to-text transcription using Whisper
- High-quality text-to-speech synthesis using Kokoro
- Voice activity detection
- Streaming audio processing
- WebSocket-based API for web integration
- Simple web demo interface included

## Demo

The application currently runs as a demo web interface where you can:
- Record audio and see real-time transcription
- Play back recordings
- Convert text to speech

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Inventor2525/PrivateWebSpeech.git
   cd PrivateWebSpeech
   ```

2. Install the required packages:
   ```
   pip install -e .
   ```

3. Install FFmpeg (required for audio processing):
   ```
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## Usage

Run the server:
```
python app.py
```

Then open a web browser and navigate to:
```
http://127.0.0.1:5000
```

## API Integration (Future)

The long-term goal of this project is to provide a simple API that web applications can use to add speech capabilities:

- POST to `/api/tts` with text to convert to speech
- WebSocket connection for real-time STT processing
- Simple JavaScript client library for easy integration
