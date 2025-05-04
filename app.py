from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit
import time
import os
import base64
import torch
import soundfile as sf
from kokoro import KPipeline
import tempfile
import shutil
import numpy as np
import threading
from faster_whisper import WhisperModel
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import subprocess
import warnings
import queue

warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Dictionaries for session management
client_files = {}
last_recordings = {}
client_chunks = {}
CHUNK_SIZE = 64 * 1024

# Pre-load models
print("Pre-loading models...")
kokoro_pipeline = None
vad_model = None
vad_pipeline = None
whisper_model = None

def get_kokoro_pipeline():
    global kokoro_pipeline
    if kokoro_pipeline is None:
        print("Initializing Kokoro TTS pipeline...")
        kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    return kokoro_pipeline

def get_vad_pipeline():
    global vad_model, vad_pipeline
    if vad_model is None:
        print("Initializing Voice Activity Detection model...")
        vad_model = Model.from_pretrained("pytorch_model.bin")
        vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
        HYPER_PARAMETERS = {"min_duration_on": 0.0, "min_duration_off": 0.0}
        vad_pipeline.instantiate(HYPER_PARAMETERS)
    return vad_pipeline

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Initializing Fast Whisper model...")
        whisper_model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
    return whisper_model

kokoro_pipeline = get_kokoro_pipeline()
vad_pipeline = get_vad_pipeline()
whisper_model = get_whisper_model()

# Utility functions
def get_file_duration(filename):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', filename
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration of {filename}: {e}")
        return 0

def convert_to_wav(input_file, output_file):
    try:
        subprocess.run([
            'ffmpeg', '-err_detect', 'ignore_err', '-fflags', '+nobuffer',
            '-flags', 'low_delay', '-i', input_file, '-ar', '16000', '-ac', '1',
            '-f', 'wav', '-y', output_file
        ], check=True, stderr=subprocess.PIPE, timeout=15)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting to wav: {e.stderr.decode()}")
        return False
    except subprocess.TimeoutExpired:
        print("FFmpeg conversion timed out")
        return False

def convert_to_webm(input_file, output_file):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file, '-c:a', 'libopus', '-b:a', '128k',
            '-f', 'webm', '-y', output_file
        ], check=True, stderr=subprocess.PIPE, timeout=10)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting to WebM: {e.stderr.decode()}")
        return False
    except subprocess.TimeoutExpired:
        print("FFmpeg WebM conversion timed out")
        return False

def remux_webm(input_file, output_file):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file, '-c', 'copy', '-f', 'webm', '-y', output_file
        ], check=True, stderr=subprocess.PIPE, timeout=10)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error remuxing WebM: {e.stderr.decode()}")
        return False
    except subprocess.TimeoutExpired:
        print("FFmpeg remuxing timed out")
        return False

def detect_voice_activity(audio_file):
    pipeline = get_vad_pipeline()
    vad_result = pipeline(audio_file)
    segments = [{"start": s.start, "end": s.end, "duration": s.duration} for s in vad_result.get_timeline()]
    return segments

def transcribe_audio(audio_file):
    model = get_whisper_model()
    segments, _ = model.transcribe(audio_file, beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

# HTML with updated JavaScript
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Audio Streaming</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; background-color: #f0f0f0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; background-color: #4CAF50; color: white; border: none; border-radius: 5px; }
        button:hover { background-color: #45a049; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }
        #audioPlayer { margin-top: 20px; width: 100%; max-width: 500px; }
        .status { margin-top: 10px; font-style: italic; color: #555; }
        textarea { width: 100%; max-width: 500px; height: 150px; margin: 10px 0; padding: 10px; border-radius: 5px; border: 1px solid #ccc; font-family: inherit; resize: vertical; }
        .section { background-color: white; padding: 20px; border-radius: 10px; margin: 15px 0; width: 100%; max-width: 550px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h2 { color: #333; margin-top: 0; }
        #transcriptionBox { width: 100%; max-width: 500px; height: 200px; margin: 10px 0; padding: 10px; border-radius: 5px; border: 1px solid #ccc; font-family: inherit; resize: vertical; background-color: #f9f9f9; overflow-y: auto; }
        .full-transcript { 
            margin-top: 20px;
            padding: 10px;
            background-color: #e8f4f8;
            border: 1px solid #a8d1df;
            border-radius: 5px;
            white-space: pre-wrap;
            font-family: monospace;
        }
    </style>
    <script>
        const socket = io();
        let mediaRecorder, stream, audioQueue = [], isPlaying = false, isRecording = false, isSpeaking = false, currentTranscript = "", chunkCount = 0, isPlayingAudio = false;

        function updateButtons() {
            document.getElementById('startBtn').disabled = isRecording || isSpeaking;
            document.getElementById('stopBtn').disabled = !isRecording;
            document.getElementById('playLastBtn').disabled = isRecording || isPlaying || isSpeaking;
            document.getElementById('speakBtn').disabled = isRecording || isPlaying || isSpeaking;
            document.getElementById('clearTranscriptBtn').disabled = isRecording;
        }

        function clearTranscript() {
            currentTranscript = "";
            document.getElementById('transcriptionBox').innerText = "";
            // Also remove full transcript if present
            const fullTranscriptDiv = document.getElementById('fullTranscript');
            if (fullTranscriptDiv) {
                fullTranscriptDiv.remove();
            }
            console.log('Transcription cleared');
        }

        async function startRecording() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
                console.log('Audio stream initialized:', stream);
                const selectedType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(selectedType)) {
                    console.warn('audio/webm;codecs=opus not supported, falling back to default');
                    mediaRecorder = new MediaRecorder(stream);
                } else {
                    console.log('Using MIME type:', selectedType);
                    mediaRecorder = new MediaRecorder(stream, { mimeType: selectedType, audioBitsPerSecond: 128000 });
                }
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        console.log('Chunk generated, size:', event.data.size, 'state:', mediaRecorder.state);
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            if (reader.result) {
                                const base64data = reader.result.split(',')[1];
                                socket.emit('audio_chunk_data', base64data);
                                console.log('Sent WebM chunk, size:', event.data.size);
                            } else {
                                console.error('Failed to read WebM chunk');
                            }
                        };
                        reader.onerror = () => console.error('Error reading WebM chunk');
                        reader.readAsDataURL(event.data);
                    } else {
                        console.warn('Received empty audio chunk');
                    }
                };
                
                mediaRecorder.onstop = () => {
                    socket.emit('stop_recording');
                    console.log('Recording stopped');
                };
                
                mediaRecorder.onerror = (event) => {
                    console.error('MediaRecorder error:', event.error);
                    document.getElementById('recordStatus').textContent = `Recording error: ${event.error}`;
                };
                
                mediaRecorder.start(500);
                console.log('MediaRecorder started, interval: 500ms');
                socket.emit('start_recording');
                isRecording = true;
                updateButtons();
                document.getElementById('recordStatus').textContent = 'Recording...';
            } catch (err) {
                console.error('Error starting recording:', err);
                alert('Failed to start recording: ' + err.message);
                document.getElementById('recordStatus').textContent = `Error: ${err.message}`;
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            isRecording = false;
            updateButtons();
            document.getElementById('recordStatus').textContent = 'Recording stopped, processing full transcription...';
        }

        function playLastRecording() {
            isPlaying = true;
            updateButtons();
            document.getElementById('playStatus').textContent = 'Loading recording...';
            audioQueue = [];
            chunkCount = 0;
            socket.emit('play_last_recording');
        }

        function speakText() {
            const text = document.getElementById('ttsText').value.trim();
            if (text.length === 0) {
                alert('Please enter some text to speak');
                return;
            }
            isSpeaking = true;
            updateButtons();
            document.getElementById('ttsStatus').textContent = 'Generating speech...';
            audioQueue = [];
            chunkCount = 0;
            socket.emit('speak_text', {text: text});
        }

        async function decodeAndPlayAudio(audioData) {
            try {
                const binaryString = atob(audioData);
                console.log('Decoding chunk, size:', binaryString.length);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
                const blob = new Blob([bytes], { type: 'audio/webm' });
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                audioQueue.push(audio);
                socket.emit('log_event', `Pushed audio chunk ${chunkCount} to queue for playback, size: ${binaryString.length}`);
                console.log(`Pushed chunk ${chunkCount} to queue, queue length: ${audioQueue.length}`);
            } catch (error) {
                console.error('Error decoding audio data:', error);
                socket.emit('log_event', `Error decoding audio chunk: ${error.message}`);
                document.getElementById('playStatus').textContent = 'Playback error';
                throw error;
            }
        }

        function playNextInQueueOnEnded() {
            if (audioQueue.length > 0) {
                if (isPlayingAudio) {
                    socket.emit('log_event', `Waiting to play chunk ${chunkCount - audioQueue.length}, another chunk is playing`);
                    return;
                }
                isPlayingAudio = true;
                const audio = audioQueue.shift();
                socket.emit('log_event', `Playing audio chunk ${chunkCount - audioQueue.length}, queue length: ${audioQueue.length}`);
                console.log(`Playing chunk ${chunkCount - audioQueue.length}, queue length: ${audioQueue.length}`);
                audio.onended = () => {
                    URL.revokeObjectURL(audio.src);
                    isPlayingAudio = false;
                    playNextInQueueOnEnded();
                };
                audio.play().catch(error => {
                    console.error('Playback error:', error);
                    socket.emit('log_event', `Error playing audio chunk: ${error.message}`);
                    document.getElementById('playStatus').textContent = 'Playback error';
                    isPlayingAudio = false;
                    playNextInQueueOnEnded();
                });
                if (isPlaying) document.getElementById('playStatus').textContent = 'Playing recording...';
                if (isSpeaking) document.getElementById('ttsStatus').textContent = 'Speaking...';
            } else {
                isPlaying = isSpeaking = isPlayingAudio = false;
                updateButtons();
                document.getElementById('playStatus').textContent = 'Playback finished';
                document.getElementById('ttsStatus').textContent = 'Speech completed';
                socket.emit('log_event', 'Playback queue emptied');
            }
        }

        function startPlayback() {
            socket.emit('log_event', `Attempting to start playback, queue length: ${audioQueue.length}, isPlayingAudio: ${isPlayingAudio}`);
            console.log(`Attempting to start playback, queue length: ${audioQueue.length}, isPlayingAudio: ${isPlayingAudio}`);
            if (audioQueue.length > 0 && !isPlayingAudio) {
                playNextInQueueOnEnded();
            }
        }

        function appendTranscript(text) {
            if (text && text.trim()) {
                currentTranscript += text;
                const transcriptionBox = document.getElementById('transcriptionBox');
                transcriptionBox.innerText = currentTranscript;
                transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
                console.log('Appended transcription:', text);
            } else {
                console.log('Empty or invalid transcription received');
            }
        }
        
        function displayFullTranscript(text) {
            // Remove any existing full transcript display
            let fullTranscriptDiv = document.getElementById('fullTranscript');
            if (fullTranscriptDiv) {
                fullTranscriptDiv.remove();
            }
            
            // Create new div for the full transcript
            fullTranscriptDiv = document.createElement('div');
            fullTranscriptDiv.id = 'fullTranscript';
            fullTranscriptDiv.className = 'full-transcript';
            fullTranscriptDiv.textContent = text;
            
            // Add it after the transcription section
            const transcriptionSection = document.querySelector('.section:nth-child(3)');
            transcriptionSection.appendChild(fullTranscriptDiv);
            console.log('Displayed full transcript');
        }

        socket.on('recording_started', (data) => console.log('Recording started, filename:', data.filename));
        socket.on('recording_stopped', () => { console.log('Recording stopped'); isRecording = false; updateButtons(); });
        socket.on('audio_chunk', (data) => {
            console.log('Received audio chunk of size:', data.chunk.length);
            socket.emit('log_event', `Received audio chunk ${chunkCount} for playback`);
            decodeAndPlayAudio(data.chunk).then(() => {
                chunkCount++;
                startPlayback();
            }).catch(() => {});
        });
        socket.on('playback_complete', () => {
            console.log('Server indicates playback complete');
            socket.emit('log_event', 'Server indicated playback complete');
        });
        socket.on('playback_error', (data) => {
            console.error('Playback error:', data.message);
            alert('Playback error: ' + data.message);
            isPlaying = isSpeaking = isPlayingAudio = false;
            updateButtons();
            document.getElementById('playStatus').textContent = 'Playback error';
            document.getElementById('ttsStatus').textContent = '';
            socket.emit('log_event', `Playback error: ${data.message}`);
        });
        socket.on('tts_started', (data) => console.log('TTS started for:', data.filename));
        socket.on('tts_chunk_ready', (data) => {
            console.log('Received TTS chunk');
            socket.emit('log_event', `Received TTS chunk ${chunkCount}`);
            decodeAndPlayAudio(data.chunk).then(() => {
                chunkCount++;
                startPlayback();
            }).catch(() => {});
        });
        socket.on('tts_error', (data) => {
            console.error('TTS error:', data.message);
            alert('TTS error: ' + data.message);
            isSpeaking = isPlayingAudio = false;
            updateButtons();
            document.getElementById('ttsStatus').textContent = 'TTS error';
            socket.emit('log_event', `TTS error: ${data.message}`);
        });
        socket.on('tts_complete', () => {
            console.log('TTS processing complete');
            socket.emit('log_event', 'TTS processing complete');
        });
        socket.on('transcription', (data) => { console.log('Received transcription:', data.text); appendTranscript(data.text); });
        socket.on('full_transcription', (data) => { 
            console.log('Received full transcription:', data.text); 
            document.getElementById('recordStatus').textContent = 'Recording stopped';
            displayFullTranscript(data.text); 
        });
        socket.on('vad_detection', (data) => console.log('Voice activity detected:', data.segments));
        socket.on('processing_error', (data) => {
            console.error('Processing error:', data.message);
            document.getElementById('recordStatus').textContent = `Error: ${data.message}`;
            socket.emit('log_event', `Processing error: ${data.message}`);
        });

        window.onload = () => updateButtons();
    </script>
</head>
<body>
    <h1>Audio Streaming & Text-to-Speech</h1>
    <div class="section">
        <h2>Voice Recording</h2>
        <button id="startBtn" onclick="startRecording()">Start Recording</button>
        <button id="stopBtn" onclick="stopRecording()">Stop Recording</button>
        <button id="playLastBtn" onclick="playLastRecording()">Play Last Recording</button>
        <div id="recordStatus" class="status">Ready to record</div>
        <div id="playStatus" class="status"></div>
    </div>
    <div class="section">
        <h2>Text-to-Speech</h2>
        <textarea id="ttsText" placeholder="Enter text to be spoken..."></textarea>
        <button id="speakBtn" onclick="speakText()">Speak Text</button>
        <div id="ttsStatus" class="status">Ready to speak</div>
    </div>
    <div class="section">
        <h2>Transcription</h2>
        <div id="transcriptionBox"></div>
        <button id="clearTranscriptBtn" onclick="clearTranscript()">Clear Transcript</button>
    </div>
</body>
</html>
'''

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    sid = request.sid
    if sid in client_files:
        client_files[sid]['file'].close()
        del client_files[sid]
    if sid in client_chunks:
        del client_chunks[sid]
    if sid in last_recordings:
        filename = last_recordings[sid]
        if os.path.exists(filename):
            os.remove(filename)
        del last_recordings[sid]

@socketio.on('log_event')
def handle_log_event(message):
    print(f"Client log: {message}")

@socketio.on('start_recording')
def start_recording():
    sid = request.sid
    timestamp = int(time.time())
    filename = f"recording_{sid}_{timestamp}.webm"
    client_files[sid] = {'file': open(filename, 'wb'), 'lock': threading.Lock()}
    client_chunks[sid] = {'header': None, 'chunk_count': 0}
    last_recordings[sid] = filename
    emit('recording_started', {'filename': filename})
    print(f"Started recording for session {sid}, saving to {filename}")

@socketio.on('audio_chunk_data')
def handle_audio_chunk_data(data):
    sid = request.sid
    if sid not in client_files:
        print(f"Session {sid} not found or file closed, ignoring chunk")
        return
    try:
        start_time = time.time()
        binary_data = base64.b64decode(data)
        with client_files[sid]['lock']:
            if client_files[sid]['file'].closed:
                print(f"File for session {sid} is closed, ignoring chunk")
                return
            client_files[sid]['file'].write(binary_data)
            client_files[sid]['file'].flush()
        if client_chunks[sid]['chunk_count'] == 0:
            client_chunks[sid]['header'] = binary_data
            client_chunks[sid]['chunk_count'] += 1
            print(f"Stored header for session {sid}, size: {len(binary_data)}, first 20 bytes: {binary_data[:20].hex()}")
            return
        client_chunks[sid]['chunk_count'] += 1
        temp_dir = tempfile.mkdtemp()
        try:
            temp_webm = os.path.join(temp_dir, "chunk.webm")
            with open(temp_webm, 'wb') as f:
                f.write(client_chunks[sid]['header'])
                f.write(binary_data)
            wav_path = os.path.join(temp_dir, "chunk.wav")
            if convert_to_wav(temp_webm, wav_path) and os.path.exists(wav_path):
                try:
                    vad_results = detect_voice_activity(wav_path)
                    print(f"Voice activity detection for session {sid}: {vad_results}")
                    if vad_results:
                        emit('vad_detection', {'segments': vad_results})
                    else:
                        print(f"No voice activity detected in chunk for session {sid}")
                except Exception as e:
                    print(f"VAD error: {e}")
                    emit('processing_error', {'message': 'VAD processing failed'})
                try:
                    transcription = transcribe_audio(wav_path)
                    if transcription:
                        transcript_text = ' '.join([segment['text'] for segment in transcription]).strip()
                        if transcript_text:
                            print(f"Transcription for session {sid}: {transcript_text}")
                            #emit('transcription', {'text': transcript_text})
                        else:
                            print(f"No transcription text generated for session {sid}")
                    else:
                        print(f"No transcription results for session {sid}")
                except Exception as e:
                    print(f"Transcription error: {e}")
                    emit('processing_error', {'message': 'Transcription failed'})
            else:
                print(f"WAV file not created for session {sid}")
                emit('processing_error', {'message': 'WAV conversion failed'})
        finally:
            shutil.rmtree(temp_dir)
        print(f"Chunk processing took {time.time() - start_time}s")
    except Exception as e:
        print(f"Error processing audio chunk for session {sid}: {e}")
        emit('processing_error', {'message': f'Error processing audio chunk: {str(e)}'})

@socketio.on('stop_recording')
def stop_recording():
    sid = request.sid
    if sid in client_files:
        with client_files[sid]['lock']:
            client_files[sid]['file'].close()
        filename = last_recordings.get(sid)
        if filename and os.path.exists(filename):
            temp_dir = tempfile.mkdtemp()
            try:
                remuxed_file = os.path.join(temp_dir, "remuxed.webm")
                if remux_webm(filename, remuxed_file):
                    shutil.move(remuxed_file, filename)
                    print(f"Remuxed WebM file for session {sid}: {filename}, duration: {get_file_duration(filename)}s")
                    
                    # Convert full recording to WAV for transcription
                    full_wav_path = os.path.join(temp_dir, "full_recording.wav")
                    if convert_to_wav(filename, full_wav_path) and os.path.exists(full_wav_path):
                        try:
                            print(f"Transcribing full recording for session {sid}")
                            transcription = transcribe_audio(full_wav_path)
                            if transcription:
                                transcript_text = ' '.join([segment['text'] for segment in transcription]).strip()
                                if transcript_text:
                                    # Format with delimiters
                                    full_transcript = transcript_text
                                    print(f"Full transcription for session {sid}: {transcript_text}")
                                    emit('transcription', {'text': full_transcript+"\n\n"})
                                    with open(filename.replace("recording_","stt_full_transcription_").replace(".webm",".txt"), 'w') as f:
                                        f.write(full_transcript)
                                else:
                                    print(f"No full transcription text generated for session {sid}")
                            else:
                                print(f"No full transcription results for session {sid}")
                        except Exception as e:
                            print(f"Full transcription error: {e}")
                            emit('processing_error', {'message': f'Full transcription failed: {str(e)}'})
                    else:
                        print(f"Full WAV conversion failed for session {sid}")
                        emit('processing_error', {'message': 'Full recording WAV conversion failed'})
                else:
                    print(f"Failed to remux WebM file for session {sid}")
            finally:
                shutil.rmtree(temp_dir)
        del client_files[sid]
    emit('recording_stopped')
    print(f"Stopped recording for session {sid}")

@socketio.on('play_last_recording')
def play_last_recording():
    sid = request.sid
    if sid not in last_recordings:
        print(f"No previous recording found for session {sid}")
        emit('playback_error', {'message': "No previous recording found"})
        return

    filename = last_recordings[sid]
    if not os.path.exists(filename):
        print(f"Recording file not found: {filename}")
        emit('playback_error', {'message': f"Recording file not found: {filename}"})
        return

    print(f"Playing last recording for session {sid}: {filename}, duration: {get_file_duration(filename)}s")

    temp_dir = tempfile.mkdtemp()
    chunk_queue = queue.Queue()
    chunk_ready = threading.Event()
    stop_event = threading.Event()

    def chunk_worker():
        try:
            # Load the entire WebM file into a temporary file
            full_webm = os.path.join(temp_dir, "full.webm")
            with open(filename, 'rb') as f:
                with open(full_webm, 'wb') as out_f:
                    out_f.write(f.read())

            # Get the duration of the file
            duration = get_file_duration(full_webm)
            if duration <= 0:
                print(f"Invalid duration for {filename}")
                chunk_queue.put(('error', {'message': "Invalid recording duration"}))
                chunk_ready.set()
                return

            # Define chunk duration (3 seconds)
            chunk_duration = 3.0
            num_chunks = int(np.ceil(duration / chunk_duration))
            chunk_num = 0

            # Process each chunk
            for i in range(num_chunks):
                if stop_event.is_set():
                    break
                start_time = i * chunk_duration
                # Cap the duration of the last chunk
                remaining_duration = min(chunk_duration, duration - start_time)
                if remaining_duration <= 0:
                    break

                output_webm = os.path.join(temp_dir, f"chunk_{chunk_num}.webm")
                try:
                    subprocess.run([
                        'ffmpeg', '-i', full_webm,
                        '-ss', str(start_time),  # Start time
                        '-t', str(remaining_duration),  # Capped duration
                        '-c:a', 'copy',  # Copy audio stream without re-encoding
                        '-f', 'webm', '-y', output_webm
                    ], check=True, stderr=subprocess.PIPE, timeout=15)

                    if os.path.exists(output_webm):
                        with open(output_webm, 'rb') as f:
                            chunk_data = f.read()
                        b64_chunk = base64.b64encode(chunk_data).decode('utf-8')
                        chunk_queue.put(('chunk', (chunk_num + 1, b64_chunk, len(chunk_data))))
                        chunk_ready.set()
                        chunk_num += 1
                        os.remove(output_webm)  # Clean up chunk file
                    else:
                        print(f"Failed to create chunk {chunk_num + 1} for session {sid}")
                        chunk_queue.put(('error', {'message': f"Failed to create chunk {chunk_num + 1}"}))
                        chunk_ready.set()
                        break

                except subprocess.CalledProcessError as e:
                    print(f"Error creating chunk {chunk_num + 1}: {e.stderr.decode()}")
                    chunk_queue.put(('error', {'message': f"Error creating chunk {chunk_num + 1}: {e.stderr.decode()}"}))
                    chunk_ready.set()
                    break
                except subprocess.TimeoutExpired:
                    print(f"FFmpeg chunk creation timed out for chunk {chunk_num + 1}")
                    chunk_queue.put(('error', {'message': f"Chunk creation timed out for chunk {chunk_num + 1}"}))
                    chunk_ready.set()
                    break

            # Signal completion
            chunk_queue.put(('complete', None))
            chunk_ready.set()

        except Exception as e:
            print(f"Worker error: {e}")
            chunk_queue.put(('error', {'message': str(e)}))
            chunk_ready.set()

    try:
        # Start the worker thread
        worker_thread = threading.Thread(target=chunk_worker)
        worker_thread.start()

        # Process chunks from the queue
        last_chunk_num = 0
        while True:
            chunk_ready.wait()  # Wait for a chunk to be ready
            chunk_ready.clear()  # Reset the event

            # Process all available chunks in the queue
            while True:
                try:
                    item_type, item_data = chunk_queue.get_nowait()
                    if item_type == 'chunk':
                        chunk_num, b64_chunk, chunk_size = item_data
                        print(f"Sending chunk {chunk_num} of {chunk_size} bytes for playback of last recording")
                        emit('audio_chunk', {'chunk': b64_chunk})
                        socketio.sleep(1)  # Delay to prevent overwhelming the client
                        last_chunk_num = chunk_num
                    elif item_type == 'error':
                        emit('playback_error', item_data)
                        return  # Exit on error
                    elif item_type == 'complete':
                        # Delay completion to ensure the last chunk is processed
                        if last_chunk_num > 0:
                            print(f"Waiting for client to process chunk {last_chunk_num} before completion")
                            socketio.sleep(1)  # Additional delay for last chunk
                        emit('playback_complete')
                        print(f"Playback completed for session {sid}, sent {last_chunk_num} chunks")
                        return  # Exit after completion
                except queue.Empty:
                    break  # No more items in queue, wait for next chunk_ready

    except Exception as e:
        print(f"Error in playback loop: {e}")
        emit('playback_error', {'message': str(e)})
    finally:
        stop_event.set()  # Signal worker to stop
        worker_thread.join()  # Wait for worker to finish
        shutil.rmtree(temp_dir)  # Clean up temporary directory

@socketio.on('speak_text')
def speak_text(data):
    sid = request.sid
    text = data.get('text', '').strip()
    if not text:
        emit('tts_error', {'message': "No text provided"})
        return
    timestamp = int(time.time())
    text_filename = f"tts_text_{sid}_{timestamp}.txt"
    audio_filename = f"tts_audio_{sid}_{timestamp}.webm"
    try:
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        emit('tts_started', {'filename': audio_filename})
        pipeline = get_kokoro_pipeline()
        temp_dir = tempfile.mkdtemp()
        try:
            full_audio = None
            generator = pipeline(text, voice='af_heart')
            chunk_num = 0
            for i, (gs, ps, audio) in enumerate(generator):
                if not isinstance(audio, np.ndarray):
                    audio = np.asarray(audio, dtype=np.float32)
                elif audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                chunk_wav = os.path.join(temp_dir, f"chunk_{sid}_{i}.wav")
                chunk_webm = os.path.join(temp_dir, f"chunk_{sid}_{i}.webm")
                sf.write(chunk_wav, audio, 24000)
                if full_audio is None:
                    full_audio = audio
                else:
                    full_audio = np.concatenate((full_audio, audio))
                if convert_to_webm(chunk_wav, chunk_webm):
                    with open(chunk_webm, 'rb') as f:
                        chunk_data = f.read()
                    b64_chunk = base64.b64encode(chunk_data).decode('utf-8')
                    chunk_num += 1
                    print(f"Sending chunk {chunk_num} of {len(chunk_data)} bytes for speaking text")
                    emit('tts_chunk_ready', {'chunk': b64_chunk})
                    socketio.sleep(0.01)
                    os.remove(chunk_wav)
                    os.remove(chunk_webm)
                else:
                    print(f"Failed to convert chunk {i} to WebM for session {sid}")
                    emit('tts_error', {'message': f"Failed to convert chunk {i} to WebM"})
                    break
            if full_audio is not None:
                temp_wav = os.path.join(temp_dir, f"full_{sid}.wav")
                sf.write(temp_wav, full_audio, 24000)
                if convert_to_webm(temp_wav, audio_filename):
                    print(f"Saved TTS audio as WebM: {audio_filename}")
                else:
                    print(f"Failed to save TTS audio as WebM: {audio_filename}")
            emit('tts_complete')
        finally:
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"TTS error: {e}")
        emit('tts_error', {'message': str(e)})

@app.route('/')
def index():
    return render_template_string(HTML)

if __name__ == '__main__':
    socketio.run(app, debug=True)
