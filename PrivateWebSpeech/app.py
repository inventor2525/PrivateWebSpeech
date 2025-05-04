from flask import Flask, render_template, request
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

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
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
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
