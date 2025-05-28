import threading
import time
import numpy as np
import torch
from datetime import datetime
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment
import soundfile as sf
from dataclasses import dataclass
from typing import Iterator, Dict

class VAD:
    """
    Voice Activity Detector.
    Processes audio chunks in a threaded loop, yielding AudioSegment objects with voice and timestamps.
    """
    class Pause:
        """Pauses the VAD when used in a with block."""
        def __init__(self, vad: 'VAD'):
            self.vad = vad

        def __enter__(self):
            self.vad.paused = True
            self.vad.clear_buffers()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.vad.clear_buffers()
            self.vad.paused = False

    @dataclass
    class _Segment:
        data: np.ndarray
        start_time: datetime
        end_time: datetime

    def __init__(self, model_path="pytorch_model.bin", sample_rate=16000, 
                 min_duration_on=0.0, min_duration_off=0.0, 
                 peek_interval=0.5, window_padding=1.0):
        """
        Initialize the Voice Activity Detector.

        Args:
            model_path (str): Path to pyannote.audio model.
            sample_rate (int): Audio sample rate (Hz, matches PrivateWebSpeech WAV).
            min_duration_on (float): Minimum duration of voice activity (seconds).
            min_duration_off (float): Minimum duration of silence (seconds).
            peek_interval (float): Interval between audio peeks (seconds).
            window_padding (float): Padding before/after voice activity (seconds).
        """
        self.sample_rate = sample_rate
        self.peek_interval = peek_interval
        self.window_padding = window_padding

        self.model = Model.from_pretrained(model_path)
        self.pipeline = VoiceActivityDetection(segmentation=self.model)
        self.pipeline.instantiate({
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off
        })
        self.pipeline.to(torch.device("cuda"))

        self.paused = False
        self.running = False
        self.vad_thread: threading.Thread = None
        self.segment_available = threading.Event()
        self.segment_lock = threading.Lock()
        self.silent_peeks_buffer: list[VAD._Segment] = []
        self.vocal_segments: list[VAD._Segment] = []
        self.audio_queue: list[VAD._Segment] = []

    def add_audio_chunk(self, wav_path: str) -> None:
        """
        Add a WAV audio chunk to the processing queue.

        Args:
            wav_path (str): Path to WAV file.
        """
        start_time = datetime.now()
        audio_data, _ = sf.read(wav_path)
        if audio_data.size == 0:
            return
        segment = VAD._Segment(
            data=audio_data,
            start_time=start_time,
            end_time=datetime.now()
        )
        with self.segment_lock:
            self.audio_queue.append(segment)

    def clear_buffers(self) -> None:
        """Clear audio buffers when pausing or resetting."""
        with self.segment_lock:
            self.audio_queue.clear()
            self.silent_peeks_buffer.clear()

    def start(self) -> None:
        """Start the voice activity detection loop."""
        if not self.running:
            self.running = True
            self.vad_thread = threading.Thread(target=self._vad_loop)
            self.vad_thread.start()

    def stop(self) -> None:
        """Stop the voice activity detection loop."""
        if self.running:
            self.running = False
            self.vad_thread.join()
            self.segment_available.set()

    def _audio_segment_duration(self, segment: np.ndarray) -> float:
        """Calculate duration of an audio segment in seconds."""
        return len(segment) / float(self.sample_rate)

    def _vad_loop(self) -> None:
        """Main loop for voice activity detection."""
        silent_duration = 0.0
        voice_detected = False
        voice_detected_segments: list[VAD._Segment] = []

        def concat(segs: list[VAD._Segment]) -> VAD._Segment:
            if not segs:
                return None
            return VAD._Segment(
                np.concatenate([b.data for b in segs]),
                segs[0].start_time,
                segs[-1].end_time
            )

        while self.running:
            time.sleep(self.peek_interval)
            peek_data = None
            with self.segment_lock:
                if self.audio_queue:
                    peek_data = self.audio_queue.pop(0)

            if peek_data is None or peek_data.data.size == 0:
                continue

            if self.paused:
                self.silent_peeks_buffer.clear()
                voice_detected_segments.clear()
                voice_detected = False
                continue

            if not voice_detected:
                if len(self.silent_peeks_buffer) > 0:
                    segment_data = concat(self.silent_peeks_buffer + [peek_data]).data
                else:
                    segment_data = peek_data.data
            else:
                segments_duration = self._audio_segment_duration(peek_data.data)
                if segments_duration > self.window_padding:
                    segment_data = peek_data.data
                else:
                    segments = [peek_data]
                    for segment in reversed(voice_detected_segments):
                        segments_duration += self._audio_segment_duration(segment.data)
                        segments.insert(0, segment)
                        if segments_duration > self.window_padding:
                            break
                    segment_data = concat(segments).data

            segment_duration = self._audio_segment_duration(segment_data)
            try:
                segment_vad_result = self._check_voice_activity(segment_data)
            except Exception as e:
                print(f"VAD processing error: {e}")
                segment_vad_result = []  # Empty result to continue processing

            if not voice_detected and len(segment_vad_result):
                voice_detected = True
                first_voice_time = list(segment_vad_result.get_timeline())[0].start
                required_padding = max(0, self.window_padding - first_voice_time)
                self._trim_buffer_queue(required_padding)
                voice_detected_segments = self.silent_peeks_buffer + [peek_data]
                self.silent_peeks_buffer.clear()
                silent_duration = segment_duration - list(segment_vad_result.get_timeline())[-1].end

            elif voice_detected:
                voice_detected_segments.append(peek_data)
                if len(segment_vad_result):
                    silent_duration = segment_duration - list(segment_vad_result.get_timeline())[-1].end
                else:
                    silent_duration += self._audio_segment_duration(peek_data.data)

                if silent_duration >= self.window_padding:
                    full_segment = concat(voice_detected_segments)
                    total_duration = self._audio_segment_duration(full_segment.data)
                    try:
                        total_vad_result = self._check_voice_activity(full_segment.data)
                    except Exception as e:
                        print(f"VAD processing error on full segment: {e}")
                        total_vad_result = []

                    if len(total_vad_result) > 0:
                        silent_duration = total_duration - list(total_vad_result.get_timeline())[-1].end
                        if silent_duration >= self.window_padding:
                            if self.paused:
                                self.silent_peeks_buffer.clear()
                                voice_detected_segments.clear()
                                voice_detected = False
                                continue
                            with self.segment_lock:
                                self.vocal_segments.append(full_segment)
                                voice_detected_segments.clear()
                                self.segment_available.set()
                    else:
                        voice_detected = False
                        self.silent_peeks_buffer.extend(voice_detected_segments)
                        voice_detected_segments.clear()
                        self._trim_buffer_queue(self.window_padding)
            else:
                self.silent_peeks_buffer.append(peek_data)
                self._trim_buffer_queue(self.window_padding)

        if voice_detected:
            last_audio = concat(voice_detected_segments)
            if last_audio and last_audio.data.size > 0:
                try:
                    if len(self._check_voice_activity(last_audio.data)) > 0:
                        with self.segment_lock:
                            self.vocal_segments.append(last_audio)
                except Exception as e:
                    print(f"VAD processing error on final segment: {e}")

    def _check_voice_activity(self, audio_data: np.ndarray) -> VoiceActivityDetection:
        """
        Check for voice activity in the given audio data.

        Args:
            audio_data (np.ndarray): The audio data to check.

        Returns:
            VoiceActivityDetection: The result of the voice activity detection.
        """
        # Log input audio data statistics
        print(f"Input audio_data shape: {audio_data.shape}, size: {audio_data.size}")
        print(f"Input audio_data dtype: {audio_data.dtype}")
        print(f"Input audio_data min: {np.min(audio_data)}, max: {np.max(audio_data)}, mean: {np.mean(audio_data)}")

        # Ensure audio_data is 1D and convert to (1, time) tensor for pyannote
        if audio_data.ndim > 1:
            print(f"Flattening audio_data from shape {audio_data.shape} to 1D")
            audio_data = audio_data.flatten()
        
        # Log audio data after flattening (if applicable)
        print(f"Post-flatten audio_data shape: {audio_data.shape}, size: {audio_data.size}")
        print(f"Post-flatten audio_data min: {np.min(audio_data)}, max: {np.max(audio_data)}, mean: {np.mean(audio_data)}")

        # Convert to PyTorch tensor and unsqueeze
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)  # Shape: (1, time)
        
        # Log final waveform tensor before pipeline
        print(f"Waveform tensor shape: {waveform.shape}, size: {waveform.numel()}")
        print(f"Waveform tensor min: {waveform.min().item()}, max: {waveform.max().item()}, mean: {waveform.mean().item()}")

        # Process with pipeline
        return self.pipeline({'waveform': waveform, 'sample_rate': self.sample_rate})
    def _trim_buffer_queue(self, required_duration: float) -> None:
        """
        Trim the silent_peeks_buffer queue to maintain only the required duration.

        Args:
            required_duration (float): Duration of audio to keep in the buffer.
        """
        current_duration = 0.0
        for i, buffer in enumerate(reversed(self.silent_peeks_buffer)):
            buffer_duration = self._audio_segment_duration(buffer.data)
            current_duration += buffer_duration
            if current_duration >= required_duration:
                self.silent_peeks_buffer = self.silent_peeks_buffer[-(i+1):]
                break

    def voice_segments(self) -> Iterator[Dict[str, AudioSegment | datetime]]:
        """
        Iterate AudioSegments containing voice as they become available.

        Yields:
            dict: Contains AudioSegment, start_time, and end_time.
        """
        while self.running:
            self.segment_available.wait()
            with self.segment_lock:
                if self.vocal_segments:
                    segment = self.vocal_segments.pop(0)
                    if not self.vocal_segments:
                        self.segment_available.clear()
                    yield_time = datetime.now()
                    print(f"Yielding audio stopped {(yield_time - segment.end_time).total_seconds()} seconds ago.")
                    # Scale float64 [-1, 1] to int16 [-32768, 32767]
                    scaled_data = (segment.data * 32767).astype(np.int16)
                    audio_segment = AudioSegment(
                        data=scaled_data.tobytes(),
                        sample_width=2,  # 16-bit PCM
                        frame_rate=self.sample_rate,
                        channels=1
                    )
                    yield {
                        "audio": audio_segment,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time
                    }

    def pauser(self) -> 'VAD.Pause':
        """Return a Pause object for blocking speech input."""
        return VAD.Pause(self)
