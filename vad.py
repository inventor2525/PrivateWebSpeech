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
from typing import Iterator, Dict, List

class VAD:
	'''
	Voice Activity Detector.
	Start after starting a recording with recorder,
	then iterate voice_segments.
	'''
	class Pause:
		'''Pauses the vad when used in a with block.'''
		def __init__(self, vad:'VAD'):
			self.vad = vad
		
		def __enter__(self):
			self.vad.paused = True
			self.vad.clear_buffers()
			return self
		
		def __exit__(self, exc_type, exc_val, exc_tb):
			self.vad.clear_buffers() # Ensure no audio leaks in from before we un-paused.
			self.vad.paused = False
	
	@dataclass
	class _Segment:
		data:np.ndarray
		start_time:datetime
		end_time:datetime

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
		
		# Prep for VAD:
		self.paused = False
		self.running = False
		self.vad_thread: threading.Thread = None
		self.segment_available = threading.Event()
		self.segment_lock = threading.Lock()
		
		self.silent_peeks_buffer: List[VAD._Segment] = []
		self.vocal_segments: List[VAD._Segment] = []
		self.audio_queue: List[VAD._Segment] = []
		
		# Initialize the model
		self.model = Model.from_pretrained(model_path)
		
		# Initialize the pipeline
		self.pipeline = VoiceActivityDetection(segmentation=self.model)
		self.pipeline.instantiate({
			"min_duration_on": min_duration_on,
			"min_duration_off": min_duration_off
		})
		self.pipeline.to(torch.device("cuda"))

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

	def start(self, ignore_prev_audio:bool=False) -> None:
		"""Start the voice activity detection loop."""
		if not self.running:
			self.running = True
			if ignore_prev_audio:
				self.recorder.peek()
			self.vad_thread = threading.Thread(target=self._vad_loop)
			self.vad_thread.start()

	def stop(self) -> None:
		"""Stop the voice activity detection loop."""
		if self.running:
			self.running = False
			self.vad_thread.join()
			self.segment_available.set()  # Ensure the iterator exits if waiting
	
	def _audio_segment_duration(self, segment:np.ndarray) -> float:
		'''Calculates how long this audio segment is (in seconds).'''
		return len(segment) / float(self.sample_rate)

	def _vad_loop(self) -> None:
		"""Main loop for voice activity detection."""
		silent_duration = 0.0
		voice_detected = False
		voice_detected_segments: List[VAD._Segment] = []
		def concat(segs:List[VAD._Segment]) -> VAD._Segment:
			if segs is None or len(segs)==0:
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
			
			# Get a longer segment we can check for voice activity more robustly:
			if not voice_detected:
				# If no voice has been detected lately, we'll use
				# the sum of the current peeked audio and all 'silent'
				# buffers we have before hand to make sure they were
				# truly silent when combined with more recent data:
				if len(self.silent_peeks_buffer) > 0:
					segment_data = concat(self.silent_peeks_buffer + [peek_data]).data
				else:
					segment_data = peek_data.data
			else:
				# Else we'll use the last so many segments, up to window_padding seconds ago:
				segments_duration = self._audio_segment_duration(peek_data.data)
				if segments_duration > self.window_padding:
					segment_data = peek_data.data
				else:
					segments = [peek_data]
					for segment in reversed(voice_detected_segments):
						# Accumulate segments off the end of voice_detected_segments
						# until we have > self.window_padding seconds of audio:
						segments_duration += self._audio_segment_duration(segment.data)
						segments.insert(0, segment)
						if segments_duration > self.window_padding:
							break
					segment_data = concat(segments).data
			
			# Check segment for vocal activity:
			segment_duration = self._audio_segment_duration(segment_data)
			segment_vad_result = self._check_voice_activity(segment_data)
			
			if not voice_detected and len(segment_vad_result):
				# If we haven't been detecting a voice but are now:
				voice_detected = True
				
				# Pull in > self.window_padding seconds worth of audio
				# data we have cached from when there was no voice detected:
				first_voice_time = list(segment_vad_result.get_timeline())[0].start
				required_padding = max(0, self.window_padding - first_voice_time)
				self._trim_buffer_queue(required_padding)
				voice_detected_segments = self.silent_peeks_buffer + [peek_data]
				self.silent_peeks_buffer.clear()
				
				# Calculate how long there hasn't been a voice detected so far:
				silent_duration = segment_duration - list(segment_vad_result.get_timeline())[-1].end
			
			elif voice_detected:
				# If we have been detecting a voice:
				voice_detected_segments.append(peek_data)
				if len(segment_vad_result):
					# And still are:
					silent_duration = segment_duration - list(segment_vad_result.get_timeline())[-1].end
				else:
					# But if we are no longer detecting voice:
					silent_duration += self._audio_segment_duration(peek_data.data)
					
				if silent_duration >= self.window_padding:
					# Then if we haven't been detecting voice long enough:
					full_segment = concat(voice_detected_segments)
					total_duration = self._audio_segment_duration(full_segment.data)
					total_vad_result = self._check_voice_activity(full_segment.data)
					
					# Make sure we actually did have a voice
					# and it ended long enough ago:
					if len(total_vad_result) > 0:
						silent_duration = total_duration - list(total_vad_result.get_timeline())[-1].end
						
						if silent_duration >= self.window_padding:
							# If it's been long enough without a voice,
							# queue this recording to be returned:
							if self.paused:
								self.silent_peeks_buffer.clear()
								voice_detected_segments.clear()
								voice_detected = False
								continue
							with self.segment_lock:
								self.vocal_segments.append(full_segment)
							voice_detected_segments.clear()
							self.segment_available.set()  # Signal that a new vocal segment is available
						# else: Continue recording until the end padding requirement is met
					else:
						# If we didn't in all that we've recorded so far, something
						# went wrong, delete it, and go back to waiting for voice:
						voice_detected = False
						self.silent_peeks_buffer.extend(voice_detected_segments)
						voice_detected_segments.clear()
						
						# Keep some of the silent peeked buffers for latter VAD:
						self._trim_buffer_queue(self.window_padding)
			else:
				# No voice detected, just keep a rolling buffer
				# to make up our padding for once there is some:
				self.silent_peeks_buffer.append(peek_data)
				self._trim_buffer_queue(self.window_padding)
		
		# Finish up by queue'ing any audio we have
		# been building out atm with voice in it:
		if voice_detected:
			last_audio = concat(voice_detected_segments)
			if last_audio and last_audio.data.size > 0:
				if len(self._check_voice_activity(last_audio.data)) > 0:
					with self.segment_lock:
						self.vocal_segments.append(last_audio)

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
		Trim the silent_peeks_buffer queue to maintain only the required duration of audio.

		Args:
			required_duration (float): The required duration of audio to keep in the buffer.
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
		Iterates AudioSegments containing voice as they become available.

		Yields:
			dict: Contains AudioSegment, start_time, and end_time.
		"""
		while self.running:
			self.segment_available.wait()  # Wait for a segment to become available
			with self.segment_lock:
				if self.vocal_segments:
					segment = self.vocal_segments.pop(0)
					if not self.vocal_segments:
						self.segment_available.clear()  # Clear the event if no more segments
					yield_time = datetime.now()
					print(f"yielding audio stopped {(yield_time-segment.end_time).total_seconds()} seconds ago.")
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
		'''
		Returns a 'Pause' object that can be used with
		'with' syntax to block speech input for a time
		'''
		return VAD.Pause(self)