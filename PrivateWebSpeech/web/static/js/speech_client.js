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
		const selectedType = 'audio/wav';
		if (!MediaRecorder.isTypeSupported(selectedType)) {
			console.warn('audio/wav not supported, falling back to default');
			mediaRecorder = new MediaRecorder(stream);
		} else {
			console.log('Using MIME type:', selectedType);
			mediaRecorder = new MediaRecorder(stream, { mimeType: selectedType });
		}
		
		mediaRecorder.ondataavailable = (event) => {
			if (event.data.size > 0) {
				console.log('Chunk generated, size:', event.data.size, 'state:', mediaRecorder.state);
				const reader = new FileReader();
				reader.onloadend = () => {
					if (reader.result) {
						const base64data = reader.result.split(',')[1];
						socket.emit('audio_chunk_data', base64data);
						console.log('Sent WAV chunk, size:', event.data.size);
					} else {
						console.error('Failed to read WAV chunk');
					}
				};
				reader.onerror = () => console.error('Error reading WAV chunk');
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
	document.getElementById('recordStatus').textContent = 'Recording stopped';
	// Add paragraph break when recording stops
	currentTranscript += "\n\n";
	const transcriptionBox = document.getElementById('transcriptionBox');
	transcriptionBox.innerText = currentTranscript;
	transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
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

async function decodeAndPlayAudio(audioData, isTTS = false) {
	try {
		const binaryString = atob(audioData);
		console.log('Decoding chunk, size:', binaryString.length);
		const bytes = new Uint8Array(binaryString.length);
		for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
		const type = isTTS ? 'audio/webm' : 'audio/wav';
		const blob = new Blob([bytes], { type: type });
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
	currentTranscript += text;
	const transcriptionBox = document.getElementById('transcriptionBox');
	transcriptionBox.innerText = currentTranscript;
	transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
	console.log('Appended transcription:', text);
}

function appendStreamingTranscript(text) {
	currentTranscript += text + " ";
	const transcriptionBox = document.getElementById('transcriptionBox');
	transcriptionBox.innerText = currentTranscript;
	transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
	console.log('Appended streaming transcription:', text);
}

function formatTimestamp(unixTimestamp) {
	const date = new Date(unixTimestamp * 1000);
	return date.toTimeString().split(' ')[0];
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
socket.on('tts_chunk_ready', (data) => {
	console.log('Received TTS chunk');
	socket.emit('log_event', `Received TTS chunk ${chunkCount}`);
	decodeAndPlayAudio(data.chunk, true).then(() => {
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
socket.on('streaming_transcription', (data) => { 
	console.log('Received streaming transcription:', data.text); 
	appendStreamingTranscript(data.text);
	// Update VAD timing display
	document.getElementById('vadStartTime').textContent = formatTimestamp(data.start_time);
	document.getElementById('vadEndTime').textContent = formatTimestamp(data.end_time);
});
socket.on('vad_detection', (data) => console.log('Voice activity detected:', data.segments));
socket.on('processing_error', (data) => {
	console.error('Processing error:', data.message);
	document.getElementById('recordStatus').textContent = `Error: ${data.message}`;
	socket.emit('log_event', `Processing error: ${data.message}`);
});

window.onload = () => updateButtons();