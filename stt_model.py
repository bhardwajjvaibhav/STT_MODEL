# stt_model.py
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

samplerate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []
latest_transcript = ""
recording_thread = None
transcriber_thread = None
stop_event = threading.Event()

model_size = "tiny.en"
model = WhisperModel(model_size, device="cpu", compute_type="float32")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def recorder():
    """Continuously records audio and pushes blocks into a queue."""
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frames_per_block):
        while not stop_event.is_set():
            sd.sleep(100)


def transcriber():
    """Processes audio chunks and updates the latest transcription."""
    global audio_buffer, latest_transcript
    while not stop_event.is_set():
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []
            audio_data = audio_data.flatten().astype(np.float32)

            segments, _ = model.transcribe(
                audio_data,
                language='en',
                vad_filter=True,
                beam_size=1
            )

            text = " ".join([seg.text for seg in segments]).strip()
            latest_transcript += " " + text
            print("Transcribed:", text)


def start_transcription():
    global stop_event, recording_thread, transcriber_thread, latest_transcript
    stop_event.clear()
    latest_transcript = ""
    recording_thread = threading.Thread(target=recorder, daemon=True)
    transcriber_thread = threading.Thread(target=transcriber, daemon=True)
    recording_thread.start()
    transcriber_thread.start()


def stop_transcription():
    stop_event.set()


def get_transcription():
    """Returns current transcription text."""
    return latest_transcript.strip()
