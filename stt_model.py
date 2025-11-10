from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionStream
import sounddevice as sd
import numpy as np
import queue
import threading
import time

# 🔧 Audio Configuration
samplerate = 16000
block_duration = 0.25  # seconds per audio block (small = low latency)
chunk_duration = 1.0   # process every 1 second for better accuracy
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

# 🧠 Model Initialization
model_size = "tiny.en"  # Fastest for CPU
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# 🎙️ Queues & Buffers
audio_queue = queue.Queue()
audio_buffer = []

# ⚡ Streaming Whisper
stream = TranscriptionStream(model, language="en")

# 🎧 Callback for microphone input
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(indata.copy())

# 🎙️ Audio Recorder Thread
def recorder():
    """Continuously capture microphone input and store it in a queue."""
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        callback=audio_callback,
        blocksize=frames_per_block,
    ):
        print("🎧 Listening... Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)

# 🧠 Transcriber Thread
def transcriber():
    """Continuously consume audio from queue and transcribe."""
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            # Combine and reset buffer
            audio_data = np.concatenate(audio_buffer)
            audio_buffer = []

            # Flatten to mono float32 array
            audio_data = audio_data.flatten().astype(np.float32)

            # Feed chunk to streaming model
            stream.feed_audio(audio_data)

            # Get live partial transcript
            text = stream.get_text().strip()
            if text:
                print(f"🗣️ {text}")
            else:
                print("... (no speech detected)")
        time.sleep(0.05)  # prevent busy-waiting

# 🧵 Launch Threads
if __name__ == "__main__":
    threading.Thread(target=recorder, daemon=True).start()
    transcriber()
