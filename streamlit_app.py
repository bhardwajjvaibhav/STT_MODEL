# app.py
import streamlit as st
import sounddevice as sd
import numpy as np
import asyncio
import websockets
import queue
import threading

# ========== Session Initialization ==========
for key, default in {
    "is_recording": False,
    "status": "🟡 Idle",
    "transcribed_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

audio_queue = queue.Queue()

# ========== Streamlit UI ==========
st.title("🎙️ Real-time Speech-to-Text (Whisper + FastAPI)")
st.markdown("Stream live microphone input to a Whisper model via WebSocket")

status = st.empty()
transcript_box = st.empty()

# ========== Audio Recorder ==========
def audio_callback(indata, frames, time, status):
    if st.session_state.get("is_recording", False):
        audio_queue.put(indata.copy())

def start_recording():
    st.session_state.is_recording = True
    st.session_state.status = "🔴 Recording..."
    threading.Thread(target=record_audio, daemon=True).start()

def stop_recording():
    st.session_state.is_recording = False
    st.session_state.status = "🟢 Stopped"

def record_audio():
    samplerate = 16000
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback, dtype='float32'):
        while st.session_state.get("is_recording", False):
            sd.sleep(100)

# ========== WebSocket Client ==========
async def websocket_client():
    uri = "ws://127.0.0.1:8000/ws"
    try:
        async with websockets.connect(uri, max_size=10_000_000) as websocket:
            print("✅ Connected to FastAPI server")

            async def sender():
                while st.session_state.get("is_recording", False):
                    if not audio_queue.empty():
                        audio_chunk = audio_queue.get()
                        await websocket.send(audio_chunk.tobytes())
                    await asyncio.sleep(0.1)

            async def receiver():
                while True:
                    try:
                        text = await websocket.recv()
                        if text.strip():
                            st.session_state.transcribed_text += " " + text
                    except websockets.ConnectionClosed:
                        break

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")

# ========== Async Wrapper ==========
def run_asyncio_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_client())
    loop.close()

# ========== Buttons ==========
col1, col2 = st.columns(2)
if col1.button("🎙️ Start Recording"):
    start_recording()
    threading.Thread(target=run_asyncio_client, daemon=True).start()

if col2.button("⏹️ Stop Recording"):
    stop_recording()

# ========== Live Status ==========
status.markdown(f"### {st.session_state.status}")
transcript_box.text_area("📝 Live Transcription", st.session_state.transcribed_text, height=300)
