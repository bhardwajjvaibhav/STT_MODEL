import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Live Speech-to-Text", page_icon="🎤", layout="centered")

st.title("🎤 Real-time Speech-to-Text (Whisper)")
st.markdown("Start and stop live transcription using the buttons below.")

if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
status_placeholder = st.empty()
text_placeholder = st.empty()

def fetch_transcript():
    try:
        r = requests.get(f"{API_URL}/transcript")
        if r.status_code == 200:
            return r.json().get("text", "")
    except Exception:
        return ""
    return ""

def start_recording():
    try:
        requests.get(f"{API_URL}/start")
        st.session_state.running = True
    except Exception:
        st.error("❌ Could not connect to backend. Make sure FastAPI is running.")

def stop_recording():
    try:
        requests.get(f"{API_URL}/stop")
        st.session_state.running = False
    except Exception:
        st.error("❌ Failed to stop transcription.")

with col1:
    if st.button("▶️ Start"):
        start_recording()

with col2:
    if st.button("⏹️ Stop"):
        stop_recording()

if st.session_state.running:
    with st.spinner("🎧 Listening and transcribing..."):
        while st.session_state.running:
            text = fetch_transcript()
            # Use a placeholder to update content dynamically
            text_placeholder.markdown(
                f"### 🗒️ Live Transcription:\n\n{text if text else '*Listening...*'}"
            )
            time.sleep(1)
else:
    status_placeholder.info("Click **Start** to begin recording.")
