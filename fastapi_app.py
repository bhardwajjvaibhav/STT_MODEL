# fastapi_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import stt_model

app = FastAPI()

# Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local testing; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/start")
def start_transcription():
    stt_model.start_transcription()
    return {"status": "started"}

@app.get("/stop")
def stop_transcription():
    stt_model.stop_transcription()
    return {"status": "stopped"}

@app.get("/transcript")
def get_transcript():
    text = stt_model.get_transcription()
    return {"text": text}
