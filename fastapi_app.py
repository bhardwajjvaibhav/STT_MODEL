# server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np

app = FastAPI()

# Load Whisper model once
model = WhisperModel("base.en", device="cpu", compute_type="int8")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            audio_array = np.frombuffer(data, dtype=np.float32)

            # Perform transcription
            segments, _ = model.transcribe(audio_array, language="en")
            text = " ".join([seg.text for seg in segments])

            if text.strip():
                await websocket.send_text(text.strip())

    except WebSocketDisconnect:
        print("⚠️ Client disconnected")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
    finally:
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close()
        print("🔚 Connection closed")
