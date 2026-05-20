from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from gtts import gTTS
import io

# Create router for TTS service
tts_router = APIRouter(prefix="/api", tags=["Text to Speech"])

@tts_router.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text")
    lang = data.get("lang", "te")  # Default to Telugu

    # Generate audio in memory
    tts = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    return StreamingResponse(mp3_fp, media_type="audio/mpeg")