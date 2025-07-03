from gtts import gTTS
import tempfile
import whisper
import os


def text_to_speech(text: str) -> str:
    """Convert text to speech and return path to mp3 file."""
    tts = gTTS(text, lang="tr")
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    tts.save(path)
    return path


def speech_to_text(audio_path: str, model_size: str = "base") -> str:
    """Transcribe Turkish audio using Whisper."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="tr")
    return result.get("text", "").strip()
