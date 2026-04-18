"""
Piper TTS wrapper with sounddevice playback.

Uses the piper-tts Python package (PiperVoice) directly rather than
the CLI binary, avoiding subprocess overhead.

speak() is synchronous — it blocks until playback is complete.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger
from piper import PiperVoice

_DEFAULT_VOICE = Path(__file__).parent.parent / "models" / "piper" / "en_US-ryan-medium.onnx"

_voice_cache: dict[str, PiperVoice] = {}


def _load_voice(model_path: Path) -> PiperVoice:
    key = str(model_path)
    if key not in _voice_cache:
        logger.info(f"Loading Piper voice: {model_path.name}")
        _voice_cache[key] = PiperVoice.load(str(model_path))
        logger.info("Piper voice loaded.")
    return _voice_cache[key]


def synthesize(text: str, model_path: Optional[Path] = None) -> tuple[np.ndarray, int]:
    """
    Synthesize text to a float32 numpy array.
    Returns (audio_array, sample_rate).
    """
    path = model_path or _DEFAULT_VOICE
    voice = _load_voice(path)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        for chunk in voice.synthesize(text):
            pcm = (chunk.audio_float_array * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

    buf.seek(44)  # skip WAV header
    raw = buf.read()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # Pad trailing silence so the last syllable isn't clipped by the audio buffer
    silence = np.zeros(int(voice.config.sample_rate * 0.35), dtype=np.float32)
    audio = np.concatenate([audio, silence])
    return audio, voice.config.sample_rate


def speak(text: str, model_path: Optional[Path] = None) -> None:
    """Synthesize text and play it through the default audio output. Blocks until done."""
    if not text or not text.strip():
        return
    try:
        audio, sample_rate = synthesize(text, model_path)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        logger.error(f"TTS playback error: {e}")
