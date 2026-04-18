"""
TTS with Kokoro (primary) falling back to Piper.

Kokoro produces significantly more natural speech than Piper.
Voices: bm_george (British male, authoritative) is the default.
Other good options: am_adam (American male), bm_lewis (British male).

speak() is synchronous — blocks until playback completes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger

# ── Kokoro ───────────────────────────────────────────────────────────────────
_DEFAULT_VOICE = "am_adam"   # American male — clear, authoritative
_kokoro_pipeline = None

def _get_kokoro(voice: str = _DEFAULT_VOICE):
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        logger.info(f"Loading Kokoro voice '{voice}' ...")
        _kokoro_pipeline = KPipeline(lang_code="a")   # 'a' = American English
        logger.info("Kokoro ready.")
    return _kokoro_pipeline


def _synthesize_kokoro(text: str, voice: str = _DEFAULT_VOICE) -> tuple[np.ndarray, int]:
    pipeline = _get_kokoro(voice)
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=0.95):
        if audio is not None:
            chunks.append(audio)
    if not chunks:
        return np.zeros(0, dtype=np.float32), 24000
    audio = np.concatenate(chunks)
    silence = np.zeros(int(24000 * 0.25), dtype=np.float32)
    return np.concatenate([audio, silence]), 24000


# ── Piper fallback ───────────────────────────────────────────────────────────
_PIPER_MODEL = Path(__file__).parent.parent / "models" / "piper" / "en_US-ryan-medium.onnx"
_piper_cache: dict = {}

def _synthesize_piper(text: str) -> tuple[np.ndarray, int]:
    import io, wave
    from piper import PiperVoice
    key = str(_PIPER_MODEL)
    if key not in _piper_cache:
        logger.info(f"Loading Piper fallback voice ...")
        _piper_cache[key] = PiperVoice.load(str(_PIPER_MODEL))
    voice = _piper_cache[key]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        for chunk in voice.synthesize(text):
            pcm = (chunk.audio_float_array * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
    buf.seek(44)
    audio = np.frombuffer(buf.read(), dtype=np.int16).astype(np.float32) / 32768.0
    silence = np.zeros(int(voice.config.sample_rate * 0.35), dtype=np.float32)
    return np.concatenate([audio, silence]), voice.config.sample_rate


# ── Public API ───────────────────────────────────────────────────────────────
def prewarm(voice: str = _DEFAULT_VOICE) -> None:
    """Load the Kokoro pipeline and GPU weights now so the first speak() has no delay."""
    _get_kokoro(voice)


def speak(text: str, voice: str = _DEFAULT_VOICE, stop_event: threading.Event | None = None) -> None:
    """Synthesize text and play through default audio output. Blocks until done or stop_event fires."""
    if not text or not text.strip():
        return
    try:
        try:
            audio, sr = _synthesize_kokoro(text, voice)
        except Exception as e:
            logger.warning(f"Kokoro failed ({e}), falling back to Piper")
            audio, sr = _synthesize_piper(text)
        sd.play(audio, samplerate=sr)
        if stop_event is not None:
            stream = sd.get_stream()
            while stream.active:
                if stop_event.is_set():
                    sd.stop()
                    return
                time.sleep(0.04)
        else:
            sd.wait()
    except Exception as e:
        logger.error(f"TTS playback error: {e}")
