"""
TTS with Kokoro (primary) falling back to Piper.

Kokoro produces significantly more natural speech than Piper.
Voices: bm_george (British male, authoritative) is the default.
Other good options: am_adam (American male), bm_lewis (British male).

speak() is synchronous — blocks until playback completes.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from voice.glossary import apply as _apply_glossary


# ── TTS text normalisation ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Expand aviation abbreviations and units before synthesis."""
    text = _apply_glossary(text)
    # Fractions + unit (must precede bare unit rules)
    text = re.sub(r'3/4\s*nm\b',  'three-quarter nautical mile',  text, flags=re.I)
    text = re.sub(r'1/2\s*nm\b',  'half nautical mile',            text, flags=re.I)
    text = re.sub(r'\b0\.1\s*nm\b', 'point-one nautical miles',    text, flags=re.I)

    # ft/min before ft
    text = re.sub(r'\b(\d[\d,]*)\s*ft/min\b', r'\1 feet per minute', text, flags=re.I)

    # Ranges  X-Y <unit>
    text = re.sub(r'\b(\d[\d,]*)-(\d[\d,]*)\s*ft\b',  r'\1 to \2 feet',           text, flags=re.I)
    text = re.sub(r'\b(\d[\d,]*)-(\d[\d,]*)\s*kts\b', r'\1 to \2 knots',          text, flags=re.I)
    text = re.sub(r'\b(\d[\d,]*)-(\d[\d,]*)\s*%',     r'\1 to \2 percent',        text)
    text = re.sub(r'\b(\d[\d,]*)-(\d[\d,]*)\s*nm\b',  r'\1 to \2 nautical miles', text, flags=re.I)

    # deg C before bare deg
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*deg\s*[Cc]\b', r'\1 degrees Celsius', text)

    # Single units
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*ft\b',  r'\1 feet',           text, flags=re.I)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*kts\b', r'\1 knots',          text, flags=re.I)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*nm\b',  r'\1 nautical miles', text, flags=re.I)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*MHz\b', r'\1 megahertz',      text)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*psi\b', r'\1 PSI',            text, flags=re.I)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*lbs\b', r'\1 pounds',         text, flags=re.I)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*%',     r'\1 percent',        text)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*deg\b', r'\1 degrees',        text, flags=re.I)

    # Acronym pronunciations
    text = re.sub(r'\bHOTAS\b', 'ho-tass', text)

    # Misc
    text = re.sub(r'\bapprox\.\s*', 'approximately ', text, flags=re.I)
    text = re.sub(r'\bapprox\b',    'approximately',  text, flags=re.I)

    return re.sub(r'  +', ' ', text).strip()

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
    normalized = _normalize(text)
    try:
        try:
            audio, sr = _synthesize_kokoro(normalized, voice)
        except Exception as e:
            logger.warning(f"Kokoro failed ({e}), falling back to Piper")
            audio, sr = _synthesize_piper(normalized)
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
