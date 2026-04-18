"""
Push-to-talk speech-to-text using faster-whisper.

Hold the PTT key → records mic audio.
Release → transcribes with Whisper → returns text.

Usage:
    from voice.stt import listen_once
    text = listen_once()          # uses default PTT key
    text = listen_once("caps_lock")
    text = listen_once("scroll_lock")
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import pyaudio
from loguru import logger
from pynput import keyboard

_DEFAULT_MODEL = "base.en"
_DEFAULT_PTT   = "caps_lock"     # change via listen_once(ptt_key=...) or voice_demo --ptt
_SAMPLE_RATE   = 16000           # Whisper native
_CHANNELS      = 1
_CHUNK         = 1024

_model      = None
_model_lock = threading.Lock()


def _get_model(model_name: str = _DEFAULT_MODEL):
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            try:
                logger.info(f"Loading Whisper '{model_name}' on CUDA ...")
                _model = WhisperModel(model_name, device="cuda", compute_type="float16")
            except Exception as e:
                logger.warning(f"CUDA unavailable ({e}), falling back to CPU/int8")
                _model = WhisperModel(model_name, device="cpu", compute_type="int8")
            logger.info("Whisper ready.")
    return _model


def _parse_key(key_spec: str) -> keyboard.Key | keyboard.KeyCode:
    """'scroll_lock', 'caps_lock', 'ctrl_r', 'f13', or single char."""
    try:
        return keyboard.Key[key_spec]
    except KeyError:
        pass
    if len(key_spec) == 1:
        return keyboard.KeyCode.from_char(key_spec)
    if key_spec.startswith("vk_"):
        return keyboard.KeyCode.from_vk(int(key_spec[3:]))
    raise ValueError(f"Unknown PTT key: {key_spec!r}")


def listen_once(
    ptt_key:       str           = _DEFAULT_PTT,
    device:        Optional[int] = None,
    model_name:    str           = _DEFAULT_MODEL,
    min_duration:  float         = 0.3,
) -> str:
    """
    Block until PTT held → audio recorded → released → transcribed.
    Returns transcript string (may be empty on silence/noise).
    """
    model      = _get_model(model_name)
    target     = _parse_key(ptt_key)
    pressed    = threading.Event()
    released   = threading.Event()

    def on_press(k):
        if k == target:
            pressed.set()

    def on_release(k):
        if k == target:
            released.set()

    # Single persistent listener covers the whole press→release window.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    logger.debug(f"Waiting for PTT [{ptt_key}] ...")
    while not pressed.wait(timeout=0.1):
        pass   # spin in short intervals so KeyboardInterrupt can land
    logger.debug("Recording ...")

    # Record while key is held
    p      = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=_CHANNELS,
        rate=_SAMPLE_RATE,
        input=True,
        input_device_index=device,
        frames_per_buffer=_CHUNK,
    )

    frames: list[bytes] = []
    while not released.is_set():
        frames.append(stream.read(_CHUNK, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    p.terminate()
    listener.stop()

    duration_s = len(frames) * _CHUNK / _SAMPLE_RATE
    logger.debug(f"Captured {duration_s:.2f}s")

    if duration_s < min_duration:
        return ""

    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    from voice.corrections import correct
    corrected = correct(text)
    if corrected != text:
        logger.info(f"STT [{duration_s:.1f}s]: {text!r}  →  {corrected!r}")
    else:
        logger.info(f"STT [{duration_s:.1f}s]: {text!r}")
    return corrected
