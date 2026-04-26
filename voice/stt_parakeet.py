"""
Parakeet STT backend — NVIDIA NeMo parakeet-tdt-0.6b-v2.

Drop-in replacement for the faster-whisper backend.  Exposes the same
``listen_once()`` entry point used by ``voice/stt.py``.

Pre-requisites (install once, separately from base requirements):
    pip install nemo_toolkit[asr]
    # or the lighter standalone build:
    # pip install nemo_asr  (unofficial but often works)

Model weights (~600 MB) are downloaded automatically on first run via
Hugging Face Hub / NVIDIA NGC and cached in the NeMo model cache dir.

Environment variables:
    PARAKEET_MODEL   override model name (default: nvidia/parakeet-tdt-0.6b-v2)
    PARAKEET_DEVICE  "cuda" | "cpu" (default: cuda if available, else cpu)
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np
import pyaudio
from loguru import logger
from pynput import keyboard

_DEFAULT_MODEL  = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
_DEFAULT_PTT    = "caps_lock"
_SAMPLE_RATE    = 16000   # Parakeet native input rate
_CHANNELS       = 1
_CHUNK          = 1024

_model      = None
_model_lock = threading.Lock()

_pa      = None
_pa_lock = threading.Lock()


def _get_pa() -> pyaudio.PyAudio:
    global _pa
    with _pa_lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
    return _pa


def _get_model(model_name: str = _DEFAULT_MODEL):
    global _model
    with _model_lock:
        if _model is None:
            try:
                import nemo.collections.asr as nemo_asr  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "NeMo toolkit is required for the Parakeet backend.\n"
                    "Install it with:  pip install nemo_toolkit[asr]\n"
                    "Then restart the app."
                ) from exc

            device_pref = os.environ.get("PARAKEET_DEVICE", "").strip().lower()
            if not device_pref:
                try:
                    import torch
                    device_pref = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device_pref = "cpu"

            logger.info(f"Loading Parakeet '{model_name}' on {device_pref.upper()} ...")
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
            if device_pref == "cuda":
                try:
                    asr_model = asr_model.cuda()
                except Exception as e:
                    logger.warning(f"Parakeet CUDA failed ({e}), falling back to CPU")
                    asr_model = asr_model.cpu()
            else:
                asr_model = asr_model.cpu()

            asr_model.eval()
            _model = asr_model
            logger.info("Parakeet ready.")

    return _model


def _parse_key(key_spec: str) -> keyboard.Key | keyboard.KeyCode:
    try:
        return keyboard.Key[key_spec]
    except KeyError:
        pass
    if len(key_spec) == 1:
        return keyboard.KeyCode.from_char(key_spec)
    if key_spec.startswith("vk_"):
        return keyboard.KeyCode.from_vk(int(key_spec[3:]))
    raise ValueError(f"Unknown PTT key: {key_spec!r}")


def _transcribe(audio_float32: np.ndarray) -> str:
    """Run Parakeet inference on a 1-D float32 array at 16 kHz."""
    model = _get_model()

    # NeMo's transcribe() accepts a list of audio arrays (batch) or file paths.
    # We wrap in a list and pass the sample rate explicitly.
    try:
        hypotheses = model.transcribe(
            [audio_float32],
            batch_size=1,
            verbose=False,
        )
        # Result is a list of strings (or Hypothesis objects depending on version)
        raw = hypotheses[0]
        if hasattr(raw, "text"):
            return str(raw.text).strip()
        return str(raw).strip()
    except Exception as e:
        logger.error(f"Parakeet transcription error: {e}")
        return ""


def listen_once(
    ptt_key:       str           = _DEFAULT_PTT,
    device:        Optional[int] = None,
    model_name:    str           = _DEFAULT_MODEL,
    min_duration:  float         = 0.3,
    context_terms: list[str]     | None = None,
    start_pressed: bool          = False,
) -> str:
    """
    Block until PTT held → audio recorded → released → transcribed.

    ``context_terms`` is accepted for API compatibility but is silently ignored —
    Parakeet is a CTC/transducer model and does not support an initial prompt.

    Returns transcript string (may be empty on silence/noise).
    """
    # Eagerly load the model so the first real PTT press is fast.
    _get_model(model_name)

    target  = _parse_key(ptt_key)
    pressed = threading.Event()
    released = threading.Event()

    def on_press(k):
        if k == target:
            pressed.set()

    def on_release(k):
        if k == target:
            released.set()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    if start_pressed:
        pressed.set()

    logger.debug(f"Waiting for PTT [{ptt_key}] ...")
    while not pressed.wait(timeout=0.1):
        pass

    logger.debug("Recording ...")

    p      = _get_pa()
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
    listener.stop()
    listener.join()

    duration_s = len(frames) * _CHUNK / _SAMPLE_RATE
    logger.debug(f"Captured {duration_s:.2f}s")

    if duration_s < min_duration:
        return ""

    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    text = _transcribe(audio)

    from voice.corrections import correct
    corrected = correct(text)
    if corrected != text:
        logger.info(f"STT [{duration_s:.1f}s]: {text!r}  →  {corrected!r}")
    else:
        logger.info(f"STT [{duration_s:.1f}s]: {text!r}")
    return corrected
