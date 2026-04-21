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

import os
import platform
import site
import threading
from typing import Optional

import numpy as np
import pyaudio
from loguru import logger
from pynput import keyboard

_DEFAULT_MODEL = "small.en"
_DEFAULT_PTT   = "caps_lock"     # change via listen_once(ptt_key=...) or voice_demo --ptt
_SAMPLE_RATE   = 16000           # Whisper native
_CHANNELS      = 1
_CHUNK         = 1024

# Primes Whisper's decoder with F/A-18C vocabulary so it stops mangling
# designations, NATO phonetics, and avionics abbreviations.
_INITIAL_PROMPT = (
    "F/A-18C Hornet cockpit query. "
    "AIM-120 AMRAAM, AIM-9X Sidewinder, AIM-7 Sparrow, "
    "AGM-65 Maverick, AGM-88 HARM, AGM-84 Harpoon, "
    "GBU-12 Paveway, GBU-38 JDAM, JSOW, MK-82, "
    "TGP, ATFLIR, LITENING, JHMCS, HMD, "
    "DDI, UFC, AMPCD, HUD, IFEI, SMS, "
    "TDC, SCS, HOTAS, master arm, laser arm, seeker, FOV, cage, uncage, "
    "radar, RWS, TWS, STT, ACM, BVR, "
    "TACAN, ILS, ICLS, ACLS, Case III, "
    "A/G, A/A, NAV, CCIP, CCRP, "
    "Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, Hotel, "
    "India, Juliet, Kilo, Lima, Mike, November, Oscar, Papa, "
    "Quebec, Romeo, Sierra, Tango, Uniform, Victor, Whiskey, X-ray, Yankee, Zulu."
)

_model      = None
_model_lock = threading.Lock()
_force_cpu  = False
_cuda_path_ready = False

_pa      = None
_pa_lock = threading.Lock()


def _get_pa() -> pyaudio.PyAudio:
    global _pa
    with _pa_lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
    return _pa


def _ensure_windows_cuda_runtime_paths() -> None:
    """Expose CUDA runtime DLL paths from pip-installed NVIDIA packages."""
    global _cuda_path_ready
    if _cuda_path_ready or platform.system() != "Windows":
        return

    candidate_dirs: list[str] = []
    for base in site.getsitepackages():
        cublas_bin = os.path.join(base, "nvidia", "cublas", "bin")
        cudnn_bin = os.path.join(base, "nvidia", "cudnn", "bin")
        for d in (cublas_bin, cudnn_bin):
            if os.path.isdir(d):
                candidate_dirs.append(d)

    if not candidate_dirs:
        _cuda_path_ready = True
        return

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    for d in candidate_dirs:
        if d not in path_parts:
            path_parts.insert(0, d)
        try:
            os.add_dll_directory(d)
        except Exception:
            pass

    os.environ["PATH"] = os.pathsep.join(path_parts)
    logger.info("Added NVIDIA runtime DLL paths from Python packages")
    _cuda_path_ready = True


def _get_model(model_name: str = _DEFAULT_MODEL):
    global _model, _force_cpu
    with _model_lock:
        if _model is None:
            _ensure_windows_cuda_runtime_paths()
            from faster_whisper import WhisperModel
            if _force_cpu:
                logger.info(f"Loading Whisper '{model_name}' on CPU/int8 ...")
                _model = WhisperModel(model_name, device="cpu", compute_type="int8")
            else:
                try:
                    logger.info(f"Loading Whisper '{model_name}' on CUDA ...")
                    _model = WhisperModel(model_name, device="cuda", compute_type="float16")
                except Exception as e:
                    logger.warning(f"CUDA unavailable ({e}), falling back to CPU/int8")
                    _force_cpu = True
                    _model = WhisperModel(model_name, device="cpu", compute_type="int8")
            logger.info("Whisper ready.")
    return _model


def _reset_model_to_cpu() -> None:
    global _model, _force_cpu
    with _model_lock:
        _model = None
        _force_cpu = True


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
    ptt_key:       str            = _DEFAULT_PTT,
    device:        Optional[int]  = None,
    model_name:    str            = _DEFAULT_MODEL,
    min_duration:  float          = 0.3,
    context_terms: list[str]      | None = None,
    start_pressed: bool           = False,
) -> str:
    """
    Block until PTT held → audio recorded → released → transcribed.

    context_terms: optional list of domain-specific words (e.g. a procedure's
    terminology array) appended to the Whisper initial_prompt so the decoder is
    biased towards them for this utterance.

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

    if start_pressed:
        pressed.set()

    logger.debug(f"Waiting for PTT [{ptt_key}] ...")
    while not pressed.wait(timeout=0.1):
        pass   # spin in short intervals so KeyboardInterrupt can land
    logger.debug("Recording ...")

    # Record while key is held
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

    # Build context-aware prompt: base vocabulary + any procedure-specific terms.
    prompt = _INITIAL_PROMPT
    if context_terms:
        # Strip definition text (e.g. "SMS: Stores Management System" → "SMS")
        # and append as a comma-separated hint list.
        bare = [t.split(":")[0].strip() for t in context_terms]
        prompt = prompt.rstrip(".") + ", " + ", ".join(bare) + "."

    try:
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("cublas", "cudnn", "cuda", "libcudart")):
            logger.warning(f"Whisper CUDA runtime failed ({e}); retrying on CPU/int8")
            _reset_model_to_cpu()
            model = _get_model(model_name)
            segments, info = model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=prompt,
            )
        else:
            raise
    text = " ".join(s.text.strip() for s in segments).strip()
    from voice.corrections import correct
    corrected = correct(text)
    if corrected != text:
        logger.info(f"STT [{duration_s:.1f}s]: {text!r}  →  {corrected!r}")
    else:
        logger.info(f"STT [{duration_s:.1f}s]: {text!r}")
    return corrected
