"""
Thread-safe in-memory store for the latest normalized LiveState dict.
Written by the UDP server, read by the HTTP API and diagnostic engine.
"""

import threading
import time
from typing import Optional

_lock = threading.Lock()
_state: Optional[dict] = None
_raw_cp: Optional[dict] = None
_raw_packet: Optional[dict] = None
_last_updated_ms: int = 0


def update(state: dict, raw_cp: Optional[dict] = None, raw_packet: Optional[dict] = None) -> None:
    """Store a new normalized LiveState dict (and optional raw data). Thread-safe."""
    global _state, _raw_cp, _raw_packet, _last_updated_ms
    now_ms = int(time.monotonic() * 1000)
    with _lock:
        _state = {**state, "timestamp_ms": now_ms}
        _last_updated_ms = now_ms
        if raw_cp is not None:
            _raw_cp = raw_cp
        if raw_packet is not None:
            _raw_packet = raw_packet


def get() -> Optional[dict]:
    """Return the latest LiveState dict with a fresh data_age_ms, or None."""
    with _lock:
        if _state is None:
            return None
        now_ms = int(time.monotonic() * 1000)
        return {**_state, "data_age_ms": now_ms - _last_updated_ms}


def get_raw_cp() -> Optional[dict]:
    """Return the latest raw CockpitParams dict, or None."""
    with _lock:
        return dict(_raw_cp) if _raw_cp else None


def get_raw_packet() -> Optional[dict]:
    """Return the last complete raw DCS packet, or None."""
    with _lock:
        return dict(_raw_packet) if _raw_packet else None


def get_age_ms() -> int:
    """Return milliseconds since the last state update, or -1 if no data yet."""
    with _lock:
        if _last_updated_ms == 0:
            return -1
        return int(time.monotonic() * 1000) - _last_updated_ms
