"""
UDP server that receives JSON export packets from DCS Export.lua on localhost:7778
and writes normalized LiveState to the shared state_store.
"""

import json
import socket
import threading
from loguru import logger

from collector import normalizer, state_store

UDP_HOST = "127.0.0.1"
UDP_PORT = 7778
BUFFER_SIZE = 65535


def _handle_packet(raw: bytes) -> None:
    try:
        packet = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Malformed UDP packet ({len(raw)} bytes): {e}")
        return

    if packet.get("event") != "export":
        return

    try:
        # Extract raw CockpitParams before normalizing (normalizer doesn't pass _raw through)
        cp_block = (packet.get("data") or {}).get("CockpitParams") or {}
        raw_cp = cp_block.pop("_raw", None) if isinstance(cp_block, dict) else None

        state = normalizer.normalize(packet)
        state_store.update(state, raw_cp=raw_cp, raw_packet=packet)
    except Exception as e:
        logger.error(f"Normalizer error: {e}")


def run(host: str = UDP_HOST, port: int = UDP_PORT, stop_event: threading.Event | None = None) -> None:
    """Block-receive DCS export packets until stop_event is set (or KeyboardInterrupt)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)
    logger.info(f"UDP collector listening on {host}:{port}")

    while not (stop_event and stop_event.is_set()):
        try:
            data, _ = sock.recvfrom(BUFFER_SIZE)
            _handle_packet(data)
        except socket.timeout:
            continue
        except OSError:
            break

    sock.close()
    logger.info("UDP collector stopped.")


def start_background(host: str = UDP_HOST, port: int = UDP_PORT) -> threading.Event:
    """Start the UDP server in a daemon thread. Returns the stop_event."""
    stop = threading.Event()
    t = threading.Thread(target=run, args=(host, port, stop), daemon=True, name="udp-collector")
    t.start()
    return stop
