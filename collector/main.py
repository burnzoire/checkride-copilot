"""
Entry point for the DCS state collector.

Starts the UDP listener (port 7778) in a background thread and serves
the HTTP API (port 7779) in the foreground via uvicorn.

Usage:
    python -m collector.main
    # or
    python collector/main.py
"""

import uvicorn
from loguru import logger

from collector import udp_server
from collector.http_api import app

UDP_PORT = 7778
HTTP_PORT = 7779


def main() -> None:
    logger.info("Starting DCS Copilot collector...")
    stop = udp_server.start_background(port=UDP_PORT)

    try:
        uvicorn.run(app, host="127.0.0.1", port=HTTP_PORT, log_level="warning")
    finally:
        stop.set()
        logger.info("Collector shut down.")


if __name__ == "__main__":
    main()
