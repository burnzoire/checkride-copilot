from __future__ import annotations

import argparse
import copy
import ctypes
import os
import sys
import threading
import time as _time
from typing import Callable

import httpx
from loguru import logger
import uvicorn

from collector import udp_server
from collector.http_api import app as collector_app
from orchestrator.llm import call_ollama_with_tools, continue_last_answer
from mission.frequencies import airfield_context_terms, detect_theatre
from orchestrator.nav_intercept import NavCommand, check as check_nav
from orchestrator.session import Session
from orchestrator.tools import format_step, load_proc, maneuver_group_end
from voice import stt, tts

DEFAULT_MODEL = "qwen2.5:7b"
_COLLECTOR_HEALTH_URL = "http://127.0.0.1:7779/health"

_PTT_VK: dict[str, int] = {
    "caps_lock": 0x14,
    "scroll_lock": 0x91,
    "num_lock": 0x90,
    **{f"f{n}": 0x6B + n for n in range(13, 25)},
}


def _ptt_vk(key_str: str) -> int | None:
    k = key_str.lower()
    if k in _PTT_VK:
        return _PTT_VK[k]
    if len(k) == 1:
        return ord(k.upper())
    return None


def _key_held(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def _collector_online() -> bool:
    try:
        r = httpx.get(_COLLECTOR_HEALTH_URL, timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_collector_running() -> Callable[[], None]:
    if _collector_online():
        logger.info("Collector already running.")
        return lambda: None

    logger.info("Collector not reachable on 127.0.0.1:7779 - starting local collector.")
    udp_stop = udp_server.start_background(port=7778)

    config = uvicorn.Config(collector_app, host="127.0.0.1", port=7779, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="collector-http")
    thread.start()

    def _stop() -> None:
        server.should_exit = True
        udp_stop.set()

    # Give it a brief moment so first turn can see fresh state.
    for _ in range(20):
        if _collector_online():
            logger.info("Local collector started.")
            break
        _time.sleep(0.1)
    return _stop


def _build_stt_context_terms(session: Session) -> list[str] | None:
    terms: list[str] = []
    seen: set[str] = set()

    def _add(items: list[str] | None) -> None:
        for item in items or []:
            term = str(item).strip()
            if len(term) < 3:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)

    if session.active_proc:
        proc = load_proc(session.active_proc)
        if proc:
            _add(proc.get("terminology") or [])

    theatre = detect_theatre()
    _add(airfield_context_terms(theatre=theatre, limit=80))

    if session.last_airfield:
        terms.insert(0, session.last_airfield)

    return terms or None


def _resolve_reply(transcript: str, session: Session, model: str) -> tuple[str, Session, list[str]]:
    next_session: Session = copy.deepcopy(session)
    nav = check_nav(transcript, next_session)
    reply = ""
    logs: list[str] = []

    if nav == NavCommand.CANCEL:
        next_session.active_proc = None
        next_session.active_step = 0
        reply = "Procedure cancelled."
        logs.append("  [PROC] cancelled")

    elif nav == NavCommand.REPEAT:
        proc = load_proc(next_session.active_proc or "")
        steps = proc.get("steps", []) if proc else []
        if steps:
            reply = format_step(next_session.active_proc or "", next_session.active_step)
            logs.append(f"  [PROC] repeat step {next_session.active_step + 1}/{len(steps)}")
        else:
            reply = "Procedure unavailable."

    elif nav == NavCommand.RESTART:
        next_session.active_step = 0
        reply = format_step(next_session.active_proc or "", 0)
        proc = load_proc(next_session.active_proc or "")
        total = len(proc.get("steps", [])) if proc else "?"
        logs.append(f"  [PROC] restart -> step 1/{total}")

    elif nav == NavCommand.NEXT:
        proc = load_proc(next_session.active_proc or "")
        steps = proc.get("steps", []) if proc else []
        if not steps:
            reply = "Procedure unavailable."
        else:
            group_end = maneuver_group_end(steps, next_session.active_step)
            next_session.active_step = group_end + 1
            if next_session.active_step >= len(steps):
                reply = f"{proc.get('canonical', next_session.active_proc)} complete."
                next_session.active_proc = None
                next_session.active_step = 0
                logs.append("  [PROC] complete")
            else:
                reply = format_step(proc.get("key", ""), next_session.active_step)
                logs.append(f"  [PROC] step {next_session.active_step + 1}/{len(steps)}")

    elif nav == NavCommand.CONTINUE_LAST_REPLY:
        reply = continue_last_answer(next_session, model)

    else:
        reply = call_ollama_with_tools(transcript, next_session, model)

    return reply, next_session, logs


def run_text(model: str) -> None:
    stop_collector = _ensure_collector_running()

    print("\nCheckride Copilot - Text Mode")
    print(f"Model : {model}")
    print("Type your query and press Enter. Type 'quit' or 'exit' to stop.\n")

    session = Session()

    try:
        while True:
            try:
                session.begin_turn()
                transcript = input("You  : ").strip()
                if not transcript:
                    continue
                if transcript.lower() in {"quit", "exit"}:
                    print("Stopped.")
                    break

                reply, session, logs = _resolve_reply(transcript, session, model)
                for line in logs:
                    print(line)
                print(f"Reply: {reply}\n")

                now = _time.time()
                session.add_user_turn(transcript, now)
                session.add_assistant_turn(reply, now)

            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except EOFError:
                print("\nStopped.")
                break
            except Exception as e:
                logger.error(f"Text mode loop error: {e}")
    finally:
        stop_collector()


def run(ptt_key: str, mic_device: int | None, speak: bool, model: str) -> None:
    stop_collector = _ensure_collector_running()

    backend = os.environ.get("STT_BACKEND", "whisper").strip().lower()
    if backend == "parakeet":
        logger.info("Pre-loading Parakeet ...")
        from voice.stt_parakeet import _get_model as _parakeet_get_model
        _parakeet_get_model()
    else:
        logger.info("Pre-loading Whisper ...")
        stt._get_model()

    if speak:
        logger.info("Pre-loading TTS ...")
        tts.prewarm()

    print("\nCheckride Copilot - Orchestrator")
    print(f"PTT : {ptt_key.upper()}  |  Model : {model}  |  Audio : {'on' if speak else 'off'}")
    print(f"Hold [{ptt_key.upper()}] and speak.  Ctrl-C to quit.\n")

    session = Session()
    ptt_vk = _ptt_vk(ptt_key)
    start_pressed = False

    try:
        while True:
            try:
                session.begin_turn()
                ctx_terms = _build_stt_context_terms(session)

                start_now = start_pressed and bool(ptt_vk and _key_held(ptt_vk))
                transcript = stt.listen_once(
                    ptt_key=ptt_key,
                    device=mic_device,
                    context_terms=ctx_terms,
                    start_pressed=start_now,
                )
                start_pressed = False
                if not transcript:
                    print("  (nothing heard)")
                    continue

                print(f"  You  : {transcript!r}")
                result: dict[str, object] = {}
                reply_done = threading.Event()

                def _reply_worker() -> None:
                    try:
                        reply, next_session, logs = _resolve_reply(transcript, session, model)
                        result["reply"] = reply
                        result["session"] = next_session
                        result["logs"] = logs
                    except Exception as exc:
                        result["error"] = exc
                    finally:
                        reply_done.set()

                threading.Thread(target=_reply_worker, daemon=True).start()

                interrupted = False
                while not reply_done.wait(timeout=0.04):
                    if ptt_vk and _key_held(ptt_vk):
                        interrupted = True
                        start_pressed = True
                        break

                if interrupted:
                    print("  [INTERRUPT] reply cancelled\n")
                    continue

                if "error" in result:
                    raise result["error"]  # type: ignore[misc]

                reply = str(result.get("reply", ""))
                session = result.get("session", session)  # type: ignore[assignment]
                for line in result.get("logs", []):
                    print(line)

                print(f"  Reply: {reply}\n")
                now = _time.time()
                session.add_user_turn(transcript, now)
                session.add_assistant_turn(reply, now)

                if speak:
                    stop_tts = threading.Event()
                    tts_done = threading.Event()
                    tts_interrupted = threading.Event()

                    def _interrupt_watch() -> None:
                        while not tts_done.is_set():
                            if ptt_vk and _key_held(ptt_vk):
                                stop_tts.set()
                                tts_interrupted.set()
                                return
                            _time.sleep(0.04)

                    threading.Thread(target=_interrupt_watch, daemon=True).start()
                    try:
                        tts.speak(reply, stop_event=stop_tts)
                    finally:
                        tts_done.set()
                    if tts_interrupted.is_set():
                        start_pressed = True

            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                logger.error(f"Orchestrator loop error: {e}")
    finally:
        stop_collector()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptt", default="caps_lock", help="PTT key (default: caps_lock)")
    parser.add_argument("--mic", type=int, default=None, help="PyAudio input device index")
    parser.add_argument("--no-speak", action="store_true", help="Print reply without audio")
    parser.add_argument("--text", action="store_true", help="Text-only interactive mode (no STT/TTS)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    if args.text:
        run_text(model=args.model)
        return

    run(ptt_key=args.ptt, mic_device=args.mic, speak=not args.no_speak, model=args.model)


if __name__ == "__main__":
    main()
