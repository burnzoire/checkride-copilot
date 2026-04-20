from __future__ import annotations

import argparse
import ctypes
import threading
import time as _time

from loguru import logger

from orchestrator.llm import call_ollama_with_tools, continue_last_answer
from orchestrator.nav_intercept import NavCommand, check as check_nav
from orchestrator.session import Session
from orchestrator.tools import format_step, load_proc, maneuver_group_end
from voice import stt, tts

DEFAULT_MODEL = "qwen2.5:7b"

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


def run(ptt_key: str, mic_device: int | None, speak: bool, model: str) -> None:
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

    while True:
        try:
            session.begin_turn()
            ctx_terms = None
            if session.active_proc:
                proc = load_proc(session.active_proc)
                if proc:
                    ctx_terms = proc.get("terminology") or None

            transcript = stt.listen_once(ptt_key=ptt_key, device=mic_device, context_terms=ctx_terms)
            if not transcript:
                print("  (nothing heard)")
                continue

            print(f"  You  : {transcript!r}")
            nav = check_nav(transcript, session)
            reply = ""

            if nav == NavCommand.CANCEL:
                session.active_proc = None
                session.active_step = 0
                reply = "Procedure cancelled."
                print("  [PROC] cancelled")

            elif nav == NavCommand.REPEAT:
                proc = load_proc(session.active_proc or "")
                steps = proc.get("steps", []) if proc else []
                if steps:
                    reply = format_step(session.active_proc or "", session.active_step)
                    print(f"  [PROC] repeat step {session.active_step + 1}/{len(steps)}")
                else:
                    reply = "Procedure unavailable."

            elif nav == NavCommand.RESTART:
                session.active_step = 0
                reply = format_step(session.active_proc or "", 0)
                proc = load_proc(session.active_proc or "")
                total = len(proc.get("steps", [])) if proc else "?"
                print(f"  [PROC] restart -> step 1/{total}")

            elif nav == NavCommand.NEXT:
                proc = load_proc(session.active_proc or "")
                steps = proc.get("steps", []) if proc else []
                if not steps:
                    reply = "Procedure unavailable."
                else:
                    group_end = maneuver_group_end(steps, session.active_step)
                    session.active_step = group_end + 1
                    if session.active_step >= len(steps):
                        reply = f"{proc.get('canonical', session.active_proc)} complete."
                        session.active_proc = None
                        session.active_step = 0
                        print("  [PROC] complete")
                    else:
                        reply = format_step(proc.get("key", ""), session.active_step)
                        print(f"  [PROC] step {session.active_step + 1}/{len(steps)}")

            elif nav == NavCommand.CONTINUE_LAST_REPLY:
                reply = continue_last_answer(session, model)

            else:
                reply = call_ollama_with_tools(transcript, session, model)

            print(f"  Reply: {reply}\n")
            now = _time.time()
            session.add_user_turn(transcript, now)
            session.add_assistant_turn(reply, now)

            if speak:
                stop_tts = threading.Event()
                tts_done = threading.Event()

                def _interrupt_watch() -> None:
                    while not tts_done.is_set():
                        if ptt_vk and _key_held(ptt_vk):
                            stop_tts.set()
                            return
                        _time.sleep(0.04)

                threading.Thread(target=_interrupt_watch, daemon=True).start()
                try:
                    tts.speak(reply, stop_event=stop_tts)
                finally:
                    tts_done.set()

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            logger.error(f"Orchestrator loop error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptt", default="caps_lock", help="PTT key (default: caps_lock)")
    parser.add_argument("--mic", type=int, default=None, help="PyAudio input device index")
    parser.add_argument("--no-speak", action="store_true", help="Print reply without audio")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    run(ptt_key=args.ptt, mic_device=args.mic, speak=not args.no_speak, model=args.model)


if __name__ == "__main__":
    main()
