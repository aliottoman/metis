"""Start the API and web app together for a local production run.

Runs both as child processes, tags their output ``[api]``/``[web]``, and brings
both down together — on Ctrl-C, or automatically if either one exits on its own.
"""
from __future__ import annotations

import signal
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMMANDS = {
    "api": [str(ROOT / ".venv" / "bin" / "waqil-api")],
    "web": ["pnpm", "--dir", str(ROOT / "apps" / "web"), "start"],
}


def relay(name: str, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{name}] {line}", end="")


def main() -> int:
    procs = {
        name: subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for name, command in COMMANDS.items()
    }
    threads = [
        threading.Thread(target=relay, args=(name, proc), daemon=True)
        for name, proc in procs.items()
    ]
    for thread in threads:
        thread.start()

    stopping = threading.Event()

    def shutdown(*_args: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        print("\nStopping...")
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while not stopping.is_set():
        for name, proc in procs.items():
            if proc.poll() is not None:
                print(f"\n[{name}] exited ({proc.returncode}) — stopping the other process.")
                shutdown()
                break
        else:
            for thread in threads:
                thread.join(timeout=0.5)
            continue
        break

    return max((proc.returncode or 0) for proc in procs.values())


if __name__ == "__main__":
    raise SystemExit(main())
