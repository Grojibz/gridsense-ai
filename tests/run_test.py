"""Open the GridSense demo in a browser, with the model warmed up.

    python tests/run_test.py

Checks the API is up, sends one throwaway question so the first *live* answer isn't
competing with model load time, then opens the chat page (question field + send button).

Everything is under ``__main__`` so pytest can import this file harmlessly — the filename
matches pytest's ``*_test.py`` collection pattern.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import webbrowser

BASE_URL = "http://localhost:8000"
WARMUP_QUESTION = "What is thermal runaway?"


def _get(path: str, timeout: int = 10) -> dict | None:
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _ask(question: str, timeout: int = 300) -> dict | None:
    request = urllib.request.Request(
        f"{BASE_URL}/ask",
        data=json.dumps({"question": question}).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        print(f"  ! warm-up failed: {exc}")
        return None


def main() -> int:
    print(f"1/3  API at {BASE_URL} ... ", end="", flush=True)
    if _get("/health") is None:
        print("unreachable")
        print("\n     Start it with:  docker compose up -d")
        return 1
    print("ok")

    print("2/3  warming the model (CPU inference, can take a minute) ... ", end="", flush=True)
    answer = _ask(WARMUP_QUESTION)
    if answer is None:
        print("failed — opening the page anyway")
    else:
        print("ok")
        print(f"     -> {answer['answer'][:70]!r}")
        if answer.get("refused"):
            # Fluent nonsense from a bad GPU surfaces here first: the guardrails refuse
            # rather than serve it, so a refusal on this question is worth noticing.
            print(f"     ! refused ({answer.get('refusal_reason')}) — check OLLAMA_NUM_GPU=0")

    print(f"3/3  opening {BASE_URL}")
    webbrowser.open(BASE_URL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
