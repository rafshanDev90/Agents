"""Latency diagnosis: split a chat round into model load / prefill / decode.

Ollama's OpenAI-compatible endpoint only reports total latency, but its native
/api/chat endpoint reports per-request durations:

    load_duration         model load (cold start — paid when the model was idle)
    prompt_eval_duration  prefill  — processing the prompt (prompt_eval_count tokens)
    eval_duration         decode   — generating the reply (eval_count tokens)

The script unloads the model first (Ollama would unload it after 5 idle minutes
anyway) to reproduce the real "first message after a pause" path, then replays
the saved session history four times:

    probe 1  cold : full history + new user message (exactly what main.py sends)
    probe 2  warm : identical request — is prompt-prefix caching active?
    probe 3  warm : same history, different user tail — the agent-loop pattern
                    (append-only history: same prefix, new suffix)
    probe 4  warm : probe 2 with stored `reasoning` fields stripped — measures
                    whether re-sending thinking tokens costs anything

Usage:
    uv run python learn/diagnose.py [session_id] [decode_tokens] [--warm]
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"
MODEL = "qwen3.5:4b"
SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"

TAIL_A = {"role": "user", "content": "Write a one-sentence greeting to a new team member."}
TAIL_B = {"role": "user", "content": "Suggest one way to improve a daily standup meeting."}


# ---------------------------------------------------------------- HTTP helpers

def _request(method: str, path: str, payload: dict | None = None, timeout: int = 600):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{OLLAMA}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def resident_models() -> list[str]:
    return [model.get("name", "?") for model in _request("GET", "/api/ps").get("models", [])]


def unload_model() -> bool:
    try:
        _request("DELETE", "/api/generate", {"model": MODEL})
        return True
    except (urllib.error.URLError, OSError) as error:
        print(f"could not unload model: {error}")
        return False


# ------------------------------------------------------------------- history

def load_history(session_id: str) -> list[dict]:
    path = SESSIONS_DIR / f"{session_id}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["history"]


def to_native(history: list[dict], keep_reasoning: bool, tail: dict) -> list[dict]:
    """Convert saved session messages to Ollama native messages + a new user turn."""
    messages = []
    for message in history:
        role = message.get("role")
        entry: dict = {"role": role, "content": message.get("content") or ""}

        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function", {})
                arguments = function.get("arguments") or "{}"
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {"_raw": arguments}
                calls.append(
                    {
                        "id": call.get("id"),
                        "type": "function",
                        "function": {"name": function.get("name"), "arguments": arguments},
                    }
                )
            entry["tool_calls"] = calls

        if role == "tool":
            if message.get("tool_call_id"):
                entry["tool_call_id"] = message["tool_call_id"]
            entry["content"] = str(message.get("content") or "")

        if keep_reasoning and message.get("reasoning"):
            entry["reasoning"] = message["reasoning"]

        messages.append(entry)

    messages.append(dict(tail))
    return messages


# -------------------------------------------------------------------- probes

def probe(label: str, messages: list[dict], decode_tokens: int) -> dict:
    result = {"label": label}
    started = time.perf_counter()
    try:
        data = _request(
            "POST",
            "/api/chat",
            {
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": decode_tokens},
            },
        )
    except urllib.error.HTTPError as error:
        result["error"] = f"HTTP {error.code}: {error.read().decode()[:400]}"
        return result
    result.update(
        wall_s=time.perf_counter() - started,
        load_s=data.get("load_duration", 0) / 1e9,
        prefill_s=data.get("prompt_eval_duration", 0) / 1e9,
        prefill_tokens=data.get("prompt_eval_count", 0),
        cached_tokens=(data.get("prompt_eval_details") or {}).get("cached_tokens", 0),
        details=data.get("prompt_eval_details"),
        decode_s=data.get("eval_duration", 0) / 1e9,
        decode_tokens=data.get("eval_count", 0),
        done_reason=data.get("done_reason", "?"),
    )
    return result


def rate(count: int, seconds: float) -> str:
    return f"{count / seconds:.0f} tok/s" if seconds > 0 else "-"


# -------------------------------------------------------------------- report

def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--warm"]
    warm_only = "--warm" in sys.argv
    session_id = args[0] if args else "user_dev_session_01"
    decode_tokens = int(args[1]) if len(args) > 1 else 32

    history = load_history(session_id)
    tool_rounds = sum(1 for m in history if m.get("role") == "tool")

    print(f"model: {MODEL}   session: {session_id}")
    print(f"history: {len(history)} messages, {tool_rounds} tool results")

    if not warm_only:
        print("unloading model to reproduce the cold path (Ollama does this after 5 idle min)...")
        unload_model()
    print(f"resident before probes: {resident_models() or 'no (cold)'}\n")

    faithful = to_native(history, keep_reasoning=True, tail=TAIL_A)

    results = [
        probe("1 cold  (history + user turn)", faithful, decode_tokens),
        probe("2 warm  (identical replay)", faithful, decode_tokens),
        probe("3 warm  (new user tail)", to_native(history, True, TAIL_B), decode_tokens),
        probe("4 warm  (reasoning stripped)", to_native(history, False, TAIL_A), decode_tokens),
    ]

    if results and "error" in results[0]:
        print(f"replay rejected: {results[0]['error']}")
        return

    header = f"{'probe':<32} {'wall':>7} {'load':>7} {'prefill':>26} {'decode':>24}"
    print(header)
    for result in results:
        if "error" in result:
            print(f"{result['label']:<32} ERROR {result['error']}")
            continue
        cache = f", cached {result['cached_tokens']}" if result["cached_tokens"] else ""
        print(
            f"{result['label']:<32}"
            f" {result['wall_s']:6.1f}s"
            f" {result['load_s']:6.2f}s"
            f" {result['prefill_tokens']:>5} tok {result['prefill_s']:6.2f}s"
            f" ({rate(result['prefill_tokens'], result['prefill_s'])}{cache})"
            f" {result['decode_tokens']:>4} tok {result['decode_s']:5.2f}s"
            f" ({rate(result['decode_tokens'], result['decode_s'])})"
            f" [{result['done_reason']}]"
        )
    if results[0].get("details") is not None:
        print(f"prompt_eval_details raw: {results[0]['details']}")
    print(f"resident after probes: {resident_models() or 'no'}")

    ok = [r for r in results if "error" not in r]
    if not ok:
        return

    print("\nverdict")
    first = ok[0]
    if first["load_s"] > 0.5:
        print(
            f"  * cold start        : {first['load_s']:.2f}s model load +"
            f" {first['prefill_s']:.2f}s first prefill = {first['wall_s']:.1f}s wall"
            f" before any token, every time the model was idle >5 min"
        )

    if first["decode_s"] > 0:
        decode_rate = first["decode_tokens"] / first["decode_s"]
        sample = 150
        print(
            f"  * decode            : {decode_rate:.1f} tok/s (includes thinking)"
            f" -> a {sample}-token reply alone needs ~{sample / decode_rate:.0f}s"
        )

    warm = [r for r in ok[1:] if r["prefill_tokens"]]
    if warm:
        hit = warm[0]
        if hit["cached_tokens"]:
            state = f"ACTIVE — {hit['cached_tokens']} of {hit['prefill_tokens']} tokens reused"
        elif hit["prefill_s"] < first["prefill_s"] / 5 and first["prefill_s"] > 1:
            state = (
                f"likely active — repeat prefill {first['prefill_s']:.2f}s ->"
                f" {hit['prefill_s']:.2f}s (cached_tokens field not reported: {hit['details']})"
            )
        else:
            state = f"NOT active — repeat prefill {hit['prefill_s']:.2f}s for {hit['prefill_tokens']} tokens"
        print(f"  * prefix cache      : {state}")

    if len(ok) > 3:
        with_reasoning = ok[1]["prefill_tokens"]
        without = ok[3]["prefill_tokens"]
        print(
            f"  * reasoning re-send : {with_reasoning - without} tokens"
            f" ({'Ollama ignores the reasoning field in requests' if without == with_reasoning else 're-sent every round'})"
        )


if __name__ == "__main__":
    main()
