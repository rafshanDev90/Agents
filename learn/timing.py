#L10 (partial): Latency tracing
'''
The Problem: "it feels slow" is not a measurement. Every agent turn silently
spends time in three different places — LLM rounds (prefill + decode), tool
executions, and persistence — and only one of them is usually suspected.

The Fix: TurnTrace records wall-clock time per component of a single turn and
renders one compact summary, so optimization targets real numbers.
'''

import time
from typing import Any, List, Optional, Tuple


class TurnTrace:
    """Collects timing data for one agent turn (one user input -> final reply)."""

    def __init__(self):
        self.started = time.perf_counter()
        # (duration_seconds, prompt_tokens, completion_tokens)
        self.rounds: List[Tuple[float, Optional[int], Optional[int]]] = []
        # (tool_name, duration_seconds)
        self.tools: List[Tuple[str, float]] = []
        self.save_seconds: Optional[float] = None

    @staticmethod
    def _tokens(usage: Any) -> Tuple[Optional[int], Optional[int]]:
        if usage is None:
            return None, None
        return (
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

    def record_round(self, seconds: float, usage: Any = None) -> None:
        prompt_tokens, completion_tokens = self._tokens(usage)
        self.rounds.append((seconds, prompt_tokens, completion_tokens))

    def record_tool(self, name: str, seconds: float) -> None:
        self.tools.append((name, seconds))

    def record_save(self, seconds: float) -> None:
        self.save_seconds = seconds

    @property
    def llm_seconds(self) -> float:
        return sum(duration for duration, _, _ in self.rounds)

    @property
    def tool_seconds(self) -> float:
        return sum(duration for _, duration in self.tools)

    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self.started

    def render(self) -> str:
        lines = ["latency trace"]
        for index, (duration, prompt, completion) in enumerate(self.rounds, start=1):
            tokens = f"in {prompt if prompt is not None else '?'} / out {completion if completion is not None else '?'} tok"
            rate = f" | {completion / duration:.1f} tok/s" if completion else ""
            lines.append(f"  round {index}: {duration:6.2f}s  {tokens}{rate}")
        for name, duration in self.tools:
            lines.append(f"  tool {name}: {duration:6.2f}s")
        if self.save_seconds is not None:
            lines.append(f"  save: {self.save_seconds:6.3f}s")
        lines.append(
            f"  total: {self.total_seconds:6.2f}s"
            f"  (llm {self.llm_seconds:.2f}s,"
            f" tools {self.tool_seconds:.2f}s,"
            f" other {self.total_seconds - self.llm_seconds - self.tool_seconds:.2f}s)"
        )
        return "\n".join(lines)
