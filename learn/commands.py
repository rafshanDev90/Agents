#CLI commands: slash commands are handled locally — 0 tokens, ~0 ms.
'''
The Problem: a raw input()/agent.run() loop can only do one thing with user
input: send it to the model. Managing conversations requires switching,
listing, inspecting — none of which should cost inference time.

The Fix: a tiny dispatcher. Slash input is parsed and executed locally;
everything else goes to the agent loop. Commands mutate AppContext, which
owns the *current* session, so /session open can swap it out from under the
loop.
'''

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import json
import re
import time
import urllib.request

from rich.console import Console
from rich.table import Table

from session import agentSession
from session_store import SessionStore, canonicalize, estimate_entry_tokens, estimate_tokens

# Matches the running llama-server (-c 4096 per `ollama ps`); used only as a
# fallback when the live value can't be read from /api/ps.
DEFAULT_CONTEXT_BUDGET = 4096
OLLAMA_PS_URL = "http://localhost:11434/api/ps"


@dataclass
class AppContext:
    """Mutable state shared between the main loop and command handlers."""

    store: SessionStore
    session: agentSession
    console: Console
    agent: Any = None          # SeniorAgent — gives /timing and /tokens the live data
    should_exit: bool = False

    def save_session(self) -> bool:
        """Persist the active session — never create rows for empty drafts."""
        if len(self.session.history) > 1:   # more than the system prompt
            self.store.save(self.session)
            return True
        return False


class CommandDispatcher:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._handlers: Dict[str, Callable[[str], None]] = {
            "/help": self._help,
            "/history": self._history,
            "/timing": self._timing,
            "/tokens": self._tokens,
            "/session": self._session,
            "/exit": self._exit,
            "/quit": self._exit,
        }

    def dispatch(self, raw: str) -> bool:
        """Execute a slash command. Returns True when the app should exit."""
        parts = raw.strip().split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        handler = self._handlers.get(command)
        if handler is None:
            self.ctx.console.print(f"[red]Unknown command:[/red] {command}   (try /help)")
            return self.ctx.should_exit

        handler(argument)
        return self.ctx.should_exit

    # ------------------------------------------------------------------ /help

    def _help(self, _argument: str) -> None:
        table = Table(title="Commands", header_style="bold blue", show_header=False)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("What it does")
        rows = [
            ("/help", "show this list"),
            ("/session", "list saved sessions (id, title, size)"),
            ("/session new [title]", "create a new session and switch to it"),
            ("/session open <id>", "resume a session (unique prefix is enough)"),
            ("/session search <words>", "full-text search across every session"),
            ("/session delete <id>", "permanently delete a session (asks to confirm)"),
            ("/history", "map of the current conversation — no model call"),
            ("/timing", "where the last turn's time went (rounds, tools, save)"),
            ("/tokens", "current prompt size vs the model context budget"),
            ("/exit", "save and quit"),
        ]
        for command, description in rows:
            table.add_row(command, description)
        self.ctx.console.print(table)

    # --------------------------------------------------------------- /session

    def _session(self, argument: str) -> None:
        parts = argument.split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if subcommand in ("list", "ls"):
            self._session_list()
        elif subcommand == "new":
            self._session_new(rest)
        elif subcommand in ("open", "switch", "resume"):
            self._session_open(rest)
        elif subcommand in ("search", "find", "query"):
            self._session_search(rest)
        elif subcommand in ("delete", "rm"):
            self._session_delete(rest)
        else:
            self.ctx.console.print(
                f"[red]Unknown subcommand:[/red] {subcommand}"
                "  (new | open | search | delete | list)"
            )

    def _session_list(self) -> None:
        table = Table(title="Saved sessions", header_style="bold blue")
        table.add_column("", style="green", no_wrap=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title")
        table.add_column("Updated", no_wrap=True)
        table.add_column("Turns", justify="right")
        table.add_column("Msgs", justify="right")
        table.add_column("~tok", justify="right")

        for record in self.ctx.store.list_sessions():
            active = "→" if record["id"] == self.ctx.session.session_id else ""
            updated = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(record["updated_at"])
            )
            table.add_row(
                active,
                record["id"],
                record["title"] or "(untitled)",
                updated,
                str(record["turn_count"]),
                str(record["message_count"]),
                f"{record['token_estimate']:,}",
            )
        self.ctx.console.print(table)

    def _session_new(self, title: str) -> None:
        self.ctx.save_session()
        self.ctx.session = self.ctx.store.create(title)
        suffix = f" — [italic]{title}[/italic]" if title else ""
        self.ctx.console.print(
            f"Switched to new session [cyan]{self.ctx.session.session_id}[/cyan]{suffix}"
        )

    def _session_open(self, identifier: str) -> None:
        matches = self._resolve_or_complain(identifier, "open")
        if matches is None:
            return
        self.ctx.save_session()
        self.ctx.session = self.ctx.store.load(matches[0])
        self.ctx.console.print(
            f"Resumed [cyan]{self.ctx.session.session_id}[/cyan]"
            f" — {len(self.ctx.session.history)} messages"
        )

    def _session_delete(self, identifier: str) -> None:
        matches = self._resolve_or_complain(identifier, "delete")
        if matches is None:
            return
        target = matches[0]
        answer = input(f"Delete '{target}' permanently? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            self.ctx.console.print("Cancelled.")
            return
        self.ctx.store.delete(target)
        if target == self.ctx.session.session_id:
            self.ctx.session = self.ctx.store.create("")
            self.ctx.console.print(
                f"Deleted [cyan]{target}[/cyan]. Active session reset to"
                f" [cyan]{self.ctx.session.session_id}[/cyan]."
            )
        else:
            self.ctx.console.print(f"Deleted [cyan]{target}[/cyan].")

    def _session_search(self, query: str) -> None:
        if not query:
            self.ctx.console.print("[red]Usage:[/red] /session search <words>")
            return
        hits = self.ctx.store.search(query)
        if not hits:
            self.ctx.console.print(f"[yellow]No matches[/yellow] for '{query}'.")
            return

        table = Table(title=f"Search — '{query}'", header_style="bold blue")
        table.add_column("", style="green", no_wrap=True)
        table.add_column("Session", style="cyan", no_wrap=True)
        table.add_column("Msg", justify="right")
        table.add_column("Role", no_wrap=True)
        table.add_column("Match")
        for hit in hits:
            active = "→" if hit["session_id"] == self.ctx.session.session_id else ""
            table.add_row(
                active,
                hit["session_id"],
                str(hit["seq"]),
                hit["role"],
                self._snippet(hit["content"], query),
            )
        self.ctx.console.print(table)
        self.ctx.console.print(
            f"[dim]{len(hits)} match(es) — /session open <id> to resume one[/dim]"
        )

    @staticmethod
    def _snippet(content: str, query: str, width: int = 90) -> str:
        """Window of text around the first matching term, terms bolded."""
        flat = " ".join((content or "").split())
        terms = [term for term in re.findall(r"\w+", query) if term]
        if not terms:
            return flat[:width] + ("…" if len(flat) > width else "")
        pattern = re.compile(
            "(" + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)) + ")",
            re.IGNORECASE,
        )
        match = pattern.search(flat)
        if match is None:
            text = flat[:width] + ("…" if len(flat) > width else "")
        else:
            start = max(0, match.start() - width // 3)
            text = (
                ("…" if start else "")
                + flat[start : start + width]
                + ("…" if len(flat) > start + width else "")
            )
        return pattern.sub(r"[bold]\1[/bold]", text)

    def _resolve_or_complain(self, identifier: str, verb: str) -> List[str] | None:
        if not identifier:
            self.ctx.console.print(f"[red]Usage:[/red] /session {verb} <id>")
            return None
        matches = self.ctx.store.resolve(identifier)
        if not matches:
            self.ctx.console.print(
                f"[red]No session matches[/red] '{identifier}' — try /session to list them"
            )
            return None
        if len(matches) > 1:
            listed = ", ".join(f"[cyan]{m}[/cyan]" for m in matches[:6])
            more = f" (+{len(matches) - 6} more)" if len(matches) > 6 else ""
            self.ctx.console.print(
                f"'{identifier}' is ambiguous: {listed}{more} — give more characters"
            )
            return None
        return matches

    # ---------------------------------------------------------------- /history

    def _history(self, _argument: str) -> None:
        session = self.ctx.session
        table = Table(
            title=f"Conversation — {session.session_id}", header_style="bold blue"
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Role", no_wrap=True)
        table.add_column("~tok", justify="right")
        table.add_column("Content")

        total = 0
        for index, message in enumerate(session.history):
            entry = canonicalize(message)
            tokens = estimate_entry_tokens(entry)
            total += tokens
            table.add_row(
                str(index),
                entry["role"],
                str(tokens),
                self._preview(entry),
            )
        self.ctx.console.print(table)
        self.ctx.console.print(
            f"[dim]{len(session.history)} messages, ~{total:,} prompt tokens[/dim]"
        )

    @staticmethod
    def _preview(entry: Dict[str, Any]) -> str:
        if entry.get("tool_calls"):
            names = ", ".join(
                (call.get("function") or {}).get("name") or "?" for call in entry["tool_calls"]
            )
            return f"[yellow]→ tool call: {names}[/yellow]"
        if entry["role"] == "tool":
            flat = " ".join((entry["content"] or "").split())
            return f"[dim]← result: {flat[:60]}{'…' if len(flat) > 60 else ''}[/dim]"
        flat = " ".join((entry["content"] or "").split())
        return flat[:70] + ("…" if len(flat) > 70 else "")

    # ----------------------------------------------------------------- /timing

    def _timing(self, _argument: str) -> None:
        trace = getattr(self.ctx.agent, "last_trace", None) if self.ctx.agent else None
        if trace is None:
            self.ctx.console.print(
                "[yellow]No model turn has run in this process yet.[/yellow]"
                " Send a message first, then /timing."
            )
            return

        rounds = Table(title="Last turn — LLM rounds", header_style="bold blue")
        rounds.add_column("Round", justify="right")
        rounds.add_column("Duration", justify="right")
        rounds.add_column("In tok", justify="right")
        rounds.add_column("Out tok", justify="right")
        rounds.add_column("tok/s", justify="right")
        for index, (duration, prompt, completion) in enumerate(trace.rounds, start=1):
            rate = f"{completion / duration:.1f}" if completion and duration else "—"
            rounds.add_row(
                str(index),
                f"{duration:.2f}s",
                str(prompt) if prompt is not None else "?",
                str(completion) if completion is not None else "?",
                rate,
            )
        self.ctx.console.print(rounds)

        if trace.tools:
            tools = Table(title="Tools", header_style="bold blue")
            tools.add_column("Tool", style="cyan")
            tools.add_column("Duration", justify="right")
            for name, duration in trace.tools:
                tools.add_row(name, f"{duration:.2f}s")
            self.ctx.console.print(tools)

        other = max(0.0, trace.total_seconds - trace.llm_seconds - trace.tool_seconds)
        save = f"{trace.save_seconds:.3f}s" if trace.save_seconds is not None else "—"
        self.ctx.console.print(
            f"[dim]llm {trace.llm_seconds:.2f}s | tools {trace.tool_seconds:.2f}s"
            f" | save {save} | other {other:.2f}s | turn total {trace.total_seconds:.2f}s[/dim]"
        )

    # ----------------------------------------------------------------- /tokens

    def _tokens(self, _argument: str) -> None:
        session = self.ctx.session
        entries = [canonicalize(message) for message in session.history]
        history_tokens = sum(estimate_entry_tokens(entry) for entry in entries)

        # Tool schemas are part of every prompt the agent loop sends
        schema_tokens = 0
        if self.ctx.agent is not None:
            schemas = getattr(self.ctx.agent, "tool_schemas", None)
            if schemas:
                schema_tokens = estimate_tokens(json.dumps(schemas))

        budget = self._context_budget()
        total = history_tokens + schema_tokens
        percent = 100 * total / budget if budget else 0.0
        filled = min(30, round(30 * total / budget)) if budget else 0
        bar = "█" * filled + "░" * (30 - filled)
        style = "green" if percent < 60 else ("yellow" if percent < 90 else "red")

        console = self.ctx.console
        console.print(
            f"prompt : [{style}]{bar}[/{style}] {total:,} / {budget:,} tok ({percent:.0f}%)"
        )
        console.print(f"  history ({len(entries)} messages)     : {history_tokens:,} tok")
        console.print(f"  tool schemas (sent every call) : {schema_tokens:,} tok")

        heaviest = sorted(enumerate(entries), key=lambda pair: estimate_entry_tokens(pair[1]), reverse=True)[:3]
        if heaviest:
            console.print("heaviest messages:")
            for index, entry in heaviest:
                console.print(
                    f"  #{index} {entry['role']} ~{estimate_entry_tokens(entry):,} tok"
                    f"  {self._preview(entry)}"
                )

        if total > budget:
            console.print(
                "[red]OVER BUDGET[/red] — llama-server silently drops the oldest"
                " tokens (--context-shift); trim history (L6 sliding window)."
            )
        elif percent >= 80:
            console.print(
                "[yellow]getting full[/yellow] — at 100% the oldest context gets"
                " shifted out; plan the sliding window (L6)."
            )

    @staticmethod
    def _context_budget() -> int:
        """Live context length from Ollama when the model is loaded; else fallback."""
        try:
            with urllib.request.urlopen(OLLAMA_PS_URL, timeout=2) as response:
                models = json.loads(response.read()).get("models", [])
            for model in models:
                length = model.get("context_length") or (
                    model.get("details") or {}
                ).get("context_length")
                if length:
                    return int(length)
        except Exception:  # server down, field absent — budgeting must never crash
            pass
        return DEFAULT_CONTEXT_BUDGET

    # ------------------------------------------------------------------- /exit

    def _exit(self, _argument: str) -> None:
        self.ctx.should_exit = True

    @property
    def command_names(self) -> List[str]:
        return sorted(self._handlers)


class CommandCompleter:
    """Tab completion for slash commands and session ids.

    Completes the first word against the command list, `/session` against its
    subcommands, and `/session open|delete` against real session ids.
    """

    SUBCOMMANDS = ["new", "open", "search", "delete", "list"]
    RESUME_SUBCOMMANDS = {"open", "switch", "resume", "delete", "rm"}

    def __init__(self, ctx: AppContext, commands: List[str]):
        self.ctx = ctx
        self.commands = commands
        self._options: List[str] = []

    def complete(self, text: str, state: int):
        """readline callback — must never raise (a crash kills the prompt)."""
        if state == 0:
            try:
                import readline

                line = readline.get_line_buffer()
            except Exception:
                line = text
            try:
                self._options = self._compute_matches(line, text)
            except Exception:
                self._options = []
        try:
            return self._options[state]
        except IndexError:
            return None

    def _compute_matches(self, line: str, text: str) -> List[str]:
        if text.startswith("/"):  # completing the command word itself
            return [name + " " for name in self.commands if name.startswith(text)]

        words = line.split()
        if not words:
            return []

        if words[0] == "/session":
            # completing the subcommand word itself ("/session " or "/session op")
            if len(words) == 1 or (len(words) == 2 and words[1] == text):
                return [sub + " " for sub in self.SUBCOMMANDS if sub.startswith(text)]
            if words[1] in self.RESUME_SUBCOMMANDS and len(words) <= 3:
                session_ids = [record["id"] for record in self.ctx.store.list_sessions()]
                return [sid + " " for sid in session_ids if sid.startswith(text)]
        return []


def install_readline(completer: CommandCompleter) -> bool:
    """Wire tab completion + line editing (arrow-key history) into input()."""
    try:
        import readline
    except ImportError:  # e.g. minimal builds — CLI still works, just no Tab
        return False
    readline.set_completer(completer.complete)
    readline.set_completer_delims(" \t\n")  # keep '/' inside the completed word
    try:
        readline.parse_and_bind("tab: complete")
    except Exception:
        return False
    return True
