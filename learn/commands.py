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

import time

from rich.console import Console
from rich.table import Table

from session import agentSession
from session_store import SessionStore, canonicalize, estimate_entry_tokens


@dataclass
class AppContext:
    """Mutable state shared between the main loop and command handlers."""

    store: SessionStore
    session: agentSession
    console: Console
    should_exit: bool = False


class CommandDispatcher:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._handlers: Dict[str, Callable[[str], None]] = {
            "/help": self._help,
            "/history": self._history,
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
            ("/session delete <id>", "permanently delete a session (asks to confirm)"),
            ("/history", "map of the current conversation — no model call"),
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
        elif subcommand in ("delete", "rm"):
            self._session_delete(rest)
        else:
            self.ctx.console.print(
                f"[red]Unknown subcommand:[/red] {subcommand}"
                "  (new | open | delete | list)"
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
        self.ctx.store.save(self.ctx.session)
        self.ctx.session = self.ctx.store.create(title)
        suffix = f" — [italic]{title}[/italic]" if title else ""
        self.ctx.console.print(
            f"Switched to new session [cyan]{self.ctx.session.session_id}[/cyan]{suffix}"
        )

    def _session_open(self, identifier: str) -> None:
        matches = self._resolve_or_complain(identifier, "open")
        if matches is None:
            return
        self.ctx.store.save(self.ctx.session)
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

    # ------------------------------------------------------------------- /exit

    def _exit(self, _argument: str) -> None:
        self.ctx.should_exit = True
