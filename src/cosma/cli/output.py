"""
Output formatting for CLI commands.

Supports three output modes:
- HUMAN: Rich tables with colors (default for TTY)
- PLAIN: Tab-separated values on stdout (for piping)
- JSON: JSON output on stdout (for scripting)
"""

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Sequence

from rich.console import Console
from rich.status import Status
from rich.table import Table


class OutputMode(Enum):
    """Output mode for CLI commands."""
    HUMAN = "human"
    PLAIN = "plain"
    JSON = "json"


@dataclass
class OutputFormatter:
    """
    Handles output formatting for CLI commands.

    Supports human-readable, plain (TSV), and JSON output modes.
    Progress and hints go to stderr in human mode only.
    """

    mode: OutputMode
    _console: Console = field(default=None, repr=False)  # type: ignore
    _stderr_console: Console = field(default=None, repr=False)  # type: ignore

    def __post_init__(self):
        if self._console is None:
            self._console = Console()
        if self._stderr_console is None:
            self._stderr_console = Console(stderr=True)

    @property
    def console(self) -> Console:
        """Get the Rich console for human output."""
        if self._console is None:
            self._console = Console()
        return self._console

    @property
    def stderr_console(self) -> Console:
        """Get the Rich console for stderr output."""
        if self._stderr_console is None:
            self._stderr_console = Console(stderr=True)
        return self._stderr_console

    def is_human(self) -> bool:
        """Check if output mode is human-readable."""
        return self.mode == OutputMode.HUMAN

    @contextmanager
    def status(self, msg: str) -> Generator[None, None, None]:
        """
        Show a loading spinner while executing code (human mode only).

        Usage:
            with formatter.status("Fetching data..."):
                result = client.fetch_data()

        The spinner auto-clears when the context exits.
        """
        if self.mode == OutputMode.HUMAN and sys.stderr.isatty():
            with self.stderr_console.status(f"[dim]{msg}[/dim]", spinner="dots"):
                yield
        else:
            yield

    def progress(self, msg: str) -> None:
        """
        Show progress message (stderr, human mode only).
        Deprecated: prefer using status() context manager for loading states.
        """
        if self.mode == OutputMode.HUMAN:
            self.stderr_console.print(f"[dim]{msg}[/dim]")

    def hint(self, msg: str) -> None:
        """
        Show hint message (stderr, human TTY only).
        """
        if self.mode == OutputMode.HUMAN and sys.stderr.isatty():
            self.stderr_console.print(f"[dim]{msg}[/dim]")

    def error(self, msg: str) -> None:
        """
        Show error message (stderr, all modes).
        """
        Console(stderr=True).print(f"[red]Error:[/red] {msg}")

    def success(self, msg: str) -> None:
        """
        Show success message.

        In human mode: green text
        In plain/JSON mode: just print to stdout
        """
        if self.mode == OutputMode.HUMAN:
            self.console.print(f"[green]{msg}[/green]")
        elif self.mode == OutputMode.PLAIN:
            print(msg)
        # In JSON mode, success messages are typically included in the data

    def print(self, msg: str) -> None:
        """
        Print a message (stdout).

        In human mode: uses Rich formatting
        In plain/JSON mode: plain text
        """
        if self.mode == OutputMode.HUMAN:
            self.console.print(msg)
        else:
            print(msg)

    def output_table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        title: Optional[str] = None,
    ) -> None:
        """
        Output tabular data.

        In human mode: Rich table with colors
        In plain mode: TSV (tab-separated values)
        In JSON mode: List of dicts
        """
        if self.mode == OutputMode.HUMAN:
            table = Table(title=title)
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*[str(cell) for cell in row])
            self.console.print(table)

        elif self.mode == OutputMode.PLAIN:
            # TSV format: headers then rows
            print("\t".join(str(h) for h in headers))
            for row in rows:
                print("\t".join(str(cell) for cell in row))

        else:  # JSON
            data = [dict(zip(headers, row)) for row in rows]
            self.output_json(data)

    def output_dict(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
    ) -> None:
        """
        Output key-value data.

        In human mode: Rich table with Key/Value columns
        In plain mode: TSV with key\tvalue per line
        In JSON mode: JSON object
        """
        if self.mode == OutputMode.HUMAN:
            table = Table(title=title, show_header=False)
            table.add_column("Key", style="bold")
            table.add_column("Value")
            for key, value in data.items():
                table.add_row(key, str(value))
            self.console.print(table)

        elif self.mode == OutputMode.PLAIN:
            for key, value in data.items():
                print(f"{key}\t{value}")

        else:  # JSON
            self.output_json(data)

    def output_list(
        self,
        items: Sequence[Any],
        title: Optional[str] = None,
    ) -> None:
        """
        Output a list of items.

        In human mode: Rich list with bullet points
        In plain mode: One item per line
        In JSON mode: JSON array
        """
        if self.mode == OutputMode.HUMAN:
            if title:
                self.console.print(f"[bold]{title}[/bold]")
            for item in items:
                self.console.print(f"  - {item}")

        elif self.mode == OutputMode.PLAIN:
            for item in items:
                print(str(item))

        else:  # JSON
            self.output_json(list(items))

    def output_json(self, data: Any) -> None:
        """
        Output JSON data to stdout.

        Used directly for JSON mode or can be called explicitly.
        """
        print(json.dumps(data, indent=2 if self.mode == OutputMode.HUMAN else None))
