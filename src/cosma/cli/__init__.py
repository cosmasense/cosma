"""
CLI infrastructure for Cosma.

Provides shared decorators and utilities for CLI commands.
"""

import functools
import sys
from typing import Callable, Optional

import click

from cosma_client import ServerNotRunningError, CosmaClientError

from .output import OutputFormatter, OutputMode


__all__ = [
    "output_options",
    "handle_client_errors",
    "OutputFormatter",
    "OutputMode",
]


def output_options(func: Callable) -> Callable:
    """
    Decorator that adds --json and --plain flags to a command.

    Injects an OutputFormatter as the 'formatter' keyword argument.
    """
    @click.option(
        "--json",
        "output_json",
        is_flag=True,
        help="Output as JSON",
    )
    @click.option(
        "--plain",
        "output_plain",
        is_flag=True,
        help="Output as plain TSV (for piping)",
    )
    @functools.wraps(func)
    def wrapper(*args, output_json: bool = False, output_plain: bool = False, **kwargs):
        # Determine output mode
        if output_json:
            mode = OutputMode.JSON
        elif output_plain:
            mode = OutputMode.PLAIN
        else:
            mode = OutputMode.HUMAN

        # Create formatter and inject it
        formatter = OutputFormatter(mode=mode)
        kwargs["formatter"] = formatter
        return func(*args, **kwargs)

    return wrapper


def handle_client_errors(func: Callable) -> Callable:
    """
    Decorator that catches client connection errors and prints friendly messages.

    Exits with code 1 on error.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from rich.console import Console

        console = Console(stderr=True)

        try:
            return func(*args, **kwargs)
        except ServerNotRunningError as e:
            console.print()
            console.print("[red bold]Connection failed[/red bold]")
            console.print(f"[red]Could not connect to Cosma server at {e.base_url}[/red]")
            console.print()
            console.print("[dim]Start the server with:[/dim]  cosma serve")
            console.print()
            sys.exit(1)
        except CosmaClientError as e:
            formatter = kwargs.get("formatter")
            if formatter:
                formatter.error(str(e))
            else:
                console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            formatter = kwargs.get("formatter")
            if formatter:
                formatter.error(f"Unexpected error: {e}")
            else:
                console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    return wrapper
