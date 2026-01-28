"""
TUI command for Cosma CLI.
"""

import sys

import click


@click.command("tui")
@click.argument("directory", default=".")
def tui_command(directory: str):
    """Launch the interactive TUI interface."""
    try:
        from cosma_tui import start_tui
    except ImportError:
        click.echo(
            "Error: cosma-tui is not installed.\n"
            "Install it with: pip install cosma[tui]",
            err=True,
        )
        sys.exit(1)

    result = start_tui(directory)
    if result:
        print(result)
