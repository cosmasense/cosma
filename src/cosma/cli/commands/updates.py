"""
Updates command for Cosma CLI.
"""

import json
import signal
import sys

import click

from cosma_client import SyncClient, ServerNotRunningError

from .. import OutputFormatter, OutputMode


@click.command("updates")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON (one object per line)",
)
def updates_command(output_json: bool):
    """Stream real-time updates from the backend (SSE)."""
    mode = OutputMode.JSON if output_json else OutputMode.HUMAN
    formatter = OutputFormatter(mode=mode)

    formatter.progress("Connecting to update stream...")
    formatter.hint("Press Ctrl+C to stop")

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        if mode == OutputMode.HUMAN:
            formatter.print("\n[dim]Disconnected[/dim]")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        client = SyncClient()

        for update in client.stream_updates():
            if output_json:
                # Output as JSON lines
                print(json.dumps(update.to_dict()))
            else:
                # Human-readable format
                formatter.print(update.get_display_message())

    except ServerNotRunningError as e:
        formatter.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        if mode == OutputMode.HUMAN:
            formatter.print("\n[dim]Disconnected[/dim]")
        sys.exit(0)
