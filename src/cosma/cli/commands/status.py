"""
Status command for Cosma CLI.
"""

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


@click.command("status")
@output_options
@handle_client_errors
def status_command(formatter: OutputFormatter):
    """Show backend server status."""
    client = SyncClient()
    with formatter.status("Checking server status..."):
        result = client.status()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    else:
        # Extract relevant status info
        data = {
            "Server": "Running",
        }

        # Add any additional status fields from the response
        for key, value in result.items():
            if key not in ("success",):
                data[key.replace("_", " ").title()] = value

        formatter.output_dict(data, title="Cosma Status")
