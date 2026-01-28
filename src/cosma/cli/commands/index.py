"""
Index command for Cosma CLI.
"""

from pathlib import Path

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


@click.command("index")
@click.argument(
    "path",
    type=click.Path(exists=True),
    default=".",
)
@output_options
@handle_client_errors
def index_command(
    path: str,
    formatter: OutputFormatter,
):
    """Index a directory or file for searching."""
    resolved_path = Path(path).resolve()

    client = SyncClient()

    with formatter.status(f"Indexing {resolved_path}..."):
        if resolved_path.is_dir():
            result = client.index_directory(str(resolved_path))
        else:
            result = client.index_file(str(resolved_path))

    success = result.get("success", False)
    message = result.get("message", "")
    files_indexed = result.get("files_indexed", 0)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json({
            "success": success,
            "message": message,
            "path": str(resolved_path),
            "files_indexed": files_indexed,
        })
    elif success:
        formatter.success(f"Indexing started: {resolved_path}")
        if files_indexed:
            formatter.print(f"Files queued: {files_indexed}")
    else:
        formatter.error(message or "Indexing failed")
