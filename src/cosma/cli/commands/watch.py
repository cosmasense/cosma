"""
Watch commands for Cosma CLI.
"""

from pathlib import Path

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


@click.group("watch")
def watch_group():
    """Manage watched directories."""
    pass


@watch_group.command("add")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@output_options
@handle_client_errors
def watch_add(path: str, formatter: OutputFormatter):
    """Start watching a directory for changes."""
    resolved_path = Path(path).resolve()

    client = SyncClient()
    with formatter.status(f"Adding watch: {resolved_path}..."):
        result = client.watch_directory(str(resolved_path))

    success = result.get("success", False)
    message = result.get("message", "")

    if formatter.mode == OutputMode.JSON:
        formatter.output_json({
            "success": success,
            "message": message,
            "path": str(resolved_path),
        })
    elif success:
        formatter.success(f"Now watching: {resolved_path}")
    else:
        formatter.error(message or "Failed to add watch")


@watch_group.command("list")
@output_options
@handle_client_errors
def watch_list(formatter: OutputFormatter):
    """List all watched directories."""
    client = SyncClient()
    with formatter.status("Fetching watched directories..."):
        result = client.get_watch_jobs()

    jobs = result.get("jobs", [])

    if formatter.mode == OutputMode.JSON:
        formatter.output_json({"jobs": jobs})
        return

    if not jobs:
        formatter.hint("No directories are being watched.")
        return

    headers = ["ID", "Path", "Status", "Files"]
    rows = []
    for job in jobs:
        rows.append([
            job.get("id", "N/A"),
            job.get("path", job.get("directory_path", "N/A")),
            "Active" if job.get("is_active", False) else "Inactive",
            job.get("files_indexed", 0),
        ])

    formatter.output_table(headers, rows, title="Watched Directories")


@watch_group.command("stop")
@click.argument("job_id", type=int)
@output_options
@handle_client_errors
def watch_stop(job_id: int, formatter: OutputFormatter):
    """Stop watching a directory by job ID."""
    client = SyncClient()
    with formatter.status(f"Stopping watch job {job_id}..."):
        result = client.delete_watch_job(job_id)

    success = result.get("success", False)
    message = result.get("message", "")

    if formatter.mode == OutputMode.JSON:
        formatter.output_json({
            "success": success,
            "message": message,
            "job_id": job_id,
        })
    elif success:
        formatter.success(f"Stopped watching job {job_id}")
    else:
        formatter.error(message or f"Failed to stop job {job_id}")
