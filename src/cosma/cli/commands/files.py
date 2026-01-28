"""
Files commands for Cosma CLI.
"""

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter


@click.group("files")
def files_group():
    """Manage indexed files."""
    pass


@files_group.command("get")
@click.argument("file_id", type=int)
@output_options
@handle_client_errors
def files_get(file_id: int, formatter: OutputFormatter):
    """Get details of a specific file by ID."""
    client = SyncClient()
    with formatter.status(f"Fetching file {file_id}..."):
        result = client.get_file(file_id)

    if formatter.mode.value == "json":
        formatter.output_json(result)
        return

    # Format as key-value pairs
    data = {
        "ID": result.get("id", "N/A"),
        "Filename": result.get("filename", "N/A"),
        "Extension": result.get("extension", "N/A"),
        "Created": result.get("created", "N/A"),
        "Modified": result.get("modified", "N/A"),
        "Summary": result.get("summary", "N/A"),
    }

    keywords = result.get("keywords")
    if keywords:
        data["Keywords"] = ", ".join(keywords)

    formatter.output_dict(data, title=f"File {file_id}")


@files_group.command("stats")
@output_options
@handle_client_errors
def files_stats(formatter: OutputFormatter):
    """Get statistics about indexed files."""
    client = SyncClient()
    with formatter.status("Fetching file statistics..."):
        result = client.get_file_stats()

    if formatter.mode.value == "json":
        formatter.output_json(result)
        return

    data = {
        "Total Files": result.get("total_files", 0),
        "Total Size": format_size(result.get("total_size", 0)),
        "Last Indexed": result.get("last_indexed") or "Never",
    }

    file_types = result.get("file_types", {})
    if file_types:
        # Show top 5 file types
        sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:5]
        for ext, count in sorted_types:
            data[f".{ext}"] = count

    formatter.output_dict(data, title="File Statistics")


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
