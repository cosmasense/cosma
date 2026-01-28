"""
Search command for Cosma CLI.
"""

import os
from pathlib import Path

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


@click.command("search")
@click.argument("query")
@click.option(
    "-n", "--limit",
    default=20,
    type=int,
    help="Maximum number of results (default: 20)",
)
@click.option(
    "-d", "--directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Filter results to a specific directory",
)
@output_options
@handle_client_errors
def search_command(
    query: str,
    limit: int,
    directory: str | None,
    formatter: OutputFormatter,
):
    """Search for files matching a query."""
    client = SyncClient()

    # Build filters if directory specified
    filters = None
    if directory:
        filters = {"directory": str(Path(directory).resolve())}

    with formatter.status(f"Searching for '{query}'..."):
        result = client.search(query, filters=filters, limit=limit)

    results = result.get("results", [])

    if not results:
        formatter.hint("No results found. Try a different query or index more files.")
        if formatter.mode == OutputMode.JSON:
            formatter.output_json({"results": [], "count": 0})
        return

    # Format results as table
    headers = ["Score", "File", "Summary"]
    rows = []
    for item in results:
        score = item.get("relevance_score", 0)
        file_info = item.get("file", {})
        file_path = file_info.get("file_path", file_info.get("filename", "N/A"))
        # Shorten path for display in human mode
        if formatter.is_human():
            try:
                file_path = os.path.relpath(file_path)
            except ValueError:
                pass  # Keep absolute if on different drive
        summary_full = file_info.get("summary") or ""
        summary = summary_full[:80]
        if len(summary_full) > 80:
            summary += "..."
        rows.append([f"{score:.2f}", file_path, summary])

    if formatter.mode == OutputMode.JSON:
        formatter.output_json({
            "results": results,
            "count": len(results),
        })
    else:
        formatter.output_table(headers, rows, title=f"Search: {query}")
        if formatter.is_human():
            formatter.hint(f"Found {len(results)} result(s)")
