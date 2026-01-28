"""
Filters commands for Cosma CLI.
"""

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


@click.group("filters")
def filters_group():
    """Manage file filtering configuration."""
    pass


@filters_group.command("show")
@output_options
@handle_client_errors
def filters_show(formatter: OutputFormatter):
    """Show current filter configuration."""
    client = SyncClient()
    with formatter.status("Fetching filter configuration..."):
        result = client.get_filter_config()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    mode = result.get("mode", "blacklist")
    exclude = result.get("exclude", [])
    include = result.get("include", [])
    config_path = result.get("config_path", "")

    formatter.print(f"[bold]Mode:[/bold] {mode}")
    formatter.print(f"[bold]Config:[/bold] {config_path or 'Default'}")
    formatter.print("")

    if mode == "blacklist":
        formatter.print("[bold]Exclude patterns:[/bold]")
        for pattern in exclude:
            formatter.print(f"  - {pattern}")
        if include:
            formatter.print("")
            formatter.print("[bold]Force include patterns:[/bold]")
            for pattern in include:
                formatter.print(f"  + {pattern}")
    else:
        formatter.print("[bold]Include patterns:[/bold]")
        for pattern in include:
            formatter.print(f"  + {pattern}")
        if exclude:
            formatter.print("")
            formatter.print("[bold]Exclude patterns:[/bold]")
            for pattern in exclude:
                formatter.print(f"  - {pattern}")


@filters_group.command("add")
@click.argument("pattern")
@click.option(
    "--type", "-t",
    "pattern_type",
    type=click.Choice(["exclude", "include"]),
    default="exclude",
    help="Pattern type (default: exclude)",
)
@output_options
@handle_client_errors
def filters_add(pattern: str, pattern_type: str, formatter: OutputFormatter):
    """Add a pattern to the filter configuration."""
    client = SyncClient()
    with formatter.status(f"Adding {pattern_type} pattern..."):
        result = client.add_filter_pattern(pattern, pattern_type)

    success = result.get("success", False)
    message = result.get("message", "")
    removed = result.get("removed_count", 0)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif success:
        formatter.success(f"Added {pattern_type} pattern: {pattern}")
        if removed > 0:
            formatter.print(f"Removed {removed} file(s) from index")
    else:
        formatter.error(message or "Failed to add pattern")


@filters_group.command("remove")
@click.argument("pattern")
@click.option(
    "--type", "-t",
    "pattern_type",
    type=click.Choice(["exclude", "include"]),
    default="exclude",
    help="Pattern type (default: exclude)",
)
@output_options
@handle_client_errors
def filters_remove(pattern: str, pattern_type: str, formatter: OutputFormatter):
    """Remove a pattern from the filter configuration."""
    client = SyncClient()
    with formatter.status(f"Removing {pattern_type} pattern..."):
        result = client.remove_filter_pattern(pattern, pattern_type)

    success = result.get("success", False)
    message = result.get("message", "")

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif success:
        formatter.success(f"Removed {pattern_type} pattern: {pattern}")
    else:
        formatter.error(message or "Failed to remove pattern")


@filters_group.command("test")
@click.argument("pattern")
@click.argument("paths", nargs=-1, required=True)
@click.option(
    "--mode", "-m",
    "filter_mode",
    type=click.Choice(["blacklist", "whitelist"]),
    default="blacklist",
    help="Filter mode (default: blacklist)",
)
@output_options
@handle_client_errors
def filters_test(pattern: str, paths: tuple, filter_mode: str, formatter: OutputFormatter):
    """Test pattern matching against file paths."""
    client = SyncClient()
    with formatter.status("Testing pattern..."):
        result = client.test_filter_patterns([pattern], list(paths), filter_mode)

    results = result.get("results", [])

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    headers = ["Path", "Included", "Matched Pattern"]
    rows = []
    for r in results:
        included = "Yes" if r.get("included") else "No"
        matched = r.get("matched_pattern") or "-"
        rows.append([r.get("file_path", "N/A"), included, matched])

    formatter.output_table(headers, rows, title="Pattern Test Results")


@filters_group.command("apply")
@output_options
@handle_client_errors
def filters_apply(formatter: OutputFormatter):
    """Apply current filter configuration to database."""
    client = SyncClient()
    with formatter.status("Applying filter changes..."):
        result = client.apply_filter_changes()

    success = result.get("success", False)
    message = result.get("message", "")
    removed = result.get("removed_count", 0)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif success:
        formatter.success(f"Filter changes applied: {removed} file(s) removed")
    else:
        formatter.error(message or "Failed to apply changes")


@filters_group.command("reset")
@click.confirmation_option(prompt="Reset filters to defaults?")
@output_options
@handle_client_errors
def filters_reset(formatter: OutputFormatter):
    """Reset filter configuration to defaults."""
    client = SyncClient()
    with formatter.status("Resetting filters to defaults..."):
        result = client.reset_filters()

    success = result.get("success", False)
    message = result.get("message", "")
    removed = result.get("removed_count", 0)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif success:
        formatter.success("Filters reset to defaults")
        if removed > 0:
            formatter.print(f"Removed {removed} file(s) from index")
    else:
        formatter.error(message or "Failed to reset filters")
