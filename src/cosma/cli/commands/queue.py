"""
Queue commands for Cosma CLI.
"""

from datetime import datetime, timezone
from pathlib import Path

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


@click.group("queue")
def queue_group():
    """Manage the indexing queue."""
    pass


# ------------------------------------------------------------------
# Status / Pause / Resume
# ------------------------------------------------------------------

@queue_group.command("status")
@output_options
@handle_client_errors
def queue_status(formatter: OutputFormatter):
    """Show queue status."""
    client = SyncClient()
    with formatter.status("Fetching queue status..."):
        result = client.get_queue_status()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    paused = result.get("paused", False)
    manually_paused = result.get("manually_paused", False)
    scheduler_paused = result.get("scheduler_paused", False)

    if manually_paused:
        pause_label = "Paused (manual)"
    elif scheduler_paused:
        pause_label = "Paused (scheduler)"
    elif paused:
        pause_label = "Paused"
    else:
        pause_label = "Running"

    data = {
        "State": pause_label,
        "Total Items": result.get("total_items", 0),
        "Processing": result.get("processing", 0),
        "Waiting": result.get("waiting", 0),
        "Cooling Down": result.get("cooling_down", 0),
    }

    formatter.output_dict(data, title="Queue Status")

    failing = result.get("failing_rules", [])
    if failing:
        formatter.print("")
        formatter.print("[bold]Failing scheduler rules:[/bold]")
        for rule in failing:
            formatter.print(f"  - {rule}")


@queue_group.command("pause")
@output_options
@handle_client_errors
def queue_pause(formatter: OutputFormatter):
    """Pause the indexing queue."""
    client = SyncClient()
    with formatter.status("Pausing queue..."):
        result = client.pause_queue()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif result.get("success"):
        formatter.success("Queue paused")
    else:
        formatter.error(result.get("message", "Failed to pause queue"))


@queue_group.command("resume")
@output_options
@handle_client_errors
def queue_resume(formatter: OutputFormatter):
    """Resume the indexing queue."""
    client = SyncClient()
    with formatter.status("Resuming queue..."):
        result = client.resume_queue()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif result.get("success"):
        formatter.success("Queue resumed")
    else:
        formatter.error(result.get("message", "Failed to resume queue"))


# ------------------------------------------------------------------
# Items
# ------------------------------------------------------------------

@queue_group.command("items")
@click.option("--limit", "-n", default=50, help="Max items to show.")
@click.option("--offset", default=0, help="Offset for pagination.")
@output_options
@handle_client_errors
def queue_items(limit: int, offset: int, formatter: OutputFormatter):
    """List items in the queue."""
    client = SyncClient()
    with formatter.status("Fetching queue items..."):
        result = client.get_queue_items(offset=offset, limit=limit)

    items = result.get("items", [])
    total = result.get("total_count", 0)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    if not items:
        formatter.hint("Queue is empty.")
        return

    headers = ["ID", "File", "Action", "Status", "Attempts"]
    rows = []
    for item in items:
        file_path = item.get("file_path", "")
        filename = Path(file_path).name if file_path else "N/A"
        rows.append([
            item.get("id", "N/A"),
            filename,
            item.get("action", "N/A"),
            item.get("status", "N/A"),
            item.get("attempts", 0),
        ])

    formatter.output_table(headers, rows, title=f"Queue Items ({total} total)")


@queue_group.command("remove")
@click.argument("item_id")
@output_options
@handle_client_errors
def queue_remove(item_id: str, formatter: OutputFormatter):
    """Remove an item from the queue."""
    client = SyncClient()
    with formatter.status(f"Removing item {item_id}..."):
        result = client.remove_queue_item(item_id)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif result.get("success"):
        formatter.success(f"Removed item {item_id}")
    else:
        formatter.error(result.get("message", "Item not found"))


# ------------------------------------------------------------------
# Failed / Recent / Reindex
# ------------------------------------------------------------------

def _format_timestamp(ts):
    """Format a unix timestamp for display."""
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


@queue_group.command("failed")
@click.option("--limit", "-n", default=20, help="Max items to show.")
@click.option("--offset", default=0, help="Offset for pagination.")
@output_options
@handle_client_errors
def queue_failed(limit: int, offset: int, formatter: OutputFormatter):
    """List files that failed processing."""
    client = SyncClient()
    with formatter.status("Fetching failed files..."):
        result = client.get_failed_files(offset=offset, limit=limit)

    files = result.get("files", [])
    total = result.get("total_count", 0)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    if not files:
        formatter.hint("No failed files.")
        return

    headers = ["File", "Extension", "Error", "Updated"]
    rows = []
    for f in files:
        rows.append([
            f.get("filename", "N/A"),
            f.get("extension", ""),
            (f.get("processing_error") or "-")[:60],
            _format_timestamp(f.get("updated_at")),
        ])

    formatter.output_table(headers, rows, title=f"Failed Files ({total} total)")


@queue_group.command("recent")
@click.option("--limit", "-n", default=20, help="Max items to show.")
@click.option("--offset", default=0, help="Offset for pagination.")
@output_options
@handle_client_errors
def queue_recent(limit: int, offset: int, formatter: OutputFormatter):
    """List recently completed files."""
    client = SyncClient()
    with formatter.status("Fetching recent files..."):
        result = client.get_recent_files(offset=offset, limit=limit)

    files = result.get("files", [])
    total = result.get("total_count", 0)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    if not files:
        formatter.hint("No recently completed files.")
        return

    headers = ["File", "Extension", "Updated"]
    rows = []
    for f in files:
        rows.append([
            f.get("filename", "N/A"),
            f.get("extension", ""),
            _format_timestamp(f.get("updated_at")),
        ])

    formatter.output_table(headers, rows, title=f"Recent Files ({total} total)")


@queue_group.command("reindex")
@click.argument("file_path", type=click.Path())
@output_options
@handle_client_errors
def queue_reindex(file_path: str, formatter: OutputFormatter):
    """Re-index a file (deletes old record and re-queues)."""
    resolved = str(Path(file_path).resolve())

    client = SyncClient()
    with formatter.status(f"Re-indexing {resolved}..."):
        result = client.reindex_file(resolved)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    elif result.get("success"):
        formatter.success(result.get("message", f"File enqueued for reindexing: {resolved}"))
    else:
        formatter.error(result.get("message", "Failed to reindex file"))


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@queue_group.command("metrics")
@output_options
@handle_client_errors
def queue_metrics(formatter: OutputFormatter):
    """Show system metrics and model status."""
    client = SyncClient()
    with formatter.status("Fetching metrics..."):
        result = client.get_queue_metrics()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    metrics = result.get("metrics", {})
    if metrics:
        data = {}
        for key, value in metrics.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                data[label] = f"{value:.1f}"
            else:
                data[label] = value
        formatter.output_dict(data, title="System Metrics")

    models = result.get("models", [])
    if models:
        formatter.print("")
        headers = ["Model", "Status", "Type"]
        rows = []
        for m in models:
            rows.append([
                m.get("name", "N/A"),
                m.get("status", "N/A"),
                m.get("type", "N/A"),
            ])
        formatter.output_table(headers, rows, title="Models")


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

@queue_group.command("scheduler")
@output_options
@handle_client_errors
def queue_scheduler(formatter: OutputFormatter):
    """Show scheduler configuration and status."""
    client = SyncClient()
    with formatter.status("Fetching scheduler status..."):
        result = client.get_scheduler()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    data = {
        "Enabled": result.get("enabled", False),
        "Combine Mode": result.get("combine_mode", "N/A"),
        "Check Interval": f"{result.get('check_interval_seconds', 0)}s",
        "Conditions Met": result.get("conditions_met", False),
    }
    formatter.output_dict(data, title="Scheduler")

    warnings = result.get("warnings", [])
    if warnings:
        formatter.print("")
        formatter.print("[bold]Warnings:[/bold]")
        for w in warnings:
            formatter.print(f"  - {w}")

    rules = result.get("rules", [])
    if rules:
        formatter.print("")
        headers = ["Type", "Threshold", "Enabled"]
        rows = []
        for r in rules:
            rows.append([
                r.get("rule_type", "N/A"),
                str(r.get("threshold", "")),
                r.get("enabled", True),
            ])
        formatter.output_table(headers, rows, title="Rules")

    rule_results = result.get("rule_results", [])
    if rule_results:
        formatter.print("")
        headers = ["Rule", "Passed", "Value"]
        rows = []
        for rr in rule_results:
            rows.append([
                rr.get("rule", "N/A"),
                rr.get("passed", "N/A"),
                str(rr.get("value", "")),
            ])
        formatter.output_table(headers, rows, title="Last Rule Results")


@queue_group.command("scheduler-test")
@output_options
@handle_client_errors
def queue_scheduler_test(formatter: OutputFormatter):
    """Dry-run scheduler rules against live metrics."""
    client = SyncClient()
    with formatter.status("Testing scheduler rules..."):
        result = client.test_scheduler()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    conditions_met = result.get("conditions_met", False)
    formatter.print(f"[bold]Conditions met:[/bold] {conditions_met}")

    rule_results = result.get("rule_results", [])
    if rule_results:
        formatter.print("")
        headers = ["Rule", "Passed", "Value", "Threshold"]
        rows = []
        for rr in rule_results:
            rows.append([
                rr.get("rule", "N/A"),
                rr.get("passed", "N/A"),
                str(rr.get("value", "")),
                str(rr.get("threshold", "")),
            ])
        formatter.output_table(headers, rows, title="Rule Results")

    warnings = result.get("warnings", [])
    if warnings:
        formatter.print("")
        formatter.print("[bold]Warnings:[/bold]")
        for w in warnings:
            formatter.print(f"  - {w}")


@queue_group.command("scheduler-set")
@click.option("--enabled/--disabled", default=None, help="Enable or disable the scheduler.")
@click.option("--interval", type=int, help="Check interval in seconds.")
@click.option("--combine-mode", type=click.Choice(["all", "any"]), help="Rule combination mode.")
@output_options
@handle_client_errors
def queue_scheduler_set(enabled, interval, combine_mode, formatter: OutputFormatter):
    """Update scheduler configuration."""
    config = {}
    if enabled is not None:
        config["enabled"] = enabled
    if interval is not None:
        config["check_interval_seconds"] = interval
    if combine_mode is not None:
        config["combine_mode"] = combine_mode

    if not config:
        formatter.error("No options specified. Use --help to see available options.")
        return

    client = SyncClient()
    with formatter.status("Updating scheduler..."):
        result = client.update_scheduler(config)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    else:
        formatter.success("Scheduler configuration updated")
        data = {
            "Enabled": result.get("enabled", False),
            "Combine Mode": result.get("combine_mode", "N/A"),
            "Check Interval": f"{result.get('check_interval_seconds', 0)}s",
        }
        formatter.output_dict(data)
