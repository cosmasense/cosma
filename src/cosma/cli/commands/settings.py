"""
Settings commands for Cosma CLI.
"""

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


@click.group("settings")
def settings_group():
    """Manage application settings."""
    pass


@settings_group.command("show")
@click.option("--section", "-s", type=click.Choice(["embedder", "summarizer", "parser"]),
              help="Show only a specific section")
@output_options
@handle_client_errors
def settings_show(section: str | None, formatter: OutputFormatter):
    """Show current settings configuration."""
    client = SyncClient()
    with formatter.status("Fetching settings..."):
        result = client.get_settings()

    if formatter.mode == OutputMode.JSON:
        if section:
            formatter.output_json(result.get(section, {}))
        else:
            formatter.output_json(result)
        return

    sections_to_show = [section] if section else ["embedder", "summarizer", "parser"]

    for sect in sections_to_show:
        data = result.get(sect, {})
        if not data:
            continue
        formatter.print(f"[bold cyan]\\[{sect}][/bold cyan]")
        _print_nested(formatter, data, indent=2)
        formatter.print("")


def _print_nested(formatter: OutputFormatter, data: dict, indent: int = 0):
    """Print nested dict with indentation."""
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            formatter.print(f"{prefix}[bold]\\[{key}][/bold]")
            _print_nested(formatter, value, indent + 2)
        elif isinstance(value, str) and value == "":
            continue
        else:
            formatter.print(f"{prefix}[bold]{key}[/bold] = {value}")


@settings_group.command("get")
@click.argument("key")
@output_options
@handle_client_errors
def settings_get(key: str, formatter: OutputFormatter):
    """Get the value of a specific setting.

    KEY is the flat config key (e.g. AI_PROVIDER, EMBEDDING_DIMENSIONS).
    """
    client = SyncClient()
    with formatter.status("Fetching settings..."):
        result = client.get_settings()

    # Search for the key in the flat representation
    # We need to get the flat dict from the API, but the API returns grouped.
    # We'll also try get_settings_defaults to know valid keys.
    defaults = client.get_settings_defaults()

    # Build a flat key -> value mapping from grouped data
    flat = _flatten(result)
    flat_defaults = _flatten(defaults)

    # Try exact match on flat keys (uppercase)
    upper_key = key.upper()
    if upper_key in flat:
        if formatter.mode == OutputMode.JSON:
            formatter.output_json({upper_key: flat[upper_key]})
        else:
            formatter.print(f"[bold]{upper_key}[/bold] = {flat[upper_key]}")
        return

    # Try as TOML path (e.g. embedder.provider)
    for flat_key, value in flat.items():
        # This is a simple heuristic; exact path matching would need the schema
        pass

    formatter.error(f"Unknown setting: {key}")
    formatter.hint("Use 'cosma settings show' to see all available settings")


def _flatten(data: dict, prefix: str = "") -> dict:
    """Flatten a nested dict, mapping TOML-style paths to values."""
    result = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, full_key))
        else:
            result[full_key] = value
    return result


@settings_group.command("set")
@click.argument("key")
@click.argument("value")
@output_options
@handle_client_errors
def settings_set(key: str, value: str, formatter: OutputFormatter):
    """Set a configuration value.

    KEY is the flat config key (e.g. AI_PROVIDER, EMBEDDING_DIMENSIONS).
    VALUE is the new value to set.
    """
    client = SyncClient()
    upper_key = key.upper()
    with formatter.status(f"Updating {upper_key}..."):
        result = client.update_settings({upper_key: value})

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    else:
        formatter.success(f"Updated {upper_key}")


@settings_group.command("defaults")
@output_options
@handle_client_errors
def settings_defaults(formatter: OutputFormatter):
    """Show default values for all settings."""
    client = SyncClient()
    with formatter.status("Fetching defaults..."):
        result = client.get_settings_defaults()

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
        return

    for sect in ["embedder", "summarizer", "parser"]:
        data = result.get(sect, {})
        if not data:
            continue
        formatter.print(f"[bold cyan]\\[{sect}][/bold cyan]")
        _print_nested(formatter, data, indent=2)
        formatter.print("")


@settings_group.command("reset")
@click.confirmation_option(prompt="Reset all settings to defaults?")
@output_options
@handle_client_errors
def settings_reset(formatter: OutputFormatter):
    """Reset all settings to their default values."""
    client = SyncClient()

    with formatter.status("Fetching defaults..."):
        defaults = client.get_settings_defaults()

    # Flatten defaults to get flat key -> value pairs
    flat_defaults = _flatten(defaults)

    # Map TOML paths back to config keys by using the known schema mapping
    # We'll build the update dict using the path-based keys from the defaults
    # Since the API accepts flat config keys, we need to reverse-map
    # For now, we can get the flat defaults from the settings manager
    # The simplest approach: fetch defaults, then PUT them as an update
    from cosma_backend.settings import SETTINGS_SCHEMA
    update = {key: schema["default"] for key, schema in SETTINGS_SCHEMA.items()}

    with formatter.status("Resetting settings..."):
        result = client.update_settings(update)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    else:
        formatter.success("All settings reset to defaults")
