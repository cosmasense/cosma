"""
Settings commands for Cosma CLI.
"""

import click

from cosma_client import SyncClient

from .. import output_options, handle_client_errors, OutputFormatter, OutputMode


SETTINGS_EPILOG = """
\b
Settings are persisted to a TOML file in your config directory and
take effect immediately. Environment variables (COSMA_<KEY>) still
override file values on server startup.
\b
Keys can be specified as either:
  - Flat config keys:  EMBEDDING_PROVIDER, AI_PROVIDER
  - TOML paths:        embedder.provider, summarizer.provider
\b
Sections:
  embedder     Embedding model and provider configuration
  summarizer   LLM provider, model selection, and context settings
  parser       File extraction strategy, Spotlight, and Whisper
\b
Common keys:
  AI_PROVIDER / summarizer.provider
      LLM provider: auto, ollama, online, llamacpp
  EMBEDDING_PROVIDER / embedder.provider
      Embedding provider: local, online
  OLLAMA_MODEL / summarizer.ollama.model
      Ollama model name for summarization
  EMBEDDING_MODEL / embedder.model
      Online embedding model name
  EXTRACTION_STRATEGY / parser.extraction_strategy
      File parsing strategy: spotlight_first, auto
\b
Examples:
  cosma settings show
  cosma settings show -s embedder
  cosma settings get AI_PROVIDER
  cosma settings set AI_PROVIDER ollama
  cosma settings set summarizer.ollama.model llama3
  cosma settings set parser.spotlight_enabled false
  cosma settings defaults
  cosma settings reset
"""


@click.group("settings", epilog=SETTINGS_EPILOG)
def settings_group():
    """Manage application settings.

    View, update, and reset runtime configuration for the embedder,
    summarizer, and parser. Changes are saved to settings.toml and
    applied immediately.
    """
    pass


@settings_group.command("show", epilog="""\b
Examples:
  cosma settings show
  cosma settings show -s embedder
  cosma settings show --json
""")
@click.option("--section", "-s", type=click.Choice(["embedder", "summarizer", "parser"]),
              help="Show only a specific section.")
@output_options
@handle_client_errors
def settings_show(section: str | None, formatter: OutputFormatter):
    """Show current settings configuration.

    Displays all settings grouped by section (embedder, summarizer,
    parser). Use --section to filter to a single section.
    """
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


@settings_group.command("get", epilog="""\b
Examples:
  cosma settings get AI_PROVIDER
  cosma settings get summarizer.provider
  cosma settings get EMBEDDING_DIMENSIONS
  cosma settings get embedder.model --json
""")
@click.argument("key")
@output_options
@handle_client_errors
def settings_get(key: str, formatter: OutputFormatter):
    """Get the value of a specific setting.

    \b
    KEY can be a flat config key or a TOML path:
      cosma settings get AI_PROVIDER
      cosma settings get summarizer.provider
    """
    client = SyncClient()
    with formatter.status("Fetching settings..."):
        result = client.get_settings()

    flat = _flatten(result)

    # Try as TOML path first (e.g. parser.spotlight_enabled)
    if key in flat:
        found_key, value = key, flat[key]
    # Try uppercased as TOML path
    elif key.lower() in {k.lower(): k for k in flat}:
        normalized = {k.lower(): k for k in flat}
        real_key = normalized[key.lower()]
        found_key, value = real_key, flat[real_key]
    else:
        formatter.error(f"Unknown setting: {key}")
        formatter.hint("Use 'cosma settings show' to see all available settings")
        return

    if formatter.mode == OutputMode.JSON:
        formatter.output_json({found_key: value})
    else:
        formatter.print(f"[bold]{found_key}[/bold] = {value}")


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


@settings_group.command("set", epilog="""\b
Examples:
  cosma settings set AI_PROVIDER ollama
  cosma settings set summarizer.ollama.model llama3
  cosma settings set EMBEDDING_DIMENSIONS 1024
  cosma settings set parser.spotlight_enabled false
  cosma settings set WHISPER_PROVIDER local
""")
@click.argument("key")
@click.argument("value")
@output_options
@handle_client_errors
def settings_set(key: str, value: str, formatter: OutputFormatter):
    """Set a configuration value.

    \b
    KEY can be a flat config key or a TOML path:
      cosma settings set AI_PROVIDER ollama
      cosma settings set summarizer.provider ollama

    Values are automatically coerced to the correct type
    (int, bool, string) based on the setting's schema.
    """
    client = SyncClient()
    with formatter.status(f"Updating {key}..."):
        result = client.update_settings({key: value})

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    else:
        formatter.success(f"Updated {key}")


@settings_group.command("defaults", epilog="""\b
Examples:
  cosma settings defaults
  cosma settings defaults --json
""")
@output_options
@handle_client_errors
def settings_defaults(formatter: OutputFormatter):
    """Show default values for all settings.

    Displays the built-in default value for every setting. Useful for
    seeing what a setting will revert to after a reset, or for comparing
    against your current configuration.
    """
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


@settings_group.command("reset", epilog="""\b
Examples:
  cosma settings reset
  cosma settings reset --yes          # skip confirmation prompt
""")
@click.confirmation_option(prompt="Reset all settings to defaults?")
@output_options
@handle_client_errors
def settings_reset(formatter: OutputFormatter):
    """Reset all settings to their default values.

    Restores every setting to its built-in default and saves the result
    to settings.toml. This does not affect bootstrap settings (HOST,
    PORT, DATABASE_PATH) which are always read from environment variables.
    """
    client = SyncClient()

    with formatter.status("Fetching defaults..."):
        defaults = client.get_settings_defaults()

    # Flatten the grouped defaults into TOML paths and send them back
    flat_defaults = _flatten(defaults)

    with formatter.status("Resetting settings..."):
        result = client.update_settings(flat_defaults)

    if formatter.mode == OutputMode.JSON:
        formatter.output_json(result)
    else:
        formatter.success("All settings reset to defaults")
