# Cosma

[![Version](https://img.shields.io/pypi/v/cosma)](https://pypi.org/project/cosma/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/Zs6vUt4t5e)

<img align="right" src="./assets/logo.png" height="150px" alt="Cosma Logo">

Search engine for your files!

Use it yourself or plug it into your agents. Comes with a CLI, TUI,
and [macOS app](https://github.com/cosmasense/macos).

### How It Works

Choose which directories to index, and Cosma will process all files
in those directories into a search-optimized index. It'll also
watch for changes to keep the index updated.

After files are indexed, you can search for them with natural language!
Cosma uses vector-powered search to find files quickly and easily.

Cosma can run 100% locally or in the cloud.

## Get Started

Cosma has been tested on MacOS ARM and Windows.
Linux support is in progress.

### Installing

Cosma can be downloaded from PyPI.
We highly recommend you do this with [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv tool install cosma
```

### Upgrading

To upgrade to the latest version:

```sh
uv tool upgrade cosma --no-cache
```

## Running

### Backend

Make sure you have [Ollama](https://ollama.com/) installed.

Cosma has a backend to serve search queries, so it must be started first.
This needs to always be running to watch for file changes and process files in the
background. For persistent use, consider running this as a background process or
configure it as a system service.

```sh
cosma serve
```

> [!IMPORTANT]  
> The backend must be running for the following commands to work (see above).

### CLI

Cosma includes a CLI with three output modes:

- **default** - Rich formatted output with colors and tables for interactive use
- `--plain` - Simple text output without colors, for piping and scripts
- `--json` - Structured JSON output for agents and programmatic access

```sh
cosma search "my query"               # Search indexed files
cosma index /path/to/dir              # Index a directory
cosma status                          # Check backend status
cosma watch add /path/to/dir          # Watch directory for changes
cosma files stats                     # Get file statistics
```

Run `cosma --help` for all commands.

### TUI

Cosma also comes with a terminal UI (TUI).

```sh
cosma tui /path/to/directory/to/search
```
> [!WARNING]
> This will begin processing all files in the directory specified,
> which will take some time if running locally.

## Contributing

Cosma is open source, and we'd love to have you contribute!
Please feel free to open an issue or pull request with code changes,
or join our [Discord](https://discord.gg/Zs6vUt4t5e).
We'll have documentation for how best to contribute soon!
