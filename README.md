# Cosma

[![Version](https://img.shields.io/pypi/v/cosma)](https://pypi.org/project/cosma/)

<img align="right" src="./assets/logo.svg" height="150px" alt="Cosma Logo">

Search engine for your files!

> [!CAUTION]
> This software is in early alpha! There will be lots of bugs.
> If you encounter any, please report them to the issue tracker.

### How It Works

Choose which directories to index, and Cosma will process all files
in those directories into a search-optimized index. It'll also
watch for for changes to keep the index updated.

After files are indexed, you can search for them with natural language!
Cosma uses vector-powered search to find files quickly and easily.

Cosma can run 100% locally or in the cloud.

## Get Started

Currently, Cosma has only been tested on MacOS ARM.
Windows and Linux support is coming soon!

### Installing

Cosma can be downloaded from PyPI.
We highly recommend you do this with [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv tool install comsa
```

### Upgrading

To upgrade to the latest version:

```sh
uv tool upgrade cosma --no-cache
```

### Setup

Make sure you have [Ollama](https://ollama.com/) installed.

Cosma has a backend to serve search queries, so it must be started first.
This needs to always be running to watch for file changes and process files in the background.

```sh
cosma serve
```

### Running

To start the terminal UI and start searching, run search.

> [!IMPORTANT]  
> The backend must be running for this command to work (see above).

```sh
cosma search /path/to/directory/to/search
```
> [!WARNING]  
> This will begin processing all files in the directory specified,
> which will take some time if running locally.

## MacOS App

We're also working on a Mac app! If this seems like a useful project,
give us a star!

## Contributing

Cosma is open source, and we'd love to have you contribute!
Please feel free to open an issue or pull request with code changes.
We'll have documentation for how best to contribute soon!
