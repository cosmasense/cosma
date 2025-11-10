# Cosma

<img align="right" src="./assets/logo.svg" height="150px" alt="Cosma Logo">

Search your files with natural language!

### How It Works

Choose which directories to index, and Cosma will process all files
in those directories into a search-optimized index. It'll also
watch for for changes to keep the index updated.

After files are indexed, you can search for them with natural language!
Cosma uses vector-powered search to find files quickly and seamlessly.

## Get Started

Currently, Cosma has only been tested on MacOS ARM.
Windows and Linux support is coming soon!

### Setup

Make sure you have [Ollama](https://ollama.com/) installed.

Cosma has a backend to serve search queries, so it must be started first.
This needs to always be running to watch for file changes and process files in the background.

```py
uvx cosma serve
```

### Running

To start the terminal UI and start searching, run the TUI.

> [!IMPORTANT]  
> The backend must be running for this command to work (see above).

```py
uvx cosma /path/to/directory/to/search
```
> [!WARNING]  
> This will begin processing all files in the directory specified,
> which will take some time if running locally.

## MacOS App

We're also working on a Mac app! If this seems like a useful project,
give us a star!
