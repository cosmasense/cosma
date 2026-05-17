from importlib.resources import files


def get_bundled_file(relative_path: str):
    """Get path to bundled file, works in dev and PyInstaller."""
    return files("cosma_backend").joinpath(relative_path)


def get_bundled_file_text(relative_path: str) -> str:
    """Get text content of bundled file, works in dev and PyInstaller."""
    return files("cosma_backend").joinpath(relative_path).read_text()
