import sys
from importlib.resources import files
from pathlib import Path

def get_bundled_file(relative_path: str):
    """Get path to bundled file, works in dev and production"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)  # type: ignore
        return base_path / relative_path
    else:
        # Running in normal Python - use importlib.resources
        # The package is 'backend' and schema.sql is in the package root
        return files('backend').joinpath(relative_path)

def get_bundled_file_text(relative_path: str) -> str:
    """Get text content of bundled file, works in dev and production"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        file_path = Path(sys._MEIPASS) / relative_path  # type: ignore
        return file_path.read_text()
    else:
        # Running in normal Python - use importlib.resources
        return files('backend').joinpath(relative_path).read_text()
