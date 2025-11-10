from collections.abc import Generator
from datetime import datetime
from pathlib import Path

from cosma_backend.models import File


class Discoverer:
    """A simple file discoverer that recursively finds files under a given path."""
        
    
    def files_in(self, path: str | Path) -> Generator[File, None, None]:
        """
        Discover all files under the specified path.
        
        Args:
            path: The root path to start discovery from
            
        Yields:
            File: Each file found under the specified path
        """
        root = Path(path)
        
        if not root.exists():
            return
        
        if root.is_file():
            yield File.from_path(root)
            return
        
        # Recursively discover all files
        for item in root.rglob("*"):
            if item.is_file():
                yield File.from_path(item)
