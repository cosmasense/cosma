from collections.abc import Generator
from datetime import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cosma_backend.logging import sm
from cosma_backend.models import File

if TYPE_CHECKING:
    from cosma_backend.filter import FilterConfig

logger = logging.getLogger(__name__)


class Discoverer:
    """A simple file discoverer that recursively finds files under a given path."""


    def files_in(
        self,
        path: str | Path,
        filter_config: Optional["FilterConfig"] = None
    ) -> Generator[File, None, None]:
        """
        Discover all files under the specified path.

        Args:
            path: The root path to start discovery from
            filter_config: Optional filter config to exclude files

        Yields:
            File: Each file found under the specified path (excluding filtered files)
        """
        root = Path(path).resolve()

        if not root.exists():
            return

        if root.is_file():
            # Single file - check filter
            if filter_config is not None:
                if not filter_config.should_include(root, root.parent):
                    logger.debug(sm("Skipping excluded file", file=str(root)))
                    return
            yield File.from_path(root)
            return

        # Track counts for logging
        discovered_count = 0
        filtered_count = 0

        # Recursively discover all files
        for item in root.rglob("*"):
            if item.is_file():
                # Check filter before yielding
                if filter_config is not None:
                    if not filter_config.should_include(item, root):
                        filtered_count += 1
                        logger.debug(sm("Skipping excluded file", file=str(item)))
                        continue

                discovered_count += 1
                yield File.from_path(item)

        if filter_config is not None:
            logger.info(sm("Discovery complete",
                          path=str(root),
                          discovered=discovered_count,
                          filtered=filtered_count))
