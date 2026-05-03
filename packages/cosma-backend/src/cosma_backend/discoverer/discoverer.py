from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cosma_backend.logging import get_logger
from cosma_backend.models import File

if TYPE_CHECKING:
    from cosma_backend.filter import FilterConfig

logger = get_logger(__name__)


def _stamp_partial_kind(file: File, decision_value: str) -> File:
    """If the filter classified this file as PARTIAL, mark it on the
    File so the pipeline routes it to embed-only with status
    INDEXED_PARTIAL. Pure helper — doesn't import FilterDecision to
    avoid a circular import; we compare against the value string.
    """
    if decision_value == "partial":
        file.metadata_only = True
        file.partial_kind = "user_elected"
        file.metadata_only_reason = (
            "User-configured filter rule classified this file as "
            "metadata-only (filename embedded; LLM summary skipped)."
        )
    return file


class Discoverer:
    """A simple file discoverer that recursively finds files under a given path."""


    def files_in(
        self,
        path: str | Path,
        filter_config: Optional["FilterConfig"] = None
    ) -> Generator[File, None, None]:
        """
        Discover all files under the specified path.

        Three-tier filter classification (when ``filter_config`` is
        provided): EXCLUDED files are dropped entirely (no DB row);
        PARTIAL files are yielded with ``metadata_only=True`` so the
        pipeline routes them to embed-only with status
        INDEXED_PARTIAL; FULL files are yielded unmarked for the
        normal parse → summarize → embed flow.

        Args:
            path: The root path to start discovery from
            filter_config: Optional filter config to exclude files

        Yields:
            File: Each file found under the specified path (excluding filtered files)
        """
        # Local import to keep `FilterDecision` out of the module's
        # public namespace and to avoid circular import at module load.
        from cosma_backend.filter import FilterDecision

        root = Path(path).resolve()

        if not root.exists():
            return

        if root.is_file():
            if filter_config is not None:
                decision = filter_config.classify(root, root.parent)
                if decision == FilterDecision.EXCLUDED:
                    logger.debug("Skipping excluded file", file=str(root))
                    return
                file = File.from_path(root)
                yield _stamp_partial_kind(file, decision.value)
                return
            yield File.from_path(root)
            return

        # Track counts for logging
        discovered_count = 0
        partial_count = 0
        filtered_count = 0

        for item in root.rglob("*"):
            if item.is_file():
                if filter_config is not None:
                    decision = filter_config.classify(item, root)
                    if decision == FilterDecision.EXCLUDED:
                        filtered_count += 1
                        logger.debug("Skipping excluded file", file=str(item))
                        continue
                    if decision == FilterDecision.PARTIAL:
                        partial_count += 1
                    discovered_count += 1
                    yield _stamp_partial_kind(File.from_path(item), decision.value)
                else:
                    discovered_count += 1
                    yield File.from_path(item)

        if filter_config is not None:
            logger.info("Discovery complete",
                          path=str(root),
                          discovered=discovered_count,
                          partial=partial_count,
                          filtered=filtered_count)
