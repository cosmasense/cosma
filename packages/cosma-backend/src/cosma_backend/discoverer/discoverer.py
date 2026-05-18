import os
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


def _stamp_tier(file: File, tier_value: str) -> File:
    """Translate the tier decision into the File flags that the pipeline
    already knows how to route on.

    The pipeline's branch table is keyed off (metadata_only, partial_kind).
    Rather than teach the pipeline a parallel "tier" concept, we map:

      * Tier.FULL          → no marking. Normal parse → summarize → embed.
      * Tier.SEMANTIC_NAME → metadata_only=True, partial_kind="user_elected".
                             Reuses the existing _run_partial_only path
                             (filename + metadata → semantic embedding).
      * Tier.LITERAL_NAME  → metadata_only=True, partial_kind="filename_only".
                             Routes to _run_filename_only — DB row + FTS
                             entry only, no embed.

    Using the existing `user_elected` value for Tier B preserves the
    on-disk schema for in-progress / resume rows (a row stuck on
    INDEXED_PARTIAL from a v1.1.x backend round-trips through the
    routing unchanged).
    """
    if tier_value == "semantic_name":
        file.metadata_only = True
        file.partial_kind = "user_elected"
        file.metadata_only_reason = (
            "Tier B (semantic-name): filename and folder embedded; "
            "content not summarized."
        )
    elif tier_value == "literal_name":
        file.metadata_only = True
        file.partial_kind = "filename_only"
        file.metadata_only_reason = (
            "Tier C (literal-name): filename indexed for keyword search; "
            "no semantic embedding."
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
                # `classify` still handles the historical PARTIAL bucket
                # (legacy metadata_only_patterns + EXCLUDED gating).
                # The new tier system runs on top: anything that came
                # back as FULL from classify() may still be downgraded
                # to SEMANTIC_NAME or LITERAL_NAME by the tier rules.
                file = _stamp_partial_kind(file, decision.value)
                if not file.metadata_only:
                    tier = filter_config.classify_tier(
                        root, root.parent,
                        file_size_bytes=file.file_size,
                    )
                    file = _stamp_tier(file, tier.value)
                yield file
                return
            yield File.from_path(root)
            return

        # Track counts for logging
        discovered_count = 0
        partial_count = 0
        filtered_count = 0
        git_repo_skipped = 0

        # `skip_git_repos` is a structural rule (drop the entire subtree
        # if it's a git working copy), not a per-file pattern, so it
        # belongs in the directory walk rather than `classify()`. The
        # `topdown=True` mode lets us mutate `dirnames` in place to
        # prune subdirectories before os.walk descends into them — this
        # is what makes the skip actually save the recursive scan cost
        # rather than just hiding files after the fact.
        skip_git_repos = (
            filter_config.skip_git_repos if filter_config is not None else False
        )

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            if skip_git_repos and ".git" in dirnames:
                # The user pointed cosma at, say, ~/Documents and one
                # subfolder is a git checkout. Don't recurse into it,
                # and don't index the .git/ admin tree either.
                git_repo_skipped += 1
                logger.debug("Skipping git repo subtree", dir=dirpath)
                dirnames.clear()
                continue

            for name in filenames:
                item = Path(dirpath) / name
                if filter_config is not None:
                    decision = filter_config.classify(item, root)
                    if decision == FilterDecision.EXCLUDED:
                        filtered_count += 1
                        logger.debug("Skipping excluded file", file=str(item))
                        continue
                    if decision == FilterDecision.PARTIAL:
                        partial_count += 1
                    discovered_count += 1
                    file_obj = _stamp_partial_kind(File.from_path(item), decision.value)
                    if not file_obj.metadata_only:
                        # File came back as FULL from the legacy pattern
                        # classifier. The tier rules get the final say —
                        # they may downgrade it to SEMANTIC_NAME or
                        # LITERAL_NAME depending on extension and size.
                        tier = filter_config.classify_tier(
                            item, root,
                            file_size_bytes=file_obj.file_size,
                        )
                        file_obj = _stamp_tier(file_obj, tier.value)
                    yield file_obj
                else:
                    discovered_count += 1
                    yield File.from_path(item)

        if filter_config is not None:
            logger.info("Discovery complete",
                          path=str(root),
                          discovered=discovered_count,
                          partial=partial_count,
                          filtered=filtered_count,
                          git_repos_skipped=git_repo_skipped)
