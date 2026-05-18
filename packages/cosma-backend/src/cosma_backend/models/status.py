"""
File Processing Status

Tracks the state of a file as it moves through the indexing pipeline.
"""

import enum


class ProcessingStatus(enum.Enum):
    """
    Pipeline stages a file progresses through during indexing.

    Normal flow:    DISCOVERED → PARSED → SUMMARIZED → COMPLETE
    Partial flow:   DISCOVERED → INDEXED_PARTIAL (filename + metadata
                    embedded; LLM summary intentionally skipped because
                    the user marked this file/folder as metadata-only
                    via the filter config)
    Error state:    FAILED (parse/embed failure, including the case
                    where a media file's codec defeated the parser)

    Values are ordered so status >= PARSED means content is available,
    etc. INDEXED_PARTIAL sits AFTER COMPLETE in the ordering because it
    is a terminal "indexed and ready to be searched" state — searches
    treat INDEXED_PARTIAL the same as COMPLETE for inclusion.

    Database serialization: stored by name (not value), so adding a
    new entry here doesn't require a schema migration. File.from_row
    decodes via ProcessingStatus[row["status"]] and falls back to
    DISCOVERED if it sees an unknown name.
    """
    DISCOVERED = 0       # File found, metadata captured, not yet processed
    PARSED = 1           # Content extracted (text, transcript, etc.)
    SUMMARIZED = 2       # AI summary and keywords generated
    COMPLETE = 3         # Embeddings generated, fully indexed

    FAILED = 4           # Processing failed (check processing_error field)
    INDEXED_PARTIAL = 5  # Filename + metadata got a SEMANTIC embedding
                         # (Tier B), no LLM summary. The intent shows in
                         # the per-file `partial_kind` column.
    INDEXED_NAME_ONLY = 6  # Tier C floor: filename in FTS only, no
                         # semantic vector. Cheap baseline so a literal-
                         # string search never returns zero results, used
                         # for code/build artifacts/anything we can't
                         # meaningfully summarize. Surfaces in search.
