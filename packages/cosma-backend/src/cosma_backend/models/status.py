"""
File Processing Status

Tracks the state of a file as it moves through the indexing pipeline.
"""

import enum


class ProcessingStatus(enum.Enum):
    """
    Pipeline stages a file progresses through during indexing.

    Normal flow: DISCOVERED → PARSED → SUMMARIZED → COMPLETE
    Error state: FAILED (can occur at any stage)

    Values are ordered so status >= PARSED means content is available, etc.
    """
    DISCOVERED = 0   # File found, metadata captured, not yet processed
    PARSED = 1       # Content extracted (text, transcript, etc.)
    SUMMARIZED = 2   # AI summary and keywords generated
    COMPLETE = 3     # Embeddings generated, fully indexed

    FAILED = 4       # Processing failed (check processing_error field)
