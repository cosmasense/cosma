"""
Processing Pipeline

Orchestrates file processing through four stages:
1. Discovery - Find files matching filter patterns
2. Parsing - Extract text content (Spotlight, MarkItDown, Whisper)
3. Summarization - Generate AI summary, title, and keywords
4. Embedding - Create vector for semantic search

Each stage emits SSE updates for real-time progress tracking.
Files can be processed individually or in batches by directory.
"""

from .pipeline import Pipeline as Pipeline
from .pipeline import PipelineResult as PipelineResult
