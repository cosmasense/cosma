"""
Base summarizer class and exceptions.
"""

from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from cosma_backend.logging import get_logger
from cosma_backend.models import ProcessingStatus
from cosma_backend.models.file import File
from cosma_backend.utils.decorators import async_wrap

from .tokenization import chunk_content, estimate_tokens

if TYPE_CHECKING:
    from cosma_backend.settings import SummarizerConfig

logger = get_logger(__name__)


class SummarizerError(Exception):
    """Base exception for summarizer errors."""
    pass


class AIProviderError(SummarizerError):
    """Exception for AI provider-specific errors."""
    pass


class BaseSummarizer(ABC):
    """Abstract base class for file summarizers."""

    def __init__(self, config: SummarizerConfig | None = None, max_tokens: Optional[int] = None, model: Optional[str] = None):
        """
        Initialize summarizer with context length limit.

        Args:
            config: SummarizerConfig instance
            max_tokens: Maximum tokens for the model context
            model: Model name for accurate tokenization (optional)
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()
        self.max_tokens = max_tokens or self.config.max_tokens_per_request
        self.chunk_overlap = self.config.chunk_overlap_tokens
        self.model = model

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this summarizer is available for use."""
        pass

    def _validate_content(self, file_metadata: File) -> bool:
        """Validate that the file metadata has content to summarize."""
        if not file_metadata.content:
            logger.warning("File content is empty, cannot summarize", filename=file_metadata.filename)
            return False

        if len(file_metadata.content.strip()) < 10:
            logger.warning("File content too short to summarize", filename=file_metadata.filename, length=len(file_metadata.content))
            return False

        return True

    @async_wrap
    def _prepare_images(self, file_metadata: File) -> list[str]:
        """Prepare images for vision analysis.

        Includes the file itself (if an image) plus any extra_images (e.g., video frames).
        """
        images = []

        # Add the file itself if it's an image
        if file_metadata.content_type and file_metadata.content_type.startswith("image"):
            with open(file_metadata.path, 'rb') as f:
                images.append(base64.b64encode(f.read()).decode('utf-8'))

        # Add any extra images (e.g., video frames)
        if file_metadata.extra_images:
            for frame_bytes in file_metadata.extra_images:
                images.append(base64.b64encode(frame_bytes).decode('utf-8'))

        return images

    async def _prepare_content(self, content: str) -> List[str]:
        """
        Prepare content for summarization, chunking if necessary.
        Uses fast token estimation for efficiency.

        Args:
            content: The content to prepare

        Returns:
            List of content chunks ready for processing
        """
        # Use fast estimation for initial analysis
        estimated_tokens = await estimate_tokens(content, self.model)

        # If within limits using fast estimation, do a quick accurate check
        if estimated_tokens <= self.max_tokens:
            return [content]

        logger.info("Content exceeds token limit, chunking required", estimated_tokens=estimated_tokens, max_tokens=self.max_tokens)

        # Use optimized chunking
        chunks = await chunk_content(content, self.max_tokens, self.chunk_overlap, self.model)
        logger.info("Content chunked (noverify)", num_chunks=len(chunks))

        if not chunks:
            # This should no longer happen (chunk_content now has a
            # character-split fallback), but if it does we absolutely
            # cannot return the full content — that sends an oversized
            # prompt straight to the model. Truncate instead.
            logger.error("Chunking produced no output even after fallback; "
                         "truncating content to max_tokens worth of chars")
            approx_chars = max(1, self.max_tokens * 3)
            return [content[:approx_chars]]

        # Fast mode collapses the per-file budget to a single chunk:
        # trades long-doc coverage for throughput. The chunk we keep
        # is the first one, which (because chunk_content() splits in
        # document order) covers the document head — usually where
        # the title, abstract, and signal-rich first paragraphs live.
        max_chunks = (
            1 if getattr(self.config, "fast_mode", False)
            else self.config.max_chunks
        )

        # Use fast estimation for chunk statistics (sample a few chunks for accurate check)
        if len(chunks) <= max_chunks:
            # For small number of chunks, verify all with accurate tokenization
            accurate_chunk_tokens = [await estimate_tokens(chunk, self.model, use_fast=False) for chunk in chunks]
            avg_chunk_tokens = sum(accurate_chunk_tokens) // len(chunks)
            max_chunk_tokens = max(accurate_chunk_tokens)
            logger.info("Content chunked and verified", num_chunks=len(chunks), avg_chunk_tokens=avg_chunk_tokens, max_chunk_tokens=max_chunk_tokens)
        else:
            # For many chunks, sample a few for accurate verification and use fast for rest
            sample_size = min(3, len(chunks))
            sample_chunks = chunks[:sample_size]
            accurate_sample_tokens = [await estimate_tokens(chunk, self.model, use_fast=False) for chunk in sample_chunks]
            fast_chunk_tokens = [await estimate_tokens(chunk, self.model, use_fast=True) for chunk in chunks]
            avg_chunk_tokens = sum(fast_chunk_tokens) // len(chunks)
            max_chunk_tokens = max(accurate_sample_tokens)
            logger.info("Content chunked", num_chunks=len(chunks), avg_chunk_tokens=avg_chunk_tokens, max_chunk_sample=max_chunk_tokens)

        if len(chunks) > max_chunks:
            reason = "fast_mode" if getattr(self.config, "fast_mode", False) else "max_chunks cap"
            logger.warning(
                "Too many chunks, processing first N only",
                chunks=len(chunks), max_chunks=max_chunks, reason=reason,
            )
            # Stash the original count so summarize() can annotate the
            # final summary with the partial-coverage note. Survives
            # only as a transient attribute on this instance — fine
            # because Pipeline summarize is gated to concurrency=1.
            self._truncated_chunks_total = len(chunks)
            chunks = chunks[:max_chunks]

        return chunks

    def _combine_chunk_summaries(self, chunk_summaries: List[Dict[str, Any]]) -> tuple[str, List[str]]:
        """
        Combine summaries and keywords from multiple chunks.

        Args:
            chunk_summaries: List of summary dictionaries from chunks

        Returns:
            Tuple of (combined_summary, combined_keywords)
        """
        if not chunk_summaries:
            return "No content available for summarization.", []

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]["summary"], chunk_summaries[0]["keywords"]

        # Combine all chunk summaries. We keep the full concatenation for
        # embedding purposes — the embedder tokenizer will truncate at its
        # own max_seq_length if needed. Previously we replaced the joined
        # summary with a "Multi-part document covering: first-3-chunks..."
        # stub once the concatenation crossed 500 chars, which silently
        # dropped every chunk past index 2 from the text the embedder sees.
        # For a 10-chunk PDF, only 30% of the summarized content survived.
        summaries = [cs["summary"] for cs in chunk_summaries]
        combined_summary = " ".join(summaries)

        # Combine and deduplicate keywords
        all_keywords = []
        for cs in chunk_summaries:
            all_keywords.extend(cs["keywords"])

        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in all_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword_lower)

        # Limit to reasonable number of keywords
        combined_keywords = unique_keywords[:15]

        logger.info("Combined chunk summaries", num_chunks=len(chunk_summaries), final_keywords=len(combined_keywords))

        return combined_summary, combined_keywords

    def _parse_ai_response(self, response_content: str) -> tuple[str, str, List[str]]:
        """
        Parse AI response JSON to extract summary and keywords.

        Args:
            response_content: Raw response from AI model

        Returns:
            Tuple of (title, summary, keywords)

        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Strip invalid control characters that LLMs sometimes embed in string values
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', response_content.strip())
            data = json.loads(cleaned)

            title = data.get("title", "").strip()
            summary = data.get("summary", "").strip()
            keywords = data.get("keywords", [])

            # Fallback: treat "text" as an alias for "summary"
            if not summary:
                summary = data.get("text", "").strip()

            # Fallback: if the model returned structured sections instead of a
            # flat summary, synthesise a summary from the section names/items.
            if not summary and "sections" in data:
                sections = data["sections"]
                if isinstance(sections, list):
                    parts = []
                    for sec in sections:
                        if isinstance(sec, dict):
                            sec_name = sec.get("name", "")
                            items = sec.get("items", [])
                            item_names = [
                                it.get("name", "") for it in items
                                if isinstance(it, dict) and it.get("name")
                            ]
                            if sec_name and item_names:
                                parts.append(f"{sec_name} ({', '.join(item_names[:3])})")
                            elif sec_name:
                                parts.append(sec_name)
                    if parts:
                        summary = f"Covers {', '.join(parts[:4])}."
                        # Extract item names as keywords when none provided
                        if not keywords:
                            keywords = []
                            for sec in sections:
                                if isinstance(sec, dict):
                                    for it in sec.get("items", []):
                                        if isinstance(it, dict) and it.get("name"):
                                            keywords.append(it["name"])
                    logger.warning("Recovered summary from structured sections response",
                                   sections_count=len(sections))

            # Ensure keywords is a list of strings
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(kw).strip() for kw in keywords if str(kw).strip()]

            if not summary:
                logger.error("Response did not contain a valid summary", response=response_content)
                raise ValueError("Response did not contain a valid summary")

            return title, summary, keywords

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse AI response as JSON, attempting fallback extraction", error=str(e))

            # Fallback: try to find a JSON block within the response (e.g. ```json ... ```)
            json_match = re.search(r'\{[^{}]*"summary"\s*:\s*"[^"]+(?:"[^{}]*)\}', response_content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    title = data.get("title", "").strip()
                    summary = data.get("summary", "").strip()
                    keywords = data.get("keywords", [])
                    if summary:
                        if not isinstance(keywords, list):
                            keywords = []
                        keywords = [str(kw).strip() for kw in keywords if str(kw).strip()]
                        logger.info("Recovered summary from embedded JSON in response")
                        return title, summary, keywords
                except json.JSONDecodeError:
                    pass

            # Final fallback: use the plain text response as the summary directly
            plain_text = response_content.strip()
            # Remove common LLM preambles
            for prefix in ["Here is", "Here's", "I've summarized", "I cannot provide", "Please note"]:
                if plain_text.startswith(prefix):
                    break
            else:
                prefix = None

            if len(plain_text) >= 20 and len(plain_text) <= 5000:
                # Truncate to first ~3 sentences for a concise summary
                sentences = re.split(r'(?<=[.!?])\s+', plain_text)
                summary = " ".join(sentences[:3]).strip()
                if summary:
                    logger.info("Used plain text response as summary fallback", length=len(summary))
                    return "", summary, []

            logger.error("Failed to parse AI response as JSON and fallback failed", response=response_content[:200], error=str(e))
            raise ValueError(f"Invalid JSON response: {str(e)}")

    def _get_system_prompt(self, include_title: bool = False) -> str:
        # Prompt-engineering notes:
        #
        # 1. The system prompt is prefilled on every chunk. Every token
        #    here costs prefill time on every file. Stay terse.
        #
        # 2. The model is being used as a *retrieval indexer* — its
        #    summaries feed FTS5 and embedding-based search. Optimize
        #    for "would the user search for these words?" not for
        #    narrative quality. That's why the instructions say
        #    "concrete nouns" and "what someone would type", not
        #    "elegant prose".
        #
        # 3. The 1-shot example shows the EXACT output shape we want.
        #    Models follow shape examples more reliably than instructions.
        #    The example is short (avoids prefill bloat) but realistic
        #    (file-content-shaped).
        #
        # 4. Hard "ONLY JSON, no prose" line — open-ended models love
        #    to say "Here is the summary:" before the JSON. Specific
        #    forbidden phrases save retry rounds.
        #
        # 5. No `{{ }}` Jinja escapes — those weren't being templated,
        #    they were rendered literally to the model and confused it.
        if include_title:
            return (
                "You index file content for search. Output ONE JSON object, "
                "nothing else — no prose, no code fences, no 'Here is'.\n"
                "Schema:\n"
                '  {"title": "<1-5 word title>",\n'
                '   "summary": "<1-2 sentences naming the topic and key '
                'concrete nouns a searcher would type>",\n'
                '   "keywords": ["<5-12 distinctive nouns or noun-phrases, '
                'lowercase>"]}\n'
                "Example for a Q3 financial report:\n"
                '  {"title": "Q3 Earnings Report",\n'
                '   "summary": "Q3 2025 earnings: revenue up 12%, AWS '
                'driving cloud growth, margins steady.",\n'
                '   "keywords": ["q3 earnings", "revenue growth", "aws", '
                '"cloud", "margins"]}'
            )
        else:
            return (
                "You index file content for search. Output ONE JSON object, "
                "nothing else — no prose, no code fences, no 'Here is'.\n"
                "Schema:\n"
                '  {"summary": "<1-2 sentences naming the topic and key '
                'concrete nouns a searcher would type>",\n'
                '   "keywords": ["<5-12 distinctive nouns or noun-phrases, '
                'lowercase>"]}\n'
                "Example for a Q3 earnings continuation chunk:\n"
                '  {"summary": "Continued Q3 cloud-segment detail: AWS '
                'revenue $26B, operating margin 36%.",\n'
                '   "keywords": ["aws revenue", "cloud segment", '
                '"operating margin"]}'
            )

    def _format_content_with_context(self, chunk: str, chunk_num: int, total_chunks: int, filename: str) -> str:
        """Format content with filename and chunk context for the first chunk."""
        if chunk_num == 0:
            if total_chunks > 1:
                return f"Filename: {filename}\n(Part 1 of {total_chunks})\n\nContent:\n{chunk}"
            else:
                return f"Filename: {filename}\n\nContent:\n{chunk}"
        return chunk

    @abstractmethod
    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        raise NotImplementedError

    async def summarize(self, file_metadata: File) -> File:
        """Summarize file content with chunking support.

        Coverage policy: a per-file wall-clock budget caps how many
        chunks we'll attempt. When the budget elapses, we stop
        dispatching new chunks and finalize with whatever chunk
        summaries we already have. Files that would otherwise have
        failed at the queue's hard timeout (long PDFs that don't fit
        in the budget) end as COMPLETE with a partial-coverage flag
        in the summary, instead of FAILED. This is "good enough"
        coverage — better an indexed-but-partial file than nothing.
        Set `summarize_budget_seconds = 0` in SummarizerConfig to
        disable.
        """
        if not self._validate_content(file_metadata):
            return file_metadata

        logger.info("Summarizing", filename=file_metadata.filename, model=self.model, summarizer=self.__class__.__name__)

        try:
            # Prepare content chunks
            content_chunks = await self._prepare_content(file_metadata.content)
            chunk_summaries = []
            resolved_title = None

            images = await self._prepare_images(file_metadata)

            # Wall-clock budget. 0 (or negative) → no budget. Read off
            # the SummarizerConfig, default 60 s.
            budget = float(getattr(self.config, "summarize_budget_seconds", 60.0))
            t_start = time.monotonic()
            stopped_early = False
            chunks_attempted = 0

            # Process each chunk with per-chunk retry
            total_chunks = len(content_chunks)
            for i, chunk in enumerate(content_chunks):
                # Budget check BEFORE dispatching a new chunk. We always
                # let chunk 0 run so a single-chunk file still has a
                # chance even if the model is unusually slow that boot.
                elapsed = time.monotonic() - t_start
                if i > 0 and budget > 0 and elapsed >= budget:
                    logger.warning(
                        "Summarize budget elapsed; stopping early with partial coverage",
                        filename=file_metadata.filename,
                        elapsed_s=round(elapsed, 2),
                        budget_s=budget,
                        chunks_done=len(chunk_summaries),
                        total_chunks=total_chunks,
                    )
                    stopped_early = True
                    break

                chunks_attempted = i + 1
                logger.info(f"Processing chunk {i+1}/{total_chunks}", length=len(chunk), images=len(images))
                parsed = False
                for attempt in range(3):
                    response = await self._get_ai_response(chunk, i, total_chunks, images, file_metadata.filename)

                    if not response:
                        logger.warning("Empty response for chunk", chunk_num=i+1)
                        break

                    try:
                        title, summary, keywords = self._parse_ai_response(response)
                        chunk_summaries.append({"summary": summary, "keywords": keywords})
                        if i == 0 and title:
                            resolved_title = title
                        parsed = True
                        break
                    except ValueError:
                        if attempt < 2:
                            logger.warning("Retrying chunk", chunk_num=i+1, attempt=attempt+1)

                if not parsed:
                    logger.warning("Skipping chunk after retries", chunk_num=i+1)

            if not chunk_summaries:
                raise AIProviderError(f"No valid responses from {self.__class__.__name__}")

            # Combine chunk summaries
            final_summary, final_keywords = self._combine_chunk_summaries(chunk_summaries)

            # If we stopped early — either by exceeding the wall-clock
            # budget OR by being truncated upstream (max_chunks /
            # fast_mode) — annotate the summary so search results are
            # honest about coverage.
            truncated_total = getattr(self, "_truncated_chunks_total", None)
            self._truncated_chunks_total = None  # consume / reset
            if stopped_early and len(chunk_summaries) < total_chunks:
                final_summary = (
                    f"{final_summary} "
                    f"[partial: covered {len(chunk_summaries)} of "
                    f"{total_chunks} chunks before {budget:.0f}s budget]"
                )
            elif truncated_total and truncated_total > total_chunks:
                final_summary = (
                    f"{final_summary} "
                    f"[partial: covered first {total_chunks} of "
                    f"{truncated_total} chunks "
                    f"({'fast mode' if getattr(self.config, 'fast_mode', False) else 'max_chunks cap'})]"
                )

            # Update file metadata
            file_metadata.title = resolved_title
            file_metadata.summary = final_summary
            file_metadata.keywords = final_keywords
            file_metadata.status = ProcessingStatus.SUMMARIZED

            logger.info("Successfully summarized", filename=file_metadata.filename,
                        summarizer=self.__class__.__name__,
                        title=resolved_title,
                        summary_length=len(final_summary), keyword_count=len(final_keywords),
                        chunks_processed=len(chunk_summaries))

            return file_metadata

        except Exception as e:
            error_msg = f"{self.__class__.__name__} summarization failed: {str(e)}"
            logger.error("Summarization failed", summarizer=self.__class__.__name__, filename=file_metadata.filename, model=self.model, error=str(e))
            raise AIProviderError(error_msg)
