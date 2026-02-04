#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   summarizer.py
@Time    :   2025/07/06 10:39:00
@Author  :   Ethan Pan 
@Version :   1.0
@Contact :   epan@cs.wisc.edu
@License :   (C)Copyright 2025, Ethan Pan
@Desc    :   None
'''


from __future__ import annotations

import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING

# Import AI libraries
from cosma_backend.utils.decorators import async_wrap
import litellm
import ollama
import tiktoken

from cosma_backend.models import ProcessingStatus
from cosma_backend.models.file import File
from cosma_backend.logging import get_logger

if TYPE_CHECKING:
    from cosma_backend.settings import SummarizerConfig

# Configure standard logger
logger = get_logger(__name__)


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """
    Get the tiktoken encoding for a given model name.
    
    Args:
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "qwen3-vl:2b-instruct")
        
    Returns:
        tiktoken.Encoding object for the model
    """
    # Try to get encoding directly from tiktoken for known models
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass
    
    # Handle common model families and aliases
    model_lower = model.lower()
    
    # OpenAI models
    if any(x in model_lower for x in ["gpt-4", "gpt-3.5", "gpt-35"]):
        return tiktoken.get_encoding("cl100k_base")
    elif "gpt-3" in model_lower or "davinci" in model_lower or "curie" in model_lower:
        return tiktoken.get_encoding("p50k_base")
    
    # Claude models use cl100k_base approximation
    elif "claude" in model_lower:
        return tiktoken.get_encoding("cl100k_base")
    
    # Gemini models use cl100k_base approximation
    elif "gemini" in model_lower:
        return tiktoken.get_encoding("cl100k_base")
    
    # Llama models (including Ollama) - use cl100k_base as approximation
    elif any(x in model_lower for x in ["llama", "mistral", "mixtral", "phi", "qwen", "gemma", "deepseek"]):
        return tiktoken.get_encoding("cl100k_base")
    
    # Default to cl100k_base for unknown models (GPT-4 tokenizer)
    logger.debug(f"Unknown model '{model}', defaulting to cl100k_base encoding")
    return tiktoken.get_encoding("cl100k_base")


def estimate_tokens_fast(text: str, model: Optional[str] = None) -> int:
    """
    Fast token estimation using length-based heuristics.
    Much faster than tiktoken but less accurate.
    
    Args:
        text: The text to estimate tokens for
        model: Optional model name (not used in fast estimation)
    
    Returns:
        Estimated number of tokens in the text
    """
    return len(text) // 4


@async_wrap  # slow and blocking
def estimate_tokens(text: str, model: Optional[str] = None, use_fast: bool = False) -> int:
    """
    Estimate the number of tokens in a text string.
    
    Args:
        text: The text to tokenize
        model: Optional model name to get the correct encoding. If not provided,
               uses cl100k_base (GPT-4 tokenizer) as default.
        use_fast: Use fast character-based estimation instead of tiktoken
    
    Returns:
        Number of tokens in the text
    """
    if use_fast or not text:
        return estimate_tokens_fast(text, model)
    
    try:
        if model:
            encoding = get_encoding_for_model(model)
        else:
            # Default to cl100k_base (used by GPT-4, GPT-3.5-turbo, etc.)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error using tiktoken: {e}, falling back to fast estimation")
        # Fallback to fast estimation
        return estimate_tokens_fast(text, model)


async def chunk_content(content: str, max_tokens: int, overlap_tokens: int = 50, model: Optional[str] = None) -> List[str]:
    """
    Split content into chunks that fit within token limits.
    Uses fast token estimation for efficiency with accuracy validation.
    
    Args:
        content: The text content to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        model: Optional model name for accurate tokenization
        
    Returns:
        List of content chunks
    """
    # Fast initial check
    if await estimate_tokens(content, model, use_fast=True) <= max_tokens:
        # Verify with accurate tokenization if it's close to the limit
        if await estimate_tokens(content, model, use_fast=False) <= max_tokens:
            return [content]
    
    # Use sentence-based chunking with fast estimation for efficiency
    sentences = content.split('. ')
    chunks = []
    current_chunk = []
    current_tokens = 0
    safety_buffer = int(max_tokens * 0.1)  # 10% safety buffer
    
    for sentence in sentences:
        sentence_tokens = await estimate_tokens(sentence, model, use_fast=True)
        
        if sentence_tokens > (max_tokens - safety_buffer):
            logger.info("Sentence too big", sentence_tokens=sentence_tokens, current_tokens=current_tokens, max=max_tokens - safety_buffer)
            continue
        
        if current_tokens + sentence_tokens > (max_tokens - safety_buffer) and current_chunk:
            logger.info("Chunk created", chunk=len(chunks) + 1, tokens=current_tokens)
            # Finalize current chunk and verify it's within limits
            chunk_text = '. '.join(current_chunk) + '.'
            
            # Safety check: verify the chunk doesn't exceed the limit with accurate tokenization
            accurate_tokens = await estimate_tokens(chunk_text, model, use_fast=False)
            if accurate_tokens > max_tokens:
                # Chunk is too large, split it further
                chunk_text = await _oversized_chunk_fix(chunk_text, max_tokens, model)
            
            chunks.append(chunk_text)
            
            # Start new chunk with overlap (fast estimation)
            overlap_sentences = max(1, overlap_tokens // 50)  # Rough overlap in sentences
            overlap_content = '. '.join(current_chunk[-overlap_sentences:])
            current_chunk = [overlap_content, sentence] if overlap_content else [sentence]
            current_tokens = await estimate_tokens('. '.join(current_chunk), model, use_fast=True)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk with safety check
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        accurate_tokens = await estimate_tokens(chunk_text, model, use_fast=False)
        if accurate_tokens > max_tokens:
            chunk_text = await _oversized_chunk_fix(chunk_text, max_tokens, model)
        chunks.append(chunk_text)
    
    return chunks


async def _oversized_chunk_fix(chunk_text: str, max_tokens: int, model: Optional[str] = None) -> str:
    """
    Fix an oversized chunk by splitting it more aggressively.
    Uses accurate tokenization for this critical operation.
    
    Args:
        chunk_text: The oversized chunk text
        max_tokens: Maximum allowed tokens
        model: Model name for tokenization
        
    Returns:
        Fixed chunk text within token limits
    """
    # If the chunk is still too large, split by paragraphs then by character count
    paragraphs = chunk_text.split('\n\n')
    if len(paragraphs) > 1:
        # Try including paragraphs one by one
        result_chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            if await estimate_tokens(test_chunk, model, use_fast=False) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    result_chunks.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:
            result_chunks.append(current_chunk)
        
        # Return the first chunk that fits
        return result_chunks[0] if result_chunks else chunk_text[:max_tokens * 4]  # Rough character fallback
    
    # Last resort: character-based splitting
    # Estimate characters needed (roughly 4 chars per token)
    max_chars = max_tokens * 4
    if len(chunk_text) <= max_chars:
        return chunk_text
    
    # Find a good breaking point near the limit
    break_point = max_chars
    # Try to break at sentence boundary
    for i in range(min(break_point, len(chunk_text)), max(0, break_point - 200), -1):
        if chunk_text[i] == '.' and i + 1 < len(chunk_text) and chunk_text[i + 1] == ' ':
            return chunk_text[:i + 1]
    
    # Fallback to hard character limit
    return chunk_text[:max_chars]
    
def extract_json_from_response(content: str):
    """Extract JSON from Gemma 3 response (handles markdown code fences)"""
    
    # Try to find JSON in code fence
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()
    
    return json_str


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
        images = []
        
        if file_metadata.content_type.startswith("image"):
            with open(file_metadata.path, 'rb') as f:
                images.append(base64.b64encode(f.read()).decode('utf-8'))
                
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
        
        # Content is too large, need to chunk
        if estimated_tokens >= 200_000:
            logger.warning("Content exceeds max token limit, will not summarize", estimated_tokens=estimated_tokens)
            raise RuntimeError("File too large to summarize")
        
        logger.info("Content exceeds token limit, chunking required", estimated_tokens=estimated_tokens, max_tokens=self.max_tokens)
        
        # Use optimized chunking
        chunks = await chunk_content(content, self.max_tokens, self.chunk_overlap, self.model)
        logger.info("Content chunked (noverify)", num_chunks=len(chunks))

        max_chunks = self.config.max_chunks

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
            logger.warning("Too many chunks, will not summarize", chunks=len(chunks), max_chunks=max_chunks)
            raise RuntimeError("Too many chunks to summarize")
        
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
        
        # Combine summaries
        summaries = [cs["summary"] for cs in chunk_summaries]
        combined_summary = " ".join(summaries)
        
        # If combined summary is too long, summarize it again
        if len(combined_summary) > 500:  # Rough character limit
            combined_summary = f"Multi-part document covering: {'; '.join(summaries[:3])}"
            if len(summaries) > 3:
                combined_summary += f" and {len(summaries) - 3} additional topics."
        
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
            Tuple of (summary, keywords)
            
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
            
    def _get_system_prompt(self, include_title: bool = False):
        if include_title:
            return (
                "You are a concise summarization assistant. "
                "**Return valid JSON only** with keys `title`, `summary`, and `keywords` (array). "
                "Title should be an extremely concise, 1-5 word title for the content. "
                "Summary should be 1-2 sentences capturing the main topic and key points. "
                "Keywords should be 5-12 relevant nouns or noun-phrases that describe the content."
                "Example: {{'title': 'Proper Title', 'summary': 'A concise summary of the file content', 'keywords': ['keyword1', 'keyword2', 'keyword3']}}"
            )
        else:
            return (
                "You are a concise summarization assistant. "
                "**Return valid JSON only** with keys `summary` and `keywords` (array). "
                "Summary should be 1-2 sentences capturing the main topic and key points. "
                "Keywords should be 5-12 relevant nouns or noun-phrases that describe the content."
                "Example: {{'summary': 'A concise summary of the file content', 'keywords': ['keyword1', 'keyword2', 'keyword3']}}"
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
        """Summarize using Ollama with chunking support."""
        if not self._validate_content(file_metadata):
            return file_metadata
        
        logger.info("Summarizing with Ollama", filename=file_metadata.filename, model=self.model)
        
        try:
            # Prepare content chunks
            content_chunks = await self._prepare_content(file_metadata.content)
            chunk_summaries = []
            resolved_title = None
            
            images = await self._prepare_images(file_metadata)
            
            # Process each chunk with per-chunk retry
            total_chunks = len(content_chunks)
            for i, chunk in enumerate(content_chunks):
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


class OllamaSummarizer(BaseSummarizer):
    """Summarizer using local Ollama models."""
    
    def __init__(self, config: SummarizerConfig | None = None, host: Optional[str] = None, model: Optional[str] = None, max_tokens: Optional[int] = None):
        """
        Initialize Ollama summarizer.

        Args:
            config: SummarizerConfig instance
            host: Ollama host URL override
            model: Model name override
            max_tokens: Maximum context tokens override
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()

        # Get model name before initializing base class
        model_name = model or self.config.ollama.model

        # Initialize base class with context length and model
        context_length = max_tokens or self.config.ollama.context_length
        super().__init__(config=self.config, max_tokens=context_length, model=model_name)

        try:
            import ollama
            self.ollama_available = True
        except ImportError:
            self.ollama_available = False
            raise ImportError("ollama package is not installed")

        self.host = host or self.config.ollama.host
        
        self._last_used_at: float = 0.0
        self._model_loaded: bool = False

        try:
            self.client = ollama.AsyncClient(host=self.host)
            logger.info("Ollama summarizer initialized", host=self.host, model=self.model, max_tokens=self.max_tokens)
        except Exception as e:
            logger.error("Failed to initialize Ollama client", host=self.host, error=str(e))
            raise AIProviderError(f"Failed to initialize Ollama: {str(e)}")

    @property
    def last_used_at(self) -> float:
        return self._last_used_at

    async def unload(self) -> None:
        """Tell Ollama to unload the model from GPU/memory."""
        try:
            await self.client.generate(model=self.model, keep_alive="0")
            self._model_loaded = False
            logger.info("Ollama model unloaded", model=self.model)
        except Exception as e:
            logger.warning("Failed to unload Ollama model", model=self.model, error=str(e))

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            # Try to list models to check if Ollama is running
            list = await self.client.list()
            if self.model and self.model not in (m.model for m in list.models):
                logger.info("Ollama model not found, pulling", model=self.model)
                await self.client.pull(self.model)
            self._model_loaded = True
            return True
        except Exception as e:
            logger.debug(f"Ollama not available - error: {str(e)}")
            return False
            
    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        content = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        user_message: dict[str, Any] = {"role": "user", "content": content}
        if images:
            user_message["images"] = images

        raw_response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt(include_title=(chunk_num == 0))},
                user_message,
            ],
            think=False,
            # format="json",
            options=ollama.Options(
                num_predict=500,
                # temperature=0.3,
                num_ctx=16_000,
            )
        )

        self._last_used_at = time.time()
        self._model_loaded = True
        logger.info("AI response", summarizer=self.__class__.__name__, response=raw_response)
        return extract_json_from_response(raw_response['message']['content'])


class OnlineSummarizer(BaseSummarizer):
    """Summarizer using online AI models via LiteLLM."""
    
    def __init__(self, config: SummarizerConfig | None = None, model: Optional[str] = None, api_key: Optional[str] = None, max_tokens: Optional[int] = None):
        """
        Initialize online summarizer.

        Args:
            config: SummarizerConfig instance
            model: Model name override
            api_key: API key override
            max_tokens: Maximum context tokens override
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()

        # Get model name before initializing base class
        model_name = model or self.config.online.model

        # Initialize base class with context length and model
        context_length = max_tokens or self.config.online.context_length
        super().__init__(config=self.config, max_tokens=context_length, model=model_name)
        
        try:
            import litellm
            self.litellm_available = True
        except ImportError:
            self.litellm_available = False
            raise ImportError("litellm package is not installed")
        
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        logger.info("Online summarizer initialized", model=self.model, max_tokens=self.max_tokens)
    
    async def is_available(self) -> bool:
        """Check if online models are available."""
        # Check for required API keys based on model
        if self.model.startswith("gpt-") or self.model.startswith("o1-"):
            return bool(os.getenv("OPENAI_API_KEY"))
        elif self.model.startswith("claude-"):
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        elif self.model.startswith("gemini-"):
            return bool(os.getenv("GOOGLE_API_KEY"))
        else:
            # Assume OpenAI by default
            return bool(os.getenv("OPENAI_API_KEY"))
            
    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        content = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        user_message = {"role": "user", "content": content}
        if images:
            user_message["images"] = images

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt(include_title=(chunk_num == 0))},
                user_message,
            ],
            temperature=0.1,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
            timeout=120,
            max_retries=2,
        )
        
        return response.choices[0].message.content


class LlamaCppSummarizer(BaseSummarizer):
    """Summarizer using local llama.cpp models."""
    
    def __init__(self, config: SummarizerConfig | None = None, model_path: Optional[str] = None, max_tokens: Optional[int] = None, n_ctx: Optional[int] = None):
        """
        Initialize llama.cpp summarizer.

        Args:
            config: SummarizerConfig instance
            model_path: Path to GGUF model file override
            max_tokens: Maximum context tokens override
            n_ctx: Context window size override
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()

        # Get model path from config if not provided
        self.model_path = model_path or self.config.llamacpp.model_path
        self.repo_id = self.config.llamacpp.repo_id
        self.filename = self.config.llamacpp.filename
        if not (self.model_path or all((self.repo_id, self.filename))):
            raise ValueError("LLAMACPP_MODEL_PATH environment variable must be set or model_path must be provided")

        # Initialize base class with context length
        context_length = max_tokens or self.config.llamacpp.context_length
        super().__init__(config=self.config, max_tokens=context_length, model="llama.cpp")

        self.n_ctx = n_ctx or self.config.llamacpp.n_ctx

        try:
            from llama_cpp import Llama
            self.llamacpp_available = True
        except ImportError:
            self.llamacpp_available = False
            raise ImportError("llama-cpp-python package is not installed. Install with: pip install llama-cpp-python")

        try:
            # Initialize llama.cpp model
            if self.repo_id and self.filename:
                self.llm = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    n_ctx=self.n_ctx,
                    n_threads=self.config.llamacpp.n_threads,
                    n_gpu_layers=self.config.llamacpp.n_gpu_layers,
                    verbose=self.config.llamacpp.verbose,
                )
                logger.info("llama.cpp summarizer initialized",
                            repo_id=self.repo_id,
                            filename=self.filename,
                            n_ctx=self.n_ctx,
                            max_tokens=self.max_tokens)
            else:
                self.llm = Llama(
                    repo_id=self.repo_id,
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.config.llamacpp.n_threads,
                    n_gpu_layers=self.config.llamacpp.n_gpu_layers,
                    verbose=self.config.llamacpp.verbose,
                )
                logger.info("llama.cpp summarizer initialized",
                            model_path=self.model_path,
                            n_ctx=self.n_ctx,
                            max_tokens=self.max_tokens)
        except Exception as e:
            logger.error("Failed to initialize llama.cpp model", model_path=self.model_path, error=str(e))
            raise AIProviderError(f"Failed to initialize llama.cpp: {str(e)}")
    
    async def is_available(self) -> bool:
        """Check if llama.cpp is available."""
        try:
            # Check if model file exists and is accessible
            # if not self.model_path or not os.path.exists(self.model_path):
            #     logger.debug(f"llama.cpp model not found at path: {self.model_path}")
            #     return False
            
            # Check if llm is initialized
            return hasattr(self, 'llm') and self.llm is not None
        except Exception as e:
            logger.debug(f"llama.cpp not available - error: {str(e)}")
            return False
            
    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        content = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self._get_system_prompt(include_title=(chunk_num == 0))},
                {"role": "user", "content": content},
            ],
            max_tokens=500,
            temperature=0.1,
            top_p=0.95,
            stream=False,
        )
        
        response_content = response['choices'][0]['message']['content'].strip()
        logger.info("llama.cpp raw response", response=response_content)
        return extract_json_from_response(response_content)


class AutoSummarizer:
    """
    Automatic summarizer that selects the best available provider.
    
    Tries providers in order of preference:
    1. User-specified provider
    2. Local llama.cpp (fastest local inference)
    3. Local Ollama (privacy-focused)
    4. Online models (fallback)
    """
    
    def __init__(self, config: SummarizerConfig | None = None, preferred_provider: Optional[str] = None):
        """
        Initialize auto summarizer.

        Args:
            config: SummarizerConfig instance
            preferred_provider: Preferred provider ('llamacpp', 'ollama', 'online', 'auto')
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()
        self.preferred_provider = preferred_provider or self.config.provider
        self.summarizers = {}
        
        logger.info("AutoSummarizer initialized", preferred_provider=self.preferred_provider)
    
    async def _get_llamacpp_summarizer(self) -> Optional[LlamaCppSummarizer]:
        """Get or create llama.cpp summarizer if available."""
        if "llamacpp" not in self.summarizers:
            try:
                logger.info("llama.cpp summarizer initalizing")
                summarizer = LlamaCppSummarizer(config=self.config)
                if await summarizer.is_available():
                    self.summarizers["llamacpp"] = summarizer
                    logger.info("llama.cpp summarizer available")
                else:
                    self.summarizers["llamacpp"] = None
                    logger.debug("llama.cpp summarizer not available")
                    return None
            except Exception as e:
                logger.warning("Failed to create llama.cpp summarizer", error=str(e))
                self.summarizers["llamacpp"] = None
                return None
        
        return self.summarizers.get("llamacpp")
    
    async def _get_ollama_summarizer(self) -> Optional[OllamaSummarizer]:
        """Get or create Ollama summarizer if available."""
        if "ollama" not in self.summarizers:
            try:
                summarizer = OllamaSummarizer(config=self.config)
                if await summarizer.is_available():
                    self.summarizers["ollama"] = summarizer
                    logger.info("Ollama summarizer available")
                else:
                    logger.debug("Ollama summarizer not available")
                    self.summarizers["ollama"] = None
                    return None
            except Exception as e:
                logger.debug(f"Failed to create Ollama summarizer - error: {str(e)}")
                self.summarizers["ollama"] = None
                return None
        
        return self.summarizers.get("ollama")
    
    async def _get_online_summarizer(self) -> Optional[OnlineSummarizer]:
        """Get or create online summarizer if available."""
        if "online" not in self.summarizers:
            try:
                summarizer = OnlineSummarizer(config=self.config)
                if await summarizer.is_available():
                    self.summarizers["online"] = summarizer
                    logger.info("Online summarizer available")
                else:
                    logger.debug("Online summarizer not available")
                    self.summarizers["online"] = None
                    return None
            except Exception as e:
                logger.debug(f"Failed to create online summarizer - error: {str(e)}")
                self.summarizers["online"] = None
                return None
        
        return self.summarizers.get("online")
    
    async def summarize(self, file_metadata: File) -> File:
        """
        Summarize using the best available provider with fallback.
        
        Args:
            file_metadata: File metadata to summarize
            
        Returns:
            Enhanced file metadata with summary and keywords
            
        Raises:
            SummarizerError: If no summarizers are available or all fail
        """
        # Get all available providers in priority order
        providers = [
            await self._get_llamacpp_summarizer(),
            await self._get_ollama_summarizer(),
            await self._get_online_summarizer()
        ]
        
        # Sort providers based on preference
        if self.preferred_provider == "online":
            providers.reverse()
        elif self.preferred_provider == "ollama":
            # Move Ollama to front
            providers = [p for p in providers if isinstance(p, OllamaSummarizer)] + \
                       [p for p in providers if not isinstance(p, OllamaSummarizer)]
        elif self.preferred_provider == "llamacpp":
            # Move llama.cpp to front (already first by default)
            pass

        summarizer = None
        for provider in providers:
            if provider and await provider.is_available():
                summarizer = provider
                try:
                    logger.info("Attempting summarization", provider=type(summarizer).__name__)
                    return await summarizer.summarize(file_metadata)
                except Exception as e:
                    logger.warning("Summarizer failed, trying next provider", provider=type(summarizer).__name__, error=str(e))
                    continue # Try next provider
        
        error_msg = "All AI summarizers failed or are unavailable"
        logger.error("All AI summarizers failed or are unavailable", preferred_provider=self.preferred_provider)
        raise SummarizerError(error_msg)
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        providers = []

        if await self._get_llamacpp_summarizer():
            providers.append("llamacpp")

        if await self._get_ollama_summarizer():
            providers.append("ollama")

        if await self._get_online_summarizer():
            providers.append("online")

        return providers

    @property
    def last_used_at(self) -> float:
        """Return the most recent last_used_at across all active summarizers."""
        latest = 0.0
        for summarizer in self.summarizers.values():
            if summarizer is not None and hasattr(summarizer, "last_used_at"):
                latest = max(latest, summarizer.last_used_at)
        return latest

    async def unload_models(self) -> None:
        """Unload models from memory for all providers that support it."""
        for name, summarizer in self.summarizers.items():
            if summarizer is not None and hasattr(summarizer, "unload"):
                await summarizer.unload()
                logger.info("Unloaded model", provider=name)


# Convenience functions for easier usage
async def summarize_file(file_metadata: File, 
                  provider: Optional[str] = None) -> File:
    """
    Convenience function to summarize a file.
    
    Args:
        file_metadata: File metadata to summarize
        provider: Preferred AI provider
        
    Returns:
        Enhanced file metadata with summary and keywords
    """
    summarizer = AutoSummarizer(preferred_provider=provider)
    return await summarizer.summarize(file_metadata)


async def get_available_providers() -> List[str]:
    """Get list of available AI providers."""
    summarizer = AutoSummarizer()
    return await summarizer.get_available_providers()


async def is_summarizer_available() -> bool:
    """Check if any summarizer is available."""
    try:
        providers = await get_available_providers()
        return len(providers) > 0
    except Exception:
        return False
