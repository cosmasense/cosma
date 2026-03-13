"""
Token estimation and content chunking utilities.
"""

from __future__ import annotations

import re
from typing import List, Optional

import tiktoken

from cosma_backend.logging import get_logger
from cosma_backend.utils.decorators import async_wrap

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


def extract_json_from_response(content: str) -> str:
    """Extract JSON from LLM response (handles markdown code fences)."""
    # Try to find JSON in code fence
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()

    return json_str
