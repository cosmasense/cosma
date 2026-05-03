"""
Token estimation and content chunking utilities.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, List, Optional

import tiktoken

from cosma_backend.logging import get_logger
from cosma_backend.utils.decorators import async_wrap

logger = get_logger(__name__)


# Sentence-splitting machinery. Used by chunk_content to break content
# on real sentence boundaries instead of the previous brittle
# `content.split('. ')`. Documented at the call site.

# Common abbreviations whose trailing period must NOT end a sentence.
# Keep this conservative — a missing entry just means an extra split,
# not a wrong one.
_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
    "vs", "etc", "e.g", "i.e", "Inc", "Ltd", "Co", "Corp",
    "U.S", "U.K", "E.U", "U.N",
    "St", "Ave", "Rd", "Blvd",
    "approx", "Fig", "fig", "No", "no",
    "Ph.D", "M.D", "B.A", "M.A", "B.S", "M.S",
    "a.m", "p.m", "A.M", "P.M",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept",
    "Oct", "Nov", "Dec",
)

# Placeholder used to mask periods inside abbreviations + decimals so
# the actual sentence splitter doesn't see them. Chosen to be a
# string that won't appear in real text. Restored at the end.
_DOT_PLACEHOLDER = "\x00DOT\x00"

# Pre-built alternation of abbreviations with a literal `.`. Used to
# mask "Mr." → "Mr<placeholder>" before splitting, then restored.
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.",
)
# Decimal numbers like 3.14, $12.50, v3.14. Mask the period so we
# don't split mid-number.
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")

# Sentence-final punctuation followed by whitespace and a likely
# next-sentence start. Latin uppercase OR CJK OR opening quote/paren.
# Python's `re` doesn't allow variable-width lookbehind; use lookahead
# only and rely on the abbreviation/decimal masking to keep us out of
# the wrong places.
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?。!?])\s+(?=[A-Z一-鿿ぁ-ヿ\(\[\"'])",
)

# Paragraph breaks always count.
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


def _split_into_sentences(text: str) -> List[str]:
    """Robust-enough sentence splitter.

    Beats `text.split('. ')` on:
      - "Mr. Smith asked Dr. Wong." → 1 sentence, not 2
      - "Use v3.14 of pi." → 1 sentence, not 3
      - "First. Second!" → 2 sentences
      - Multi-paragraph text → split on \\n\\n boundaries too
      - Chinese / Japanese punctuation supported

    Implementation: mask abbreviation- and decimal-internal periods
    with a placeholder, split on real sentence boundaries, then
    restore the periods. Avoids Python's no-variable-width-lookbehind
    limitation that bit the first version of this code.
    """
    if not text:
        return []

    # First split on paragraph breaks — unambiguous boundaries that
    # don't depend on punctuation (markdown headings, logs).
    paragraphs = _PARAGRAPH_BREAK_RE.split(text)
    sentences: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Mask periods inside abbreviations and decimals so the
        # sentence-end regex won't trip on them.
        masked = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT_PLACEHOLDER, para)
        masked = _DECIMAL_RE.sub(lambda m: m.group(1) + _DOT_PLACEHOLDER + m.group(2), masked)

        parts = _SENTENCE_END_RE.split(masked)
        for p in parts:
            p = p.replace(_DOT_PLACEHOLDER, ".").strip()
            if p:
                sentences.append(p)
    return sentences


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """
    Get a tiktoken encoding for OpenAI-family model names.

    Kept for backward compatibility and for online (OpenAI-compatible)
    providers. For non-OpenAI models (Qwen, Llama, Gemma, ...) callers
    should prefer ``_get_token_counter`` which may return a proper HF
    tokenizer keyed off an HF repo id ("owner/repo").
    """
    # Try to get encoding directly from tiktoken for known models
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass

    model_lower = model.lower()

    if any(x in model_lower for x in ["gpt-4", "gpt-3.5", "gpt-35"]):
        return tiktoken.get_encoding("cl100k_base")
    elif "gpt-3" in model_lower or "davinci" in model_lower or "curie" in model_lower:
        return tiktoken.get_encoding("p50k_base")
    elif "claude" in model_lower or "gemini" in model_lower:
        return tiktoken.get_encoding("cl100k_base")

    logger.debug(f"Unknown model '{model}', defaulting to cl100k_base encoding")
    return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=8)
def _load_hf_tokenizer(repo_id: str):
    """Load and cache an HF tokenizer. Returns None on any failure so the
    caller can fall back to tiktoken/heuristic counting without crashing.

    The tokenizer download is tiny (~5 MB) compared to the GGUF weights,
    and `transformers` is already pulled in by sentence-transformers.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=False)
        logger.info("Loaded HF tokenizer for token counting",
                    repo_id=repo_id, vocab_size=tok.vocab_size)
        return tok
    except Exception as e:
        logger.warning("Failed to load HF tokenizer, will fall back",
                       repo_id=repo_id, error=str(e))
        return None


def _get_token_counter(model: Optional[str]) -> Callable[[str], int]:
    """Return a callable that counts tokens for `model`.

    Resolution order:
      1. `model` looks like an HF repo ("owner/repo"): try AutoTokenizer.
      2. Otherwise: tiktoken encoding_for_model / family heuristic.
      3. On any failure: 4-chars-per-token heuristic.

    The GGUF repos the bootstrap pulls (e.g. unsloth/*-GGUF) don't carry
    a usable `config.json` for AutoTokenizer, so callers should pass the
    upstream tokenizer repo (see `LlamaCppConfig.tokenizer_repo`) rather
    than the GGUF repo.
    """
    if model and "/" in model:
        tok = _load_hf_tokenizer(model)
        if tok is not None:
            def _count(text: str, _tok=tok) -> int:
                return len(_tok.encode(text, add_special_tokens=False))
            return _count

    try:
        enc = get_encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except Exception as e:
        logger.warning("Token counter fallback to char heuristic", error=str(e))
        return lambda text: len(text) // 4


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
        return _get_token_counter(model)(text)
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}, falling back to fast estimation")
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

    # Use sentence-based chunking with fast estimation for efficiency.
    # The previous version used `content.split('. ')`, which:
    #   * mis-splits "Mr. Smith", "U.S.A.", "v3.14" into pseudo-sentences
    #   * loses every legitimate sentence-final period (the trailing
    #     ". " was the delimiter so it gets stripped)
    #   * does nothing with "?" / "!" / Asian punctuation / line breaks
    # The split below uses a regex with negative lookbehinds for the
    # most common abbreviations and decimal patterns, splits on real
    # sentence-final punctuation followed by whitespace, AND keeps the
    # delimiter on the previous sentence so reconstructed chunks read
    # naturally.
    sentences = _split_into_sentences(content)
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
            # Sentences already carry their terminators (the new
            # _split_into_sentences keeps them), so reassembly is a
            # plain ' ' join instead of '. '+suffix.
            chunk_text = ' '.join(current_chunk)

            # Safety check: verify the chunk doesn't exceed the limit with accurate tokenization
            accurate_tokens = await estimate_tokens(chunk_text, model, use_fast=False)
            if accurate_tokens > max_tokens:
                # Chunk is too large, split it further
                chunk_text = await _oversized_chunk_fix(chunk_text, max_tokens, model)

            chunks.append(chunk_text)

            # Start new chunk with overlap (fast estimation)
            overlap_sentences = max(1, overlap_tokens // 50)  # Rough overlap in sentences
            overlap_content = ' '.join(current_chunk[-overlap_sentences:])
            current_chunk = [overlap_content, sentence] if overlap_content else [sentence]
            current_tokens = await estimate_tokens(' '.join(current_chunk), model, use_fast=True)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    # Add final chunk with safety check
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        accurate_tokens = await estimate_tokens(chunk_text, model, use_fast=False)
        if accurate_tokens > max_tokens:
            chunk_text = await _oversized_chunk_fix(chunk_text, max_tokens, model)
        chunks.append(chunk_text)

    # Fallback: if sentence-based splitting produced no chunks (e.g. content
    # with no ". " sequences — raw markdown, PDF-extracted text without
    # sentence delimiters, log files), break the content by character
    # ranges guaranteed to fit ``max_tokens``. Without this, callers treated
    # the original content as a single chunk and sent it to the model, which
    # then blew past the context window.
    if not chunks:
        logger.warning("Sentence-based chunking produced no chunks; "
                       "falling back to character-based splitting",
                       content_length=len(content), max_tokens=max_tokens)
        chunks = await _character_split(content, max_tokens, model)

    return chunks


async def _character_split(content: str, max_tokens: int, model: Optional[str] = None) -> List[str]:
    """Split by character ranges, verifying each piece against ``max_tokens``.

    Used as a last resort when sentence-based chunking yields nothing.
    Uses a conservative ~3 chars/token estimate and then trims any chunk
    that still measures over the limit.
    """
    approx_chars_per_chunk = max(1, max_tokens * 3)
    chunks: List[str] = []
    pos = 0
    while pos < len(content):
        end = min(pos + approx_chars_per_chunk, len(content))
        piece = content[pos:end]
        # Shrink until the piece genuinely fits — the 3 chars/token estimate
        # understates tokens for CJK / base64 / code.
        while piece and await estimate_tokens(piece, model, use_fast=False) > max_tokens:
            piece = piece[: max(1, int(len(piece) * 0.8))]
        if piece:
            chunks.append(piece)
            pos += len(piece)
        else:
            # Safety: avoid infinite loop if tokenization keeps rejecting.
            break
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
