"""
FTS5 query parsing and sanitization for safe full-text search queries.
"""

import re

# FTS5 special characters that need escaping
FTS5_SPECIAL_CHARS = set('"*():^')


def escape_fts5_token(token: str) -> str:
    """
    Escape special FTS5 characters in a token.

    FTS5 treats the following as special: " * ( ) : ^
    We escape by wrapping the token in double quotes and escaping internal quotes.

    Args:
        token: Raw search token

    Returns:
        Escaped token safe for FTS5 queries
    """
    if not token:
        return ""

    # Escape internal double quotes by doubling them
    escaped = token.replace('"', '""')

    # Wrap in double quotes to treat as literal
    return f'"{escaped}"'


def parse_fts5_query(query: str, allow_operators: bool = False) -> str:
    """
    Parse user query into safe FTS5 syntax.

    By default, treats all input as literal search terms (no operators).
    When allow_operators=True, preserves AND, OR, NOT operators.

    Args:
        query: Raw user search query
        allow_operators: If True, preserve boolean operators

    Returns:
        Sanitized FTS5 query string
    """
    if not query or not query.strip():
        return ""

    query = query.strip()

    if allow_operators:
        return _parse_with_operators(query)
    else:
        return _parse_literal(query)


def _parse_literal(query: str) -> str:
    """
    Parse query as literal search terms, escaping all special characters.

    Splits on whitespace and treats each word as a literal token.
    """
    # Split on whitespace
    tokens = query.split()

    if not tokens:
        return ""

    # Escape each token and join with spaces (implicit AND in FTS5)
    escaped_tokens = [escape_fts5_token(token) for token in tokens if token]

    return " ".join(escaped_tokens)


def _parse_with_operators(query: str) -> str:
    """
    Parse query preserving AND, OR, NOT operators.

    Operators must be uppercase to be recognized.
    Non-operator terms are escaped.
    """
    # FTS5 boolean operators (case-sensitive, must be uppercase)
    operators = {"AND", "OR", "NOT"}

    # Simple tokenization - split on whitespace while preserving operators
    tokens = query.split()

    if not tokens:
        return ""

    result_parts = []

    for token in tokens:
        if token in operators:
            result_parts.append(token)
        else:
            result_parts.append(escape_fts5_token(token))

    return " ".join(result_parts)


def build_prefix_query(prefix: str) -> str:
    """
    Build an FTS5 prefix search query.

    Args:
        prefix: Search prefix

    Returns:
        FTS5 query for prefix matching (e.g., 'test*')
    """
    if not prefix or not prefix.strip():
        return ""

    # Clean the prefix - remove special chars and whitespace
    cleaned = re.sub(r'["\*\(\)\:\^\s]+', "", prefix.strip())

    if not cleaned:
        return ""

    # FTS5 prefix search uses * suffix
    return f'"{cleaned}"*'
