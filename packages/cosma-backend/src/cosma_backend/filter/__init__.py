"""
Filter Module

Provides file filtering functionality based on configurable patterns.
Supports both blacklist (exclude matching) and whitelist (only include
matching) modes, plus a metadata-only third tier (filename embedding,
no LLM summary) for power users.
"""

from .filter_config import (
    FilterConfig,
    FilterConfigManager,
    FilterDecision,
    FilterMode,
)

__all__ = ['FilterConfig', 'FilterConfigManager', 'FilterDecision', 'FilterMode']
