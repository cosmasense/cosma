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
from .tier_rules import (
    DEFAULT_LARGE_FILE_DOWNGRADE_MB,
    DEFAULT_TIER_RULES,
    Tier,
    TierRule,
    default_tier_rules,
)

__all__ = [
    'DEFAULT_LARGE_FILE_DOWNGRADE_MB',
    'DEFAULT_TIER_RULES',
    'FilterConfig',
    'FilterConfigManager',
    'FilterDecision',
    'FilterMode',
    'Tier',
    'TierRule',
    'default_tier_rules',
]
