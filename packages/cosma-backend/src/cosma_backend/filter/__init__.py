"""
Filter Module

Provides file filtering functionality based on configurable patterns.
Supports both blacklist (exclude matching) and whitelist (only include matching) modes.
"""

from .filter_config import FilterConfig, FilterMode, FilterConfigManager

__all__ = ['FilterConfig', 'FilterMode', 'FilterConfigManager']
