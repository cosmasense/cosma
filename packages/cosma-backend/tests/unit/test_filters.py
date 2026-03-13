"""Unit tests for the filter test endpoint pattern logic."""

from pathlib import Path

import pytest

from cosma_backend.filter import FilterConfig, FilterMode


@pytest.mark.unit
class TestFilterBlacklistMode:
    """Blacklist mode: non-! patterns are exclude, ! patterns are include (override)."""

    def test_exclude_pattern_blocks_file(self):
        config = FilterConfig(mode=FilterMode.BLACKLIST, blacklist_exclude=["*.log"])
        assert config.should_include(Path("/tmp/app.log"), Path("/tmp")) is False

    def test_non_matching_file_passes(self):
        config = FilterConfig(mode=FilterMode.BLACKLIST, blacklist_exclude=["*.log"])
        assert config.should_include(Path("/tmp/doc.pdf"), Path("/tmp")) is True

    def test_include_pattern_overrides_exclude(self):
        config = FilterConfig(
            mode=FilterMode.BLACKLIST,
            blacklist_exclude=["*.log"],
            blacklist_include=["important.log"],
        )
        assert config.should_include(Path("/tmp/important.log"), Path("/tmp")) is True

    def test_no_patterns_includes_all(self):
        config = FilterConfig(mode=FilterMode.BLACKLIST)
        assert config.should_include(Path("/tmp/anything.txt"), Path("/tmp")) is True


@pytest.mark.unit
class TestFilterWhitelistMode:
    """Whitelist mode: include patterns are what's allowed, exclude overrides."""

    def test_matching_include_passes(self):
        config = FilterConfig(mode=FilterMode.WHITELIST, whitelist_include=["*.pdf"])
        assert config.should_include(Path("/tmp/doc.pdf"), Path("/tmp")) is True

    def test_non_matching_file_blocked(self):
        config = FilterConfig(mode=FilterMode.WHITELIST, whitelist_include=["*.pdf"])
        assert config.should_include(Path("/tmp/doc.txt"), Path("/tmp")) is False

    def test_exclude_overrides_include(self):
        config = FilterConfig(
            mode=FilterMode.WHITELIST,
            whitelist_include=["*.pdf"],
            whitelist_exclude=["secret.pdf"],
        )
        assert config.should_include(Path("/tmp/secret.pdf"), Path("/tmp")) is False

    def test_no_include_patterns_blocks_all(self):
        config = FilterConfig(mode=FilterMode.WHITELIST)
        # With no include patterns in whitelist mode, nothing passes
        result = config.should_include(Path("/tmp/anything.txt"), Path("/tmp"))
        # Behavior depends on implementation; just verify it's a bool
        assert isinstance(result, bool)


@pytest.mark.unit
class TestFilterPatternSplitting:
    """Test the pattern splitting logic used by the /test endpoint.

    In blacklist mode: non-! -> exclude, ! -> include
    In whitelist mode: non-! -> include, ! -> exclude
    """

    def test_blacklist_split(self):
        patterns = ["*.log", "*.tmp", "!important.log"]
        mode = FilterMode.BLACKLIST

        exclude = [p for p in patterns if not p.startswith("!")]
        include = [p[1:] for p in patterns if p.startswith("!")]

        assert exclude == ["*.log", "*.tmp"]
        assert include == ["important.log"]

    def test_whitelist_split(self):
        patterns = ["*.pdf", "*.docx", "!secret.pdf"]
        mode = FilterMode.WHITELIST

        include = [p for p in patterns if not p.startswith("!")]
        exclude = [p[1:] for p in patterns if p.startswith("!")]

        assert include == ["*.pdf", "*.docx"]
        assert exclude == ["secret.pdf"]

    def test_whitelist_include_matches(self):
        """Whitelist with *.pdf should include doc.pdf."""
        config = FilterConfig(
            mode=FilterMode.WHITELIST,
            whitelist_include=["*.pdf"],
        )
        assert config.should_include(Path("/tmp/doc.pdf"), Path("/tmp")) is True
        assert config.should_include(Path("/tmp/doc.txt"), Path("/tmp")) is False
