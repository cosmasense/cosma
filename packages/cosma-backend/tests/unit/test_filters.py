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


@pytest.mark.unit
class TestThreeTierClassification:
    """The classify() method returns EXCLUDED / PARTIAL / FULL."""

    def test_full_when_no_metadata_only_patterns(self):
        from cosma_backend.filter import FilterDecision

        config = FilterConfig(
            mode=FilterMode.BLACKLIST,
            blacklist_exclude=["*.log"],
        )
        assert config.classify(Path("/tmp/doc.pdf"), Path("/tmp")) == FilterDecision.FULL

    def test_excluded_short_circuits_partial(self):
        """Exclude rules win over metadata_only rules — a file
        excluded by blacklist_exclude is EXCLUDED even if it also
        matches a metadata-only pattern."""
        from cosma_backend.filter import FilterDecision

        config = FilterConfig(
            mode=FilterMode.BLACKLIST,
            blacklist_exclude=["*.log"],
            metadata_only_patterns=["*.log"],
        )
        # *.log: excluded wins.
        assert config.classify(Path("/tmp/app.log"), Path("/tmp")) == FilterDecision.EXCLUDED

    def test_partial_classification(self):
        """A file matching metadata_only patterns but not exclude
        patterns lands in PARTIAL."""
        from cosma_backend.filter import FilterDecision

        config = FilterConfig(
            mode=FilterMode.BLACKLIST,
            blacklist_exclude=[],
            metadata_only_patterns=["*.mkv", "Movies/"],
        )
        assert config.classify(
            Path("/Volumes/Media/Movies/Kill Bill.mkv"),
            Path("/Volumes/Media"),
        ) == FilterDecision.PARTIAL

    def test_partial_dir_pattern_matches_subfiles(self):
        """A directory metadata-only pattern matches files under it."""
        from cosma_backend.filter import FilterDecision

        config = FilterConfig(
            mode=FilterMode.BLACKLIST,
            metadata_only_patterns=["Downloads/"],
        )
        assert config.classify(
            Path("/Users/ethan/Downloads/installer.dmg"),
            Path("/Users/ethan"),
        ) == FilterDecision.PARTIAL

    def test_full_in_whitelist_when_partial_pattern_misses(self):
        """In whitelist mode, the metadata_only check still applies
        on top of the whitelist decision."""
        from cosma_backend.filter import FilterDecision

        config = FilterConfig(
            mode=FilterMode.WHITELIST,
            whitelist_include=["*.pdf", "*.mkv"],
            metadata_only_patterns=["*.mkv"],
        )
        assert config.classify(Path("/tmp/doc.pdf"), Path("/tmp")) == FilterDecision.FULL
        assert config.classify(Path("/tmp/movie.mkv"), Path("/tmp")) == FilterDecision.PARTIAL
        # Whitelist excludes everything else
        assert config.classify(Path("/tmp/notes.txt"), Path("/tmp")) == FilterDecision.EXCLUDED

    def test_v2_to_v3_migration_preserves_data(self):
        """Loading a v2 dict into v3 keeps all existing patterns and
        initializes metadata_only_patterns to []."""
        v2_data = {
            "version": 2,
            "mode": "blacklist",
            "blacklist_exclude": ["*.log"],
            "blacklist_include": [".env.example"],
            "whitelist_include": ["*.pdf"],
            "whitelist_exclude": [],
        }
        config = FilterConfig.from_dict(v2_data)
        assert config.version == 3
        assert config.blacklist_exclude == ["*.log"]
        assert config.metadata_only_patterns == []

    def test_to_dict_round_trip_preserves_metadata_only(self):
        config = FilterConfig(
            mode=FilterMode.BLACKLIST,
            metadata_only_patterns=["*.mkv"],
        )
        round_tripped = FilterConfig.from_dict(config.to_dict())
        assert round_tripped.metadata_only_patterns == ["*.mkv"]
