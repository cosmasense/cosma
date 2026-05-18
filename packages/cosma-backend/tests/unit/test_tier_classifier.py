"""Unit tests for the declarative tier classifier.

The contract under test:

  * Defaults route common extensions to the expected tier (code → C,
    office docs → A, archives → B, unknown → C floor).
  * Size downgrade routes a FULL-by-extension file above the
    `large_file_downgrade_mb` cap to SEMANTIC_NAME instead.
  * Empty user `tier_rules` falls back to the curated defaults
    (so first-launch behavior matches the design).
  * `metadata_only_patterns` from a pre-v5 config migrates into
    SEMANTIC_NAME tier rules without losing user intent.
"""

from pathlib import Path

import pytest

from cosma_backend.filter import (
    DEFAULT_LARGE_FILE_DOWNGRADE_MB,
    FilterConfig,
    Tier,
    TierRule,
    default_tier_rules,
)


def _classify(cfg: FilterConfig, name: str, base: Path, *, size_bytes: int | None = None) -> Tier:
    path = base / name
    return cfg.classify_tier(path, base, file_size_bytes=size_bytes)


@pytest.mark.unit
class TestDefaultTierAssignments:
    """The defaults are user-facing — if these change, the user-visible
    behavior changes. Pin the most consequential ones so a future edit
    to the default list can't silently demote PDFs (or promote .so
    files) without somebody noticing.
    """

    def setup_method(self):
        self.cfg = FilterConfig()  # empty tier_rules → uses defaults
        self.base = Path("/Users/ethanpan/Documents")

    @pytest.mark.parametrize("name", [
        "report.pdf", "notes.md", "draft.docx", "presentation.pptx",
        "data.xlsx", "image.png", "scan.jpeg", "phone.heic",
        "interview.mp3", "clip.mp4", "memo.txt",
        "Project.ipynb", "email.eml", "outlook_save.msg",
    ])
    def test_full_tier_for_office_text_media(self, name: str):
        assert _classify(self.cfg, name, self.base) is Tier.FULL

    @pytest.mark.parametrize("name", [
        "backup.dmg", "ubuntu.iso", "installer.pkg",
        "archive.rar", "old_photos.7z", "logs.tar.gz",  # .gz matches
    ])
    def test_semantic_name_tier_for_archives_and_disk_images(self, name: str):
        # .tar.gz matches "*.gz" (the last suffix wins because the
        # rule list iterates and `*.gz` is in SEMANTIC_NAME).
        assert _classify(self.cfg, name, self.base) is Tier.SEMANTIC_NAME

    @pytest.mark.parametrize("name", [
        "main.c", "header.h", "module.py", "ui.swift", "service.go",
        "lib.rs", "App.java", "bundle.js", "types.ts", "build.sh",
        "compiled.class", "lib.dylib", "module.so",
    ])
    def test_literal_name_tier_for_code_and_artifacts(self, name: str):
        assert _classify(self.cfg, name, self.base) is Tier.LITERAL_NAME

    def test_unknown_extension_falls_through_to_literal_name(self):
        # The trailing `*` rule is the floor. A made-up extension
        # ("foo.qwerty") must still get a tier so the search baseline
        # invariant holds.
        assert _classify(self.cfg, "weird.qwerty", self.base) is Tier.LITERAL_NAME


@pytest.mark.unit
class TestSizeDowngrade:
    """A 3-hour podcast.mp3 shouldn't get whisper-transcribed just
    because the extension rule says FULL."""

    def setup_method(self):
        self.cfg = FilterConfig()
        self.base = Path("/Users/ethanpan/Documents")

    def test_small_file_keeps_full_tier(self):
        # 50 MB — well under the default 200 MB cap.
        size = 50 * 1024 * 1024
        assert _classify(self.cfg, "lecture.mp4", self.base, size_bytes=size) is Tier.FULL

    def test_oversize_full_downgrades_to_semantic_name(self):
        # 500 MB — 2.5× the default cap, definitely a long-form video.
        size = 500 * 1024 * 1024
        assert _classify(self.cfg, "lecture.mp4", self.base, size_bytes=size) is Tier.SEMANTIC_NAME

    def test_literal_name_is_not_promoted_by_size(self):
        # A huge .c file (somehow) is still LITERAL_NAME — the
        # downgrade only applies *from FULL*, not the other direction.
        big = 500 * 1024 * 1024
        assert _classify(self.cfg, "monster.c", self.base, size_bytes=big) is Tier.LITERAL_NAME

    def test_no_size_skips_downgrade(self):
        # Callers without stat info (test paths, in-memory files) skip
        # the downgrade entirely instead of guessing.
        assert _classify(self.cfg, "lecture.mp4", self.base) is Tier.FULL


@pytest.mark.unit
class TestCustomTierRules:
    """User-edited rules override the defaults — but an empty list
    still uses the defaults (so 'reset' is just 'save with []')."""

    def test_empty_user_list_uses_defaults(self):
        cfg = FilterConfig(tier_rules=[])
        assert cfg.tier_rules == []
        # Defaults are visible through effective_tier_rules.
        assert len(cfg.effective_tier_rules()) == len(default_tier_rules())
        assert _classify(cfg, "doc.pdf", Path("/x")) is Tier.FULL

    def test_user_can_promote_code_to_full(self):
        # A developer who wants .py files summarized like docs adds
        # one rule. First-match-wins means it wins.
        cfg = FilterConfig(tier_rules=[
            TierRule("*.py", Tier.FULL),
            # No catch-all → unmatched still defaults to LITERAL_NAME
            # via the floor in classify_tier.
        ])
        assert _classify(cfg, "script.py", Path("/x")) is Tier.FULL
        # Other extensions hit the defensive floor.
        assert _classify(cfg, "image.png", Path("/x")) is Tier.LITERAL_NAME

    def test_first_match_wins(self):
        cfg = FilterConfig(tier_rules=[
            TierRule("*.md", Tier.LITERAL_NAME),     # demote
            TierRule("*.md", Tier.FULL),             # would re-promote
        ])
        assert _classify(cfg, "README.md", Path("/x")) is Tier.LITERAL_NAME


@pytest.mark.unit
class TestMetadataOnlyMigration:
    """A v3/v4 config with the legacy `metadata_only_patterns` list
    upgrades to v5 by folding those patterns into the SEMANTIC_NAME
    tier — the user's intent ("just index by filename") is preserved
    exactly because Tier B IS filename-only embedding."""

    def test_v4_metadata_only_migrates_to_semantic_name_rules(self):
        legacy = {
            "version": 4,
            "mode": "blacklist",
            "metadata_only_patterns": ["~/Movies/**", "*.iso"],
        }
        cfg = FilterConfig.from_dict(legacy)

        assert cfg.version == 5
        # Two new rules in the tier table, both SEMANTIC_NAME.
        assert TierRule("~/Movies/**", Tier.SEMANTIC_NAME) in cfg.tier_rules
        assert TierRule("*.iso", Tier.SEMANTIC_NAME) in cfg.tier_rules
        # Legacy field retained on the way out (downgrade safety).
        assert cfg.metadata_only_patterns == ["~/Movies/**", "*.iso"]

    def test_explicit_v5_tier_rules_not_overwritten_by_migration(self):
        # If a config arrives with both legacy patterns AND an
        # explicit tier_rules list, the explicit list wins. (No double
        # migration — wouldn't matter today since SEMANTIC_NAME×2 is
        # idempotent, but the rule could be different in future.)
        cfg = FilterConfig.from_dict({
            "version": 5,
            "mode": "blacklist",
            "metadata_only_patterns": ["*.iso"],
            "tier_rules": [{"pattern": "*.iso", "tier": "literal_name"}],
        })
        assert cfg.tier_rules == [TierRule("*.iso", Tier.LITERAL_NAME)]


@pytest.mark.unit
class TestTierConfigRoundTrip:
    """Serialization round-trips so persisting and reloading doesn't
    silently strip the new fields."""

    def test_to_dict_includes_tier_rules_and_downgrade_cap(self):
        cfg = FilterConfig(
            tier_rules=[TierRule("*.foo", Tier.FULL)],
            large_file_downgrade_mb=42,
        )
        d = cfg.to_dict()
        assert d["tier_rules"] == [{"pattern": "*.foo", "tier": "full"}]
        assert d["large_file_downgrade_mb"] == 42

    def test_round_trip_preserves_tier_rules(self):
        cfg = FilterConfig(tier_rules=[
            TierRule("*.foo", Tier.FULL),
            TierRule("*.bar", Tier.SEMANTIC_NAME),
        ])
        rt = FilterConfig.from_dict(cfg.to_dict())
        assert rt.tier_rules == cfg.tier_rules
        assert rt.large_file_downgrade_mb == DEFAULT_LARGE_FILE_DOWNGRADE_MB

    def test_malformed_tier_rule_entry_dropped_not_fatal(self):
        # An on-disk corrupted entry shouldn't crash discovery —
        # warn and skip the bad row.
        cfg = FilterConfig.from_dict({
            "version": 5,
            "mode": "blacklist",
            "tier_rules": [
                {"pattern": "*.ok", "tier": "full"},
                {"pattern": "missing-tier"},               # malformed
                {"pattern": "*.bad", "tier": "nonsense"},  # malformed
            ],
        })
        assert cfg.tier_rules == [TierRule("*.ok", Tier.FULL)]
