"""End-to-end-ish tests for tier wiring through the discoverer + pipeline.

We don't spin up the full Pipeline (slow, needs embedder + summarizer
mocks). Instead these tests pin:

  * The discoverer translates the tier classifier output into the
    pipeline's existing routing flags (`metadata_only`, `partial_kind`)
    correctly per tier.
  * The pipeline source declares the new short-circuit branch for
    `partial_kind == "filename_only"` and lists INDEXED_NAME_ONLY in
    the "fully processed" set so re-runs skip Tier C rows.
"""

import inspect
from pathlib import Path

import pytest

from cosma_backend.discoverer import Discoverer
from cosma_backend.filter import (
    DEFAULT_LARGE_FILE_DOWNGRADE_MB,
    FilterConfig,
    Tier,
    TierRule,
)


def _make_files(root: Path, names: list[str]) -> None:
    for n in names:
        (root / n).write_text("placeholder\n")


@pytest.mark.unit
class TestDiscovererTierStamping:
    """The discoverer's job: turn the classify_tier() result into the
    File flags the pipeline already understands. Branching logic stays
    in the pipeline — the discoverer is just a label."""

    @pytest.mark.asyncio
    async def test_tier_full_leaves_file_unmarked(self, tmp_path: Path):
        _make_files(tmp_path, ["report.pdf"])

        files = list(Discoverer().files_in(tmp_path, filter_config=FilterConfig()))
        assert len(files) == 1
        f = files[0]
        assert f.metadata_only is False
        assert f.partial_kind is None

    @pytest.mark.asyncio
    async def test_tier_literal_name_stamps_filename_only(self, tmp_path: Path):
        # Code file → Tier C in the defaults.
        _make_files(tmp_path, ["script.py"])

        files = list(Discoverer().files_in(tmp_path, filter_config=FilterConfig()))
        assert len(files) == 1
        f = files[0]
        assert f.metadata_only is True
        assert f.partial_kind == "filename_only"
        assert "Tier C" in (f.metadata_only_reason or "")

    @pytest.mark.asyncio
    async def test_tier_semantic_name_stamps_user_elected(self, tmp_path: Path):
        # Archive → Tier B in the defaults.
        _make_files(tmp_path, ["backup.dmg"])

        files = list(Discoverer().files_in(tmp_path, filter_config=FilterConfig()))
        assert len(files) == 1
        f = files[0]
        assert f.metadata_only is True
        assert f.partial_kind == "user_elected"
        # New reason text identifies the tier so log readers can tell
        # tier-B-by-rule apart from the legacy user_elected path.
        assert "Tier B" in (f.metadata_only_reason or "")

    @pytest.mark.asyncio
    async def test_size_downgrade_routes_huge_video_to_tier_b(self, tmp_path: Path):
        # The discoverer reads st_size for the cap check. Write a real
        # file but lie about the size before classification by patching
        # the on-disk file's content to exceed the cap. Easier: use a
        # custom FilterConfig with a tiny downgrade cap.
        _make_files(tmp_path, ["clip.mp4"])
        cfg = FilterConfig(
            tier_rules=[],
            large_file_downgrade_mb=0,  # everything FULL gets downgraded
        )

        files = list(Discoverer().files_in(tmp_path, filter_config=cfg))
        f = next(f for f in files if f.filename == "clip.mp4")
        # FULL → SEMANTIC_NAME via size downgrade → user_elected flag.
        assert f.metadata_only is True
        assert f.partial_kind == "user_elected"

    @pytest.mark.asyncio
    async def test_legacy_metadata_only_pattern_still_wins(self, tmp_path: Path):
        # An EXISTING metadata_only_patterns user (pre-v5) lands on
        # SEMANTIC_NAME via the legacy classify() path. The new
        # classify_tier() must NOT overwrite that (would re-route the
        # file to LITERAL_NAME for, say, "*.pdf" in the user's
        # metadata-only list).
        _make_files(tmp_path, ["sensitive.pdf"])
        # Set up a v4-style config: metadata_only_patterns catches the
        # PDF, classify() returns PARTIAL.
        cfg = FilterConfig(metadata_only_patterns=["sensitive.pdf"])

        files = list(Discoverer().files_in(tmp_path, filter_config=cfg))
        f = files[0]
        assert f.metadata_only is True
        # Stays user_elected from the legacy stamp; classify_tier is
        # skipped because metadata_only is already true.
        assert f.partial_kind == "user_elected"


@pytest.mark.unit
class TestPipelineFilenameOnlyRouting:
    """The pipeline turns `partial_kind` into the final status. We
    pin the new branch by inline-source assertion — same pattern as
    test_partial_kind_routing.py — to keep the test fast and free
    of Pipeline init.
    """

    def test_routing_table_handles_filename_only(self):
        from cosma_backend.pipeline import pipeline as pipeline_module
        src = inspect.getsource(pipeline_module)
        # New short-circuit must exist.
        assert 'file.partial_kind == "filename_only"' in src, (
            "Pipeline must short-circuit Tier C (filename_only) to "
            "the cheap _run_filename_only path. If you renamed the "
            "value or the helper, update this test and the discoverer's "
            "_stamp_tier in lockstep."
        )
        assert "_run_filename_only" in src
        # Existing Tier B branch must still be present.
        assert 'file.partial_kind == "user_elected"' in src

    def test_terminal_statuses_include_indexed_name_only(self):
        from cosma_backend.pipeline import pipeline as pipeline_module
        src = inspect.getsource(pipeline_module)
        # Without this, the skip-already-processed gate would keep
        # re-running Tier C files on every restart.
        assert "ProcessingStatus.INDEXED_NAME_ONLY" in src

    def test_indexed_name_only_status_exists(self):
        from cosma_backend.models.status import ProcessingStatus
        assert ProcessingStatus.INDEXED_NAME_ONLY is not None
        assert ProcessingStatus.INDEXED_NAME_ONLY.name == "INDEXED_NAME_ONLY"


@pytest.mark.unit
class TestDefaultLargeFileDowngradeUnchanged:
    """Sanity guard: the default cap shouldn't drift unless we mean
    it. Frontend Step 5 will eventually surface this as a knob."""

    def test_default_downgrade_cap_is_200_mb(self):
        assert DEFAULT_LARGE_FILE_DOWNGRADE_MB == 200
        assert FilterConfig().large_file_downgrade_mb == 200

    def test_filter_config_compiles_default_rules_on_init(self):
        """Empty user list must trigger the defaults — otherwise
        classify_tier() has nothing to match and silently floors
        every file to LITERAL_NAME."""
        cfg = FilterConfig()
        # 70+ rules from the defaults; the count is informative,
        # the exact number is allowed to drift as the default list
        # is curated.
        assert len(cfg._compiled_tier_rules) > 50
        # And it works end to end.
        assert cfg.classify_tier(
            Path("/x/foo.pdf"), Path("/x"),
        ) is Tier.FULL
