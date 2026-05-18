"""Discoverer bypass for `.git/`-containing directories.

A user pointing cosma at ~/Documents almost certainly has a project
folder or two with `.git/` inside. Walking into those folders pulls
in source/build/lockfile/node_modules-style noise the user did not
mean to search. The default-on `FilterConfig.skip_git_repos` prunes
those subtrees during discovery.

These tests pin two contracts:
  * default-on behavior: a git working copy is dropped entirely
  * opt-out: setting `skip_git_repos=False` reverts to old behavior
"""

from pathlib import Path

import pytest

from cosma_backend.discoverer import Discoverer
from cosma_backend.filter import FilterConfig


def _make_repo(parent: Path, name: str) -> Path:
    """Create a `parent/name/` folder shaped like a git working copy:
    a `.git/` admin tree plus a couple of real files we'd otherwise
    index.
    """
    repo = parent / name
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / "README.md").write_text("# project\n")
    (repo / "src.py").write_text("print('hi')\n")
    return repo


@pytest.mark.unit
class TestGitRepoBypass:
    @pytest.mark.asyncio
    async def test_default_on_skips_git_subtree(self, tmp_path: Path):
        # Layout: tmp/
        #   notes.md           ← should be discovered
        #   myproject/         ← git repo, should be pruned entirely
        #     .git/HEAD
        #     README.md
        #     src.py
        (tmp_path / "notes.md").write_text("free-floating note\n")
        _make_repo(tmp_path, "myproject")

        cfg = FilterConfig()  # defaults: skip_git_repos=True
        assert cfg.skip_git_repos is True

        files = list(Discoverer().files_in(tmp_path, filter_config=cfg))
        names = {f.filename for f in files}

        assert "notes.md" in names
        assert "README.md" not in names, "git repo should not be descended into"
        assert "src.py" not in names
        assert "HEAD" not in names, ".git admin files must stay invisible"

    @pytest.mark.asyncio
    async def test_opt_out_indexes_everything(self, tmp_path: Path):
        # Same layout, but the user explicitly turned the bypass off.
        # They still get the default filter exclude for `.git/` patterns
        # (so HEAD won't show up — that's the existing blacklist), but
        # the repo's normal files (README.md, src.py) should be visible.
        (tmp_path / "notes.md").write_text("free-floating note\n")
        _make_repo(tmp_path, "myproject")

        cfg = FilterConfig.load_global()  # picks up the default blacklist
        cfg.skip_git_repos = False

        files = list(Discoverer().files_in(tmp_path, filter_config=cfg))
        names = {f.filename for f in files}

        assert "notes.md" in names
        assert "README.md" in names
        assert "src.py" in names
        # The default blacklist still excludes hidden files (.git/HEAD).
        assert "HEAD" not in names

    @pytest.mark.asyncio
    async def test_nested_git_repo_also_skipped(self, tmp_path: Path):
        # A repo nested inside a regular folder should still be pruned.
        outer = tmp_path / "documents"
        outer.mkdir()
        (outer / "report.md").write_text("# report\n")
        _make_repo(outer, "side-project")

        cfg = FilterConfig()
        files = list(Discoverer().files_in(tmp_path, filter_config=cfg))
        names = {f.filename for f in files}

        assert "report.md" in names
        assert "README.md" not in names

    def test_filter_config_persists_skip_git_repos(self, tmp_path: Path):
        """Round-trip the field through to_dict/from_dict so old configs
        keep the default and the field survives a save/load cycle."""
        cfg = FilterConfig()
        d = cfg.to_dict()
        assert d["skip_git_repos"] is True

        # Older configs missing the key fall back to True.
        legacy = FilterConfig.from_dict({"version": 3, "mode": "blacklist"})
        assert legacy.skip_git_repos is True

        # Explicit false survives the round-trip.
        cfg2 = FilterConfig.from_dict({"version": 4, "mode": "blacklist",
                                        "skip_git_repos": False})
        assert cfg2.skip_git_repos is False
