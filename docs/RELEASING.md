# Releasing to PyPI

`.github/workflows/release.yml` builds and publishes every package in the
monorepo to PyPI on any `v*` tag push. The Swift frontend runs
`uv tool upgrade cosma` on every launch (see
`CosmaManager.upgradeCosmaIfNeeded`), so once a tag lands on PyPI every
user picks up the new backend automatically on their next app start.

## Steps

1. Bump `version` in every `pyproject.toml` that changed (the root
   orchestrator plus any package under `packages/*`). Keep versions in
   lockstep — `cosma` root depends on the others and mixed versions will
   confuse `uv tool`.
2. Commit the bump: `git commit -m "v0.8.0"`.
3. Tag the commit: `git tag v0.8.0`.
4. Push both: `git push && git push --tags`.
5. Watch the `release` workflow on GitHub Actions. It builds wheels,
   publishes to PyPI via trusted publishing, and drafts a release on the
   matching tag.

## Notes

- PyPI trusted publishing requires the `pypi` environment (already
  configured). No API tokens needed.
- `skip-existing: true` is set on the publish step, so re-running the
  workflow after a partial failure is safe.
- Never push a tag without bumping the version first — PyPI rejects
  duplicate versions, and the upload step will skip silently.
- Frontend clients auto-upgrade on next launch, so there's no separate
  client-side rollout.
