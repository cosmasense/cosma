"""Walk `/Applications` and extract Info.plist metadata for each .app.

The result is a list of `Application` dataclasses ready to be UPSERTed
into the `applications` table. We deliberately stop at Info.plist —
the keys it carries (CFBundleDisplayName, CFBundleIdentifier,
LSApplicationCategoryType, CFBundleShortVersionString,
CFBundleGetInfoString) are enough to power the "I forgot what it's
called but it's the one that does X" query without dragging the
LLM into discovery. A later enrichment pass can fill `use_cases`.

We use stdlib `plistlib` for both XML and binary plists so this
module has zero ObjC dependency — the parser already pulls in pyobjc
for Vision OCR, but this code stays portable in case anyone wants to
run the backend headless.
"""

from __future__ import annotations

import plistlib
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from cosma_backend.logging import get_logger
from cosma_backend.models import Application

logger = get_logger(__name__)

# Standard macOS application install roots. /Applications covers
# globally-installed bundles; ~/Applications covers per-user installs
# (some apps default there, plus Setapp, Homebrew Cask --appdir, etc.).
DEFAULT_SEARCH_ROOTS: tuple[str, ...] = (
    "/Applications",
    "~/Applications",
)


class ApplicationsDiscoverer:
    """Walks application install roots and yields `Application` records.

    Designed to be cheap to call: a full scan of /Applications on a
    typical Mac (~100 bundles) finishes in tens of ms because we
    open each Info.plist once and never recurse into the bundle's
    Resources tree. There's intentionally no caching layer here —
    the repository UPSERT is idempotent, so a periodic re-scan
    every 6 h is a no-op for unchanged bundles.
    """

    def __init__(self, search_roots: Optional[tuple[str, ...]] = None) -> None:
        self._search_roots = search_roots or DEFAULT_SEARCH_ROOTS

    def discover(self) -> list[Application]:
        """Eager scan. Returns the full list. Use ``iter_discover``
        if you want to stream into a long-running upsert."""
        return list(self.iter_discover())

    def iter_discover(self) -> Iterator[Application]:
        """Generator variant — yields each bundle as we read its plist.

        Bundles whose Info.plist is missing, corrupted, or doesn't
        carry a usable display name are skipped with a debug log
        (not warn — broken bundles in /Applications are common,
        e.g. partial installer turds, and we don't want to spam
        logs every 6 hours).
        """
        seen_paths: set[str] = set()
        for root in self._search_roots:
            base = Path(root).expanduser()
            if not base.is_dir():
                continue
            # We iterate the top level only. macOS apps live at
            # /Applications/<Name>.app; nested .apps (helpers,
            # auto-updaters) are bundle internals the user doesn't
            # launch directly. Indexing them would dilute search
            # results with "Safari Web Content" etc.
            try:
                entries = sorted(base.iterdir())
            except OSError as e:
                logger.debug("Cannot list app root", root=str(base), error=str(e))
                continue
            for entry in entries:
                if not entry.name.endswith(".app"):
                    continue
                resolved = entry.resolve()
                key = str(resolved)
                if key in seen_paths:
                    # Same .app symlinked into multiple roots — only
                    # surface it once. Honor the first root order.
                    continue
                seen_paths.add(key)
                app = self._read_bundle(resolved)
                if app is not None:
                    yield app

    def _read_bundle(self, bundle_path: Path) -> Optional[Application]:
        info_plist = bundle_path / "Contents" / "Info.plist"
        if not info_plist.is_file():
            logger.debug("Skipping bundle without Info.plist",
                         bundle=str(bundle_path))
            return None
        try:
            with info_plist.open("rb") as f:
                plist = plistlib.load(f)
        except Exception as e:
            logger.debug("Skipping bundle with unreadable Info.plist",
                         bundle=str(bundle_path), error=str(e))
            return None

        # Display name resolution per Apple's documented precedence:
        # CFBundleDisplayName > CFBundleName > bundle directory name.
        # The directory name is the only field guaranteed to exist
        # because bundles by definition live at SomeName.app.
        display_name = (
            (plist.get("CFBundleDisplayName") or "").strip()
            or (plist.get("CFBundleName") or "").strip()
            or bundle_path.stem
        )
        if not display_name:
            return None

        return Application(
            app_path=str(bundle_path),
            bundle_id=(plist.get("CFBundleIdentifier") or None),
            display_name=display_name,
            short_version=(plist.get("CFBundleShortVersionString") or None),
            category=(plist.get("LSApplicationCategoryType") or None),
            # Many apps populate CFBundleGetInfoString with copyright
            # boilerplate ("Copyright 2024 Apple Inc."). Filter the
            # obvious-noise variants so they don't poison the FTS
            # `description` column.
            description=_clean_description(
                plist.get("CFBundleGetInfoString")
                or plist.get("NSHumanReadableCopyright"),
            ),
            icon_path=_resolve_icon_path(bundle_path, plist),
        )


def _clean_description(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Reject pure copyright notices — they look meaningful in the
    # plist but don't help "what does this app do" search at all.
    low = s.lower()
    if low.startswith("copyright") or "©" in s or "all rights reserved" in low:
        return None
    return s


def _resolve_icon_path(bundle_path: Path, plist: dict) -> Optional[str]:
    """Resolve the .icns path so the future UI can lazy-load it
    without a second plist read. Returns absolute path or None."""
    icon_file = plist.get("CFBundleIconFile")
    if not icon_file:
        return None
    icon_file = str(icon_file)
    resources = bundle_path / "Contents" / "Resources"
    candidates = [resources / icon_file]
    # Apps may store the icon as either "AppIcon" or "AppIcon.icns";
    # the .icns extension is implicit in CFBundleIconFile per Apple
    # docs but commonly written either way.
    if not icon_file.endswith(".icns"):
        candidates.append(resources / f"{icon_file}.icns")
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None
