"""Application metadata model.

Mirrors the `applications` table from schema.sql. Apps live in a
separate row-set from files because the schemas don't overlap (apps
have a bundle_id, category, use_cases; files have content_hash, parse
status, embeddings). Keeping the dataclass lean — we only store what
comes off Info.plist plus a few FTS-relevant fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Self


@dataclass
class Application:
    """A macOS application as discovered from `/Applications`."""
    id: Optional[int] = None
    app_path: str = ""
    bundle_id: Optional[str] = None
    display_name: str = ""
    short_version: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    # Optional, LLM-filled later: "what is this app used for, in one
    # sentence." Populated by the enrichment pass in Step 4. Until
    # then, this stays None and the user still finds apps by name +
    # description + category.
    use_cases: Optional[str] = None
    icon_path: Optional[str] = None
    # SHA-256 of the text body used to produce the current embedding.
    # The indexer compares it before re-embedding so a routine re-scan
    # of an unchanged app costs zero embedder calls.
    embedding_text_hash: Optional[str] = None
    indexed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> Self:
        """Build from a sqlite3.Row (or asqlite Row). Tolerates missing
        columns so a SELECT * against an older schema doesn't crash —
        unknown columns just default."""
        def _at(key, default=None):
            try:
                return row[key]
            except (IndexError, KeyError):
                return default

        def _ts(key) -> Optional[datetime]:
            v = _at(key)
            return datetime.fromtimestamp(v) if v else None

        return cls(
            id=_at("id"),
            app_path=_at("app_path") or "",
            bundle_id=_at("bundle_id"),
            display_name=_at("display_name") or "",
            short_version=_at("short_version"),
            category=_at("category"),
            description=_at("description"),
            use_cases=_at("use_cases"),
            icon_path=_at("icon_path"),
            embedding_text_hash=_at("embedding_text_hash"),
            indexed_at=_ts("indexed_at"),
            updated_at=_ts("updated_at"),
        )
