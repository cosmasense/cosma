"""Tiered indexing rules.

Three tiers govern how much work we spend per file. The shape exists
so the user can index *everything* (so a literal-filename search
never returns zero hits) while still spending LLM time only on files
where summarization actually pays off.

  * FULL           — parse → summarize → embed → FTS. The standard
                     pipeline. Status ends up COMPLETE.
  * SEMANTIC_NAME  — filename + filesystem metadata gets a semantic
                     embedding plus an FTS entry. No parse, no LLM.
                     Used for files we can't extract content from
                     (disk images, archives) and for long media where
                     transcribing a 3-hour video isn't worth it.
                     Status ends up INDEXED_PARTIAL.
  * LITERAL_NAME   — filename goes only into the FTS5 index. No
                     semantic vector at all. Cheap, comprehensive
                     baseline that guarantees coverage on code,
                     compiled artifacts, and miscellaneous binaries.
                     Status ends up INDEXED_NAME_ONLY.
  * EXCLUDED       — drop entirely; never reaches the DB. The existing
                     filter blacklist (`.git/`, `node_modules/`, etc.)
                     produces this before the tier classifier ever
                     runs, so EXCLUDED is not in the default tier table.

Rules are *declarative*: a flat list of (pattern, tier) entries
matched in order. The first match wins. A trailing `*` catch-all in
the defaults guarantees every file gets *some* tier; users can drop
or reorder rules in the Advanced Settings UI without ever risking
"this extension wasn't in any rule, what happens?"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    """Indexing tier — declarative classification of how much pipeline
    work a file should receive. Stored on disk as the lowercase value
    string so existing config files round-trip cleanly."""
    FULL = "full"
    SEMANTIC_NAME = "semantic_name"
    LITERAL_NAME = "literal_name"


@dataclass
class TierRule:
    """One row in the tier table.

    A rule matches if the file's relative path or filename matches
    `pattern` (same gitignore-style syntax used by exclude/include).
    """
    pattern: str
    tier: Tier


# Size-based downgrade: a file matched to FULL whose size exceeds
# `large_file_downgrade_mb` is automatically routed to SEMANTIC_NAME
# instead — same code path as if the user had put "*.mp4" in the
# SEMANTIC_NAME table. Keeps the rule table simple (one entry per
# extension, not per size bucket) while still encoding the "short
# video = transcribe, long video = name-only" intuition.
DEFAULT_LARGE_FILE_DOWNGRADE_MB: int = 200


# Default tier table. Matched in order; first match wins. The trailing
# "*" rule is the floor — anything not otherwise classified ends up in
# LITERAL_NAME so it's still searchable by filename.
#
# Grouping below mirrors how a user would reason about it:
#   - FULL    = files whose *content* is worth feeding to the LLM.
#   - SEMANTIC_NAME = files we genuinely can't read but whose names
#                     carry meaning ("Backup_Q3_2024.dmg",
#                     "vacation_iceland_raw.zip").
#   - LITERAL_NAME  = code, build outputs, dotfiles by extension —
#                     a filename-only literal search is the only
#                     thing a human would ever do with these.
DEFAULT_TIER_RULES: list[TierRule] = [
    # ---- SEMANTIC_NAME: file we can't open, name still meaningful ----
    TierRule("*.dmg", Tier.SEMANTIC_NAME),
    TierRule("*.iso", Tier.SEMANTIC_NAME),
    TierRule("*.pkg", Tier.SEMANTIC_NAME),
    TierRule("*.rar", Tier.SEMANTIC_NAME),
    TierRule("*.7z",  Tier.SEMANTIC_NAME),
    TierRule("*.tar", Tier.SEMANTIC_NAME),
    TierRule("*.gz",  Tier.SEMANTIC_NAME),
    TierRule("*.bz2", Tier.SEMANTIC_NAME),
    TierRule("*.xz",  Tier.SEMANTIC_NAME),

    # ---- LITERAL_NAME: code (user can browse with Finder/grep) ----
    # Source. Indexing each .c file's content into the LLM is wasted
    # compute on a typical Documents/code tree.
    TierRule("*.c",    Tier.LITERAL_NAME),
    TierRule("*.h",    Tier.LITERAL_NAME),
    TierRule("*.cpp",  Tier.LITERAL_NAME),
    TierRule("*.hpp",  Tier.LITERAL_NAME),
    TierRule("*.cc",   Tier.LITERAL_NAME),
    TierRule("*.cs",   Tier.LITERAL_NAME),
    TierRule("*.java", Tier.LITERAL_NAME),
    TierRule("*.kt",   Tier.LITERAL_NAME),
    TierRule("*.swift", Tier.LITERAL_NAME),
    TierRule("*.py",   Tier.LITERAL_NAME),
    TierRule("*.js",   Tier.LITERAL_NAME),
    TierRule("*.ts",   Tier.LITERAL_NAME),
    TierRule("*.tsx",  Tier.LITERAL_NAME),
    TierRule("*.jsx",  Tier.LITERAL_NAME),
    TierRule("*.mjs",  Tier.LITERAL_NAME),
    TierRule("*.mts",  Tier.LITERAL_NAME),
    TierRule("*.go",   Tier.LITERAL_NAME),
    TierRule("*.rs",   Tier.LITERAL_NAME),
    TierRule("*.rb",   Tier.LITERAL_NAME),
    TierRule("*.php",  Tier.LITERAL_NAME),
    TierRule("*.scala", Tier.LITERAL_NAME),
    TierRule("*.lua",  Tier.LITERAL_NAME),
    TierRule("*.sh",   Tier.LITERAL_NAME),
    TierRule("*.bash", Tier.LITERAL_NAME),
    TierRule("*.zsh",  Tier.LITERAL_NAME),
    TierRule("*.fish", Tier.LITERAL_NAME),
    TierRule("*.ps1",  Tier.LITERAL_NAME),
    TierRule("*.bat",  Tier.LITERAL_NAME),
    TierRule("*.r",    Tier.LITERAL_NAME),
    TierRule("*.dart", Tier.LITERAL_NAME),
    TierRule("*.m",    Tier.LITERAL_NAME),
    TierRule("*.mm",   Tier.LITERAL_NAME),
    TierRule("*.s",    Tier.LITERAL_NAME),
    TierRule("*.sql",  Tier.LITERAL_NAME),
    # Compiled / bundled artifacts. Names are still useful ("did I
    # build this thing?") but content is binary noise.
    TierRule("*.class", Tier.LITERAL_NAME),
    TierRule("*.o",    Tier.LITERAL_NAME),
    TierRule("*.so",   Tier.LITERAL_NAME),
    TierRule("*.dylib", Tier.LITERAL_NAME),
    TierRule("*.a",    Tier.LITERAL_NAME),
    TierRule("*.pyc",  Tier.LITERAL_NAME),
    TierRule("*.pyo",  Tier.LITERAL_NAME),
    TierRule("*.map",  Tier.LITERAL_NAME),
    TierRule("*.abilist", Tier.LITERAL_NAME),
    TierRule("*.lock", Tier.LITERAL_NAME),

    # ---- FULL: everything else worth summarizing ----
    # Office docs.
    TierRule("*.pdf",  Tier.FULL),
    TierRule("*.docx", Tier.FULL),
    TierRule("*.doc",  Tier.FULL),
    TierRule("*.pptx", Tier.FULL),
    TierRule("*.ppt",  Tier.FULL),
    TierRule("*.xlsx", Tier.FULL),
    TierRule("*.xls",  Tier.FULL),
    TierRule("*.rtf",  Tier.FULL),
    TierRule("*.epub", Tier.FULL),
    # Email + notebooks.
    TierRule("*.msg",  Tier.FULL),
    TierRule("*.eml",  Tier.FULL),
    TierRule("*.ipynb", Tier.FULL),
    # Text / markup.
    TierRule("*.md",   Tier.FULL),
    TierRule("*.txt",  Tier.FULL),
    TierRule("*.rst",  Tier.FULL),
    TierRule("*.tex",  Tier.FULL),
    TierRule("*.adoc", Tier.FULL),
    TierRule("*.html", Tier.FULL),
    TierRule("*.htm",  Tier.FULL),
    TierRule("*.xml",  Tier.FULL),
    TierRule("*.json", Tier.FULL),
    TierRule("*.csv",  Tier.FULL),
    TierRule("*.yaml", Tier.FULL),
    TierRule("*.yml",  Tier.FULL),
    TierRule("*.toml", Tier.FULL),
    TierRule("*.ini",  Tier.FULL),
    TierRule("*.cfg",  Tier.FULL),
    TierRule("*.log",  Tier.FULL),
    TierRule("*.svg",  Tier.FULL),
    # Images.
    TierRule("*.png",  Tier.FULL),
    TierRule("*.jpg",  Tier.FULL),
    TierRule("*.jpeg", Tier.FULL),
    TierRule("*.heic", Tier.FULL),
    TierRule("*.gif",  Tier.FULL),
    TierRule("*.webp", Tier.FULL),
    TierRule("*.tiff", Tier.FULL),
    TierRule("*.bmp",  Tier.FULL),
    # Audio. Size downgrade automatically routes long files (podcasts,
    # raw multi-hour recordings) to SEMANTIC_NAME so we don't spend
    # whisper time on them.
    TierRule("*.mp3",  Tier.FULL),
    TierRule("*.wav",  Tier.FULL),
    TierRule("*.aac",  Tier.FULL),
    TierRule("*.m4a",  Tier.FULL),
    TierRule("*.m4b",  Tier.FULL),
    TierRule("*.flac", Tier.FULL),
    TierRule("*.ogg",  Tier.FULL),
    TierRule("*.opus", Tier.FULL),
    TierRule("*.wma",  Tier.FULL),
    TierRule("*.aiff", Tier.FULL),
    TierRule("*.aif",  Tier.FULL),
    TierRule("*.alac", Tier.FULL),
    TierRule("*.amr",  Tier.FULL),
    TierRule("*.ac3",  Tier.FULL),
    # Video. Same size-downgrade logic as audio — short clips get
    # transcribed, full-length movies become SEMANTIC_NAME.
    TierRule("*.mp4",  Tier.FULL),
    TierRule("*.mov",  Tier.FULL),
    TierRule("*.mkv",  Tier.FULL),
    TierRule("*.avi",  Tier.FULL),
    TierRule("*.webm", Tier.FULL),
    TierRule("*.wmv",  Tier.FULL),
    TierRule("*.flv",  Tier.FULL),
    # Plain archives we *can* extract (markitdown handles .zip).
    TierRule("*.zip",  Tier.FULL),

    # ---- catch-all floor ----
    # Anything not matched above gets a literal filename entry. This
    # guarantees the "search returns SOMETHING for everything in your
    # Documents folder" invariant.
    TierRule("*", Tier.LITERAL_NAME),
]


def default_tier_rules() -> list[TierRule]:
    """Fresh copy of the default rules — callers mutate the returned
    list (e.g. when persisting to a user config), so don't hand out
    the module-level constant."""
    return [TierRule(r.pattern, r.tier) for r in DEFAULT_TIER_RULES]
