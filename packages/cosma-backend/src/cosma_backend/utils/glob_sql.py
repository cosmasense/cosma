"""Convert simple file-path globs to SQL LIKE patterns.

Used by search to honor frontend filter tokens like `@*.pdf` without
exposing full regex (ReDoS hazard) or shelling out to fnmatch on every
candidate row. The conversion is intentionally narrow:

  *           → %
  ?           → _
  literal %   → !% (escaped, with ESCAPE '!')
  literal _   → !_ (escaped, with ESCAPE '!')
  literal !   → !! (the escape char itself must be escaped)

Everything else passes through unchanged. SQL queries that use the
returned pattern MUST add ``ESCAPE '!'`` to the LIKE clause. We chose
``!`` over backslash to side-step SQL/Python double-escaping pitfalls
— SQLite does not treat backslash specially, but several drivers and
DDL paths do, so a printable ASCII escape that nobody else cares about
keeps the surface predictable.
"""

from typing import Final

LIKE_ESCAPE_CHAR: Final[str] = "!"


def glob_to_like(pattern: str) -> str:
    """Convert a glob to a SQL LIKE pattern. See module docstring."""
    out: list[str] = []
    for ch in pattern:
        if ch == "*":
            out.append("%")
        elif ch == "?":
            out.append("_")
        elif ch in ("%", "_", LIKE_ESCAPE_CHAR):
            out.append(LIKE_ESCAPE_CHAR + ch)
        else:
            out.append(ch)
    return "".join(out)


def is_glob(pattern: str) -> bool:
    """True if pattern contains glob metacharacters."""
    return "*" in pattern or "?" in pattern
