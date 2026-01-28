"""
Filters API Blueprint

Handles endpoints related to file filtering configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from quart import Blueprint, current_app
from quart_schema import validate_request, validate_response

from cosma_backend.filter import FilterConfig, FilterMode

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

filters_bp = Blueprint('filters', __name__)


# ============================================================================
# Request/Response Models
# ============================================================================

@dataclass
class FilterConfigResponse:
    """Response containing filter configuration."""
    version: int
    mode: str  # "blacklist" or "whitelist"
    # Legacy fields (deprecated but kept for compatibility)
    exclude: list[str]
    include: list[str]
    # Mode-specific pattern storage (NEW)
    blacklist_exclude: list[str]
    blacklist_include: list[str]
    whitelist_include: list[str]
    whitelist_exclude: list[str]
    config_path: str


@dataclass
class UpdateFilterConfigRequest:
    """Request to update filter configuration."""
    mode: Optional[str] = None  # "blacklist" or "whitelist"
    # Legacy fields (deprecated but kept for compatibility)
    exclude: Optional[list[str]] = None
    include: Optional[list[str]] = None
    # Mode-specific pattern storage (NEW)
    blacklist_exclude: Optional[list[str]] = None
    blacklist_include: Optional[list[str]] = None
    whitelist_include: Optional[list[str]] = None
    whitelist_exclude: Optional[list[str]] = None
    # Control whether to apply changes immediately
    apply_immediately: bool = True  # If False, just update config without cleaning DB


@dataclass
class UpdateFilterConfigResponse:
    """Response after updating filter configuration."""
    success: bool
    message: str
    config: FilterConfigResponse
    removed_count: int  # Number of files removed from index


@dataclass
class TestPatternRequest:
    """Request to test pattern matching."""
    patterns: list[str]
    file_paths: list[str]
    mode: str = "blacklist"  # "blacklist" or "whitelist"


@dataclass
class PatternTestResult:
    """Result of testing a single file path."""
    file_path: str
    included: bool
    matched_pattern: Optional[str]


@dataclass
class TestPatternResponse:
    """Response for pattern testing."""
    results: list[PatternTestResult]


@dataclass
class AddPatternRequest:
    """Request to add a pattern."""
    pattern: str
    pattern_type: str = "exclude"  # "exclude" or "include"


@dataclass
class AddPatternResponse:
    """Response after adding a pattern."""
    success: bool
    message: str
    removed_count: int  # Files removed if exclude pattern added


@dataclass
class RemovePatternRequest:
    """Request to remove a pattern."""
    pattern: str
    pattern_type: str = "exclude"  # "exclude" or "include"


@dataclass
class RemovePatternResponse:
    """Response after removing a pattern."""
    success: bool
    message: str


# ============================================================================
# Endpoints
# ============================================================================

@filters_bp.get("/config")
@validate_response(FilterConfigResponse, 200)
async def get_filter_config() -> tuple[FilterConfigResponse, int]:
    """
    Get the current global filter configuration.

    GET /api/filters/config

    Returns:
        200: Current filter configuration with mode-specific patterns
    """
    config = current_app.filter_manager.global_config

    return FilterConfigResponse(
        version=config.version,
        mode=config.mode.value,
        # Legacy fields for backward compatibility
        exclude=config.exclude,
        include=config.include,
        # Mode-specific patterns
        blacklist_exclude=config.blacklist_exclude,
        blacklist_include=config.blacklist_include,
        whitelist_include=config.whitelist_include,
        whitelist_exclude=config.whitelist_exclude,
        config_path=str(config.config_path) if config.config_path else "",
    ), 200


@filters_bp.put("/config")
@validate_request(UpdateFilterConfigRequest)
@validate_response(UpdateFilterConfigResponse, 200)
async def update_filter_config(data: UpdateFilterConfigRequest) -> tuple[UpdateFilterConfigResponse, int]:
    """
    Update the global filter configuration.

    PUT /api/filters/config

    This will:
    1. Update the configuration
    2. Save to the config file
    3. Optionally remove files from index (if apply_immediately=True)

    NEW: Supports mode-specific pattern storage to prevent data loss
    when switching modes. Set apply_immediately=False to update config
    without triggering database cleanup (useful for frontend local editing).

    Returns:
        200: Configuration updated successfully
    """
    # Parse mode if provided
    mode = None
    if data.mode is not None:
        try:
            mode = FilterMode(data.mode)
        except ValueError:
            return UpdateFilterConfigResponse(
                success=False,
                message=f"Invalid mode: {data.mode}. Must be 'blacklist' or 'whitelist'",
                config=FilterConfigResponse(
                    version=2,
                    mode="blacklist",
                    exclude=[],
                    include=[],
                    blacklist_exclude=[],
                    blacklist_include=[],
                    whitelist_include=[],
                    whitelist_exclude=[],
                    config_path="",
                ),
                removed_count=0,
            ), 400

    # Update configuration
    new_config = current_app.filter_manager.update_global_config(
        mode=mode,
        exclude=data.exclude,
        include=data.include,
        blacklist_exclude=data.blacklist_exclude,
        blacklist_include=data.blacklist_include,
        whitelist_include=data.whitelist_include,
        whitelist_exclude=data.whitelist_exclude,
    )

    # Clean up excluded files from database only if apply_immediately=True
    removed_count = 0
    if data.apply_immediately:
        removed_count = await cleanup_excluded_files()

    return UpdateFilterConfigResponse(
        success=True,
        message="Filter configuration updated successfully",
        config=FilterConfigResponse(
            version=new_config.version,
            mode=new_config.mode.value,
            exclude=new_config.exclude,
            include=new_config.include,
            blacklist_exclude=new_config.blacklist_exclude,
            blacklist_include=new_config.blacklist_include,
            whitelist_include=new_config.whitelist_include,
            whitelist_exclude=new_config.whitelist_exclude,
            config_path=str(new_config.config_path) if new_config.config_path else "",
        ),
        removed_count=removed_count,
    ), 200


@filters_bp.post("/pattern")
@validate_request(AddPatternRequest)
@validate_response(AddPatternResponse, 200)
async def add_pattern(data: AddPatternRequest) -> tuple[AddPatternResponse, int]:
    """
    Add a single pattern to the filter configuration.

    POST /api/filters/pattern

    Returns:
        200: Pattern added successfully
    """
    config = current_app.filter_manager.global_config

    if data.pattern_type == "exclude":
        if data.pattern in config.exclude:
            return AddPatternResponse(
                success=False,
                message=f"Pattern '{data.pattern}' already exists in exclude list",
                removed_count=0,
            ), 400

        new_exclude = config.exclude + [data.pattern]
        current_app.filter_manager.update_global_config(exclude=new_exclude)

        # Clean up files matching new exclude pattern
        removed_count = await cleanup_excluded_files()

        return AddPatternResponse(
            success=True,
            message=f"Added exclude pattern: {data.pattern}",
            removed_count=removed_count,
        ), 200

    elif data.pattern_type == "include":
        if data.pattern in config.include:
            return AddPatternResponse(
                success=False,
                message=f"Pattern '{data.pattern}' already exists in include list",
                removed_count=0,
            ), 400

        new_include = config.include + [data.pattern]
        current_app.filter_manager.update_global_config(include=new_include)

        return AddPatternResponse(
            success=True,
            message=f"Added include pattern: {data.pattern}",
            removed_count=0,
        ), 200

    else:
        return AddPatternResponse(
            success=False,
            message=f"Invalid pattern_type: {data.pattern_type}. Must be 'exclude' or 'include'",
            removed_count=0,
        ), 400


@filters_bp.delete("/pattern")
@validate_request(RemovePatternRequest)
@validate_response(RemovePatternResponse, 200)
async def remove_pattern(data: RemovePatternRequest) -> tuple[RemovePatternResponse, int]:
    """
    Remove a pattern from the filter configuration.

    DELETE /api/filters/pattern

    Returns:
        200: Pattern removed successfully
    """
    config = current_app.filter_manager.global_config

    if data.pattern_type == "exclude":
        if data.pattern not in config.exclude:
            return RemovePatternResponse(
                success=False,
                message=f"Pattern '{data.pattern}' not found in exclude list",
            ), 404

        new_exclude = [p for p in config.exclude if p != data.pattern]
        current_app.filter_manager.update_global_config(exclude=new_exclude)

        return RemovePatternResponse(
            success=True,
            message=f"Removed exclude pattern: {data.pattern}",
        ), 200

    elif data.pattern_type == "include":
        if data.pattern not in config.include:
            return RemovePatternResponse(
                success=False,
                message=f"Pattern '{data.pattern}' not found in include list",
            ), 404

        new_include = [p for p in config.include if p != data.pattern]
        current_app.filter_manager.update_global_config(include=new_include)

        return RemovePatternResponse(
            success=True,
            message=f"Removed include pattern: {data.pattern}",
        ), 200

    else:
        return RemovePatternResponse(
            success=False,
            message=f"Invalid pattern_type: {data.pattern_type}. Must be 'exclude' or 'include'",
        ), 400


@filters_bp.post("/test")
@validate_request(TestPatternRequest)
@validate_response(TestPatternResponse, 200)
async def test_patterns(data: TestPatternRequest) -> tuple[TestPatternResponse, int]:
    """
    Test pattern matching against a list of file paths.

    POST /api/filters/test

    This is useful for previewing what files would be filtered
    before applying the configuration.

    Returns:
        200: Test results
    """
    try:
        mode = FilterMode(data.mode)
    except ValueError:
        mode = FilterMode.BLACKLIST

    # Create temporary config for testing
    # Split patterns into exclude and include based on ! prefix
    exclude_patterns = [p for p in data.patterns if not p.startswith("!")]
    include_patterns = [p[1:] for p in data.patterns if p.startswith("!")]

    test_config = FilterConfig(
        mode=mode,
        exclude=exclude_patterns,
        include=include_patterns,
    )

    results = []
    for file_path_str in data.file_paths:
        file_path = Path(file_path_str)
        # Use parent as base path for testing
        base_path = file_path.parent

        included = test_config.should_include(file_path, base_path)

        # Find which pattern matched (for debugging)
        matched_pattern = None
        filename = file_path.name

        if not included:
            # Check exclude patterns
            for pattern in exclude_patterns:
                temp_config = FilterConfig(mode=mode, exclude=[pattern], include=[])
                if not temp_config.should_include(file_path, base_path):
                    matched_pattern = pattern
                    break

        results.append(PatternTestResult(
            file_path=file_path_str,
            included=included,
            matched_pattern=matched_pattern,
        ))

    return TestPatternResponse(results=results), 200


@filters_bp.get("/defaults")
@validate_response(FilterConfigResponse, 200)
async def get_default_config() -> tuple[FilterConfigResponse, int]:
    """
    Get the default filter configuration.

    GET /api/filters/defaults

    Returns the default patterns without loading from file.

    Returns:
        200: Default filter configuration
    """
    from cosma_backend.filter.filter_config import (
        DEFAULT_EXCLUDE_PATTERNS,
        DEFAULT_INCLUDE_PATTERNS,
    )

    # Get default whitelist patterns from load_global logic
    default_whitelist_include = [
        "*.pdf", "*.docx", "*.doc", "*.pptx", "*.ppt", "*.xlsx", "*.xls",
        "*.odt", "*.ods", "*.odp", "*.pages", "*.numbers", "*.key",
        "*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.heic", "*.gif", "*.webp",
        "*.svg", "*.ico", "*.psd",
        "*.mp3", "*.wav", "*.aac", "*.m4a", "*.flac", "*.ogg",
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm",
        "*.html", "*.htm", "*.txt", "*.csv", "*.json", "*.xml", "*.md",
        "*.rtf", "*.log",
        "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg",
        "*.tex", "*.rst", "*.adoc",
        "*.zip", "*.epub", "*.rar", "*.7z", "*.tar", "*.gz",
        "*.py", "*.js", "*.java", "*.cpp", "*.c", "*.go", "*.rs", "*.sql", "*.sh",
        "*.eml", "*.msg",
        "*.dwg", "*.dxf", "*.skp",
    ]

    return FilterConfigResponse(
        version=2,
        mode="blacklist",
        exclude=DEFAULT_EXCLUDE_PATTERNS,
        include=DEFAULT_INCLUDE_PATTERNS,
        blacklist_exclude=DEFAULT_EXCLUDE_PATTERNS,
        blacklist_include=DEFAULT_INCLUDE_PATTERNS,
        whitelist_include=default_whitelist_include,
        whitelist_exclude=[],
        config_path="",
    ), 200


@filters_bp.post("/apply")
@validate_response(UpdateFilterConfigResponse, 200)
async def apply_filter_changes() -> tuple[UpdateFilterConfigResponse, int]:
    """
    Apply current filter configuration to database.

    POST /api/filters/apply

    This triggers cleanup of files that should now be excluded based on
    the current filter configuration. Use this after updating config with
    apply_immediately=False.

    Returns:
        200: Filter changes applied successfully
    """
    config = current_app.filter_manager.global_config

    # Clean up excluded files from database
    removed_count = await cleanup_excluded_files()

    return UpdateFilterConfigResponse(
        success=True,
        message=f"Filter changes applied: {removed_count} files removed from index",
        config=FilterConfigResponse(
            version=config.version,
            mode=config.mode.value,
            exclude=config.exclude,
            include=config.include,
            blacklist_exclude=config.blacklist_exclude,
            blacklist_include=config.blacklist_include,
            whitelist_include=config.whitelist_include,
            whitelist_exclude=config.whitelist_exclude,
            config_path=str(config.config_path) if config.config_path else "",
        ),
        removed_count=removed_count,
    ), 200


@filters_bp.post("/reset")
@validate_response(UpdateFilterConfigResponse, 200)
async def reset_to_defaults() -> tuple[UpdateFilterConfigResponse, int]:
    """
    Reset filter configuration to defaults.

    POST /api/filters/reset

    Returns:
        200: Configuration reset successfully
    """
    from cosma_backend.filter.filter_config import (
        DEFAULT_EXCLUDE_PATTERNS,
        DEFAULT_INCLUDE_PATTERNS,
    )

    new_config = current_app.filter_manager.update_global_config(
        mode=FilterMode.BLACKLIST,
        exclude=DEFAULT_EXCLUDE_PATTERNS.copy(),
        include=DEFAULT_INCLUDE_PATTERNS.copy(),
    )

    # Clean up excluded files
    removed_count = await cleanup_excluded_files()

    return UpdateFilterConfigResponse(
        success=True,
        message="Filter configuration reset to defaults",
        config=FilterConfigResponse(
            version=new_config.version,
            mode=new_config.mode.value,
            exclude=new_config.exclude,
            include=new_config.include,
            blacklist_exclude=new_config.blacklist_exclude,
            blacklist_include=new_config.blacklist_include,
            whitelist_include=new_config.whitelist_include,
            whitelist_exclude=new_config.whitelist_exclude,
            config_path=str(new_config.config_path) if new_config.config_path else "",
        ),
        removed_count=removed_count,
    ), 200


# ============================================================================
# Helper Functions
# ============================================================================

async def cleanup_excluded_files() -> int:
    """
    Remove files from the database that now match exclusion patterns.

    Returns:
        Number of files removed
    """
    from cosma_backend.logging import get_logger
    import logging

    logger = get_logger(__name__)

    db = current_app.db
    filter_manager = current_app.filter_manager

    # Get all watched directories
    watched_dirs = await db.get_watched_directories(active_only=True)

    if not watched_dirs:
        return 0

    removed_count = 0

    # For each watched directory, check files against filter
    for watched_dir in watched_dirs:
        base_path = watched_dir.path
        config = filter_manager.get_config_for_directory(base_path)

        # Get all files under this directory from database
        # We need to add a method to get files by directory prefix
        async with db.acquire() as conn:
            rows = await conn.fetchall(
                "SELECT id, file_path FROM files WHERE file_path LIKE ? || '/%' OR file_path = ?",
                (str(base_path), str(base_path))
            )

            for row in rows:
                file_path = Path(row["file_path"])

                if not config.should_include(file_path, base_path):
                    # File should be excluded, delete it
                    await conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
                    removed_count += 1
                    logger.info("Removed excluded file from index",
                                  file_path=str(file_path))

    logger.info("Cleanup completed", removed_count=removed_count)
    return removed_count
