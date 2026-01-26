"""
Filter Configuration Module

Handles loading, saving, and applying file filter patterns.
Supports gitignore-style patterns with blacklist/whitelist modes.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from platformdirs import PlatformDirs

from cosma_backend.logging import sm

logger = logging.getLogger(__name__)

# Config file names
GLOBAL_CONFIG_FILENAME = "filter.json"  # Global config in app data dir
LOCAL_CONFIG_FILENAME = ".cosmaconfig"   # Per-folder config

# App directories (same as used by app.py and tui)
APP_DIRS = PlatformDirs("cosma", ensure_exists=True)


class FilterMode(str, Enum):
    """Filter mode determines base behavior."""
    BLACKLIST = "blacklist"  # Exclude matching files (default)
    WHITELIST = "whitelist"  # Only include matching files


# Default patterns for blacklist mode
DEFAULT_EXCLUDE_PATTERNS = [
    # Hidden files and directories
    ".*",

    # Version control
    ".git/",
    ".svn/",
    ".hg/",

    # Dependencies and virtual environments
    "node_modules/",
    "vendor/",
    "venv/",
    ".venv/",
    "__pycache__/",
    "*.egg-info/",

    # Build outputs and compiled files
    "build/",
    "dist/",
    "target/",
    "*.pyc",
    "*.pyo",
    "*.class",
    "*.o",
    "*.so",
    "*.dylib",

    # IDE and editor files
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    "*~",

    # OS files
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",

    # Logs and caches
    "*.log",
    "*.cache",
    ".cache/",
    ".pytest_cache/",

    # Secrets and sensitive files
    ".env",
    "*.pem",
    "*.key",
    "*.p12",
]

DEFAULT_INCLUDE_PATTERNS = [
    # Common exceptions - files we usually want to index even if they start with .
    ".gitignore",
    ".env.example",
]


@dataclass
class FilterConfig:
    """
    Configuration for file filtering.

    Supports both global config and per-folder config.
    Per-folder configs extend (not replace) global patterns.

    NEW: Stores separate pattern lists for each mode to prevent data loss
    when switching modes. When mode changes, patterns switch to the
    appropriate list without semantic confusion.
    """
    mode: FilterMode = FilterMode.WHITELIST  # Default to whitelist for new users

    # Mode-specific pattern storage (NEW)
    blacklist_exclude: list[str] = field(default_factory=list)
    blacklist_include: list[str] = field(default_factory=list)
    whitelist_include: list[str] = field(default_factory=list)
    whitelist_exclude: list[str] = field(default_factory=list)

    # Legacy fields for backward compatibility (deprecated)
    # These are auto-populated based on current mode
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)

    version: int = 2  # Bumped to version 2 for new structure
    config_path: Optional[Path] = None

    # Cached compiled patterns for performance
    _exclude_regex: Optional[re.Pattern] = field(default=None, repr=False, compare=False)
    _include_regex: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Compile patterns after initialization."""
        # Sync legacy fields with mode-specific fields
        self._sync_legacy_fields()
        self._compile_patterns()

    def _sync_legacy_fields(self):
        """Sync legacy exclude/include fields with mode-specific fields."""
        if self.mode == FilterMode.BLACKLIST:
            self.exclude = self.blacklist_exclude
            self.include = self.blacklist_include
        else:  # WHITELIST
            self.exclude = self.whitelist_exclude
            self.include = self.whitelist_include

    def _compile_patterns(self):
        """Compile glob patterns to regex for efficient matching."""
        if self.exclude:
            self._exclude_regex = self._patterns_to_regex(self.exclude)
        else:
            self._exclude_regex = None

        if self.include:
            self._include_regex = self._patterns_to_regex(self.include)
        else:
            self._include_regex = None

    @staticmethod
    def _pattern_to_regex(pattern: str) -> str:
        """
        Convert a single glob pattern to regex.

        Supports gitignore-style patterns:
        - * matches any characters except /
        - ** matches any characters including /
        - ? matches single character
        - / at end means directory
        - / at start means from root
        - .* means files/directories starting with dot (hidden items)
        """
        original_pattern = pattern

        # Handle directory patterns (trailing /)
        is_directory = pattern.endswith("/")
        if is_directory:
            pattern = pattern[:-1]

        # Handle root-relative patterns (leading /)
        is_root_relative = pattern.startswith("/")
        if is_root_relative:
            pattern = pattern[1:]

        # Special case: ".*" pattern for hidden files/directories
        # This should match:
        # 1. Files starting with dot (e.g., .env, .gitignore)
        # 2. Directories starting with dot (e.g., .git)
        # 3. All contents inside directories starting with dot (e.g., .git/config)
        if pattern == ".*":
            # Match: hidden file OR path containing hidden directory
            return r"(?:^|.*/)\.[^/]+(?:/.*)?$"

        # Convert glob pattern to regex character by character
        regex = ""
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == "*":
                # Check for **
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    # ** matches anything including /
                    # Check for /**/ pattern
                    if i + 2 < len(pattern) and pattern[i + 2] == "/":
                        regex += "(?:.*/)?"  # Match any path segment or nothing
                        i += 3
                        continue
                    else:
                        regex += ".*"
                        i += 2
                        continue
                else:
                    regex += "[^/]*"  # * matches anything except /
            elif c == "?":
                regex += "[^/]"  # ? matches single char except /
            elif c in ".^$+{}[]|()":
                regex += "\\" + c  # Escape special chars
            else:
                regex += c
            i += 1

        # Add anchoring based on pattern type
        if is_directory:
            # Directory pattern: match the directory name anywhere in path
            # or match anything under that directory
            # e.g., "node_modules/" should match:
            # - /path/to/node_modules
            # - /path/to/node_modules/anything
            # - node_modules (at root)
            regex = f"(?:^{regex}(?:/.*)?$|.*/{regex}(?:/.*)?$)"
        else:
            # File pattern
            if "/" in pattern or is_root_relative:
                # Path pattern - must match from position
                if is_root_relative:
                    regex = "^" + regex + "$"
                else:
                    regex = "(?:^|.*/)" + regex + "$"
            else:
                # Simple filename pattern - match the filename part
                # e.g., "*.pyc" should match any .pyc file
                # e.g., ".env" should match exactly .env file
                regex = "(?:^|.*/)" + regex + "$"

        return regex

    @staticmethod
    def _patterns_to_regex(patterns: list[str]) -> re.Pattern:
        """Combine multiple patterns into a single regex."""
        if not patterns:
            return re.compile("^$")  # Match nothing

        regex_parts = [FilterConfig._pattern_to_regex(p) for p in patterns]
        combined = "|".join(f"(?:{r})" for r in regex_parts)
        return re.compile(combined, re.IGNORECASE)  # macOS is case-insensitive

    def _matches_patterns(self, relative_path: str, regex: Optional[re.Pattern]) -> bool:
        """Check if path matches compiled regex patterns."""
        if regex is None:
            return False
        return regex.search(relative_path) is not None

    def should_include(self, file_path: Path, base_path: Path) -> bool:
        """
        Determine if a file should be included based on filter config.

        Args:
            file_path: Absolute path to the file
            base_path: Base directory (watched directory root)

        Returns:
            True if file should be indexed, False if excluded
        """
        try:
            # Get path relative to base directory
            relative_path = str(file_path.relative_to(base_path))
        except ValueError:
            # File is not under base_path, use filename only
            relative_path = file_path.name

        # Also check against just the filename for simple patterns
        filename = file_path.name

        if self.mode == FilterMode.BLACKLIST:
            # Blacklist mode: Include all, then exclude matching, then include exceptions

            # Check if excluded by any exclude pattern
            is_excluded = (
                self._matches_patterns(relative_path, self._exclude_regex) or
                self._matches_patterns(filename, self._exclude_regex)
            )

            if not is_excluded:
                return True  # Not excluded, include it

            # Check if there's an include exception
            is_included = (
                self._matches_patterns(relative_path, self._include_regex) or
                self._matches_patterns(filename, self._include_regex)
            )

            return is_included  # Include only if there's an exception

        else:  # WHITELIST mode
            # Whitelist mode: Exclude all, then include matching, then exclude specific

            # Check if included by any include pattern
            is_included = (
                self._matches_patterns(relative_path, self._include_regex) or
                self._matches_patterns(filename, self._include_regex)
            )

            if not is_included:
                return False  # Not in whitelist, exclude it

            # Check if there's an exclude exception
            is_excluded = (
                self._matches_patterns(relative_path, self._exclude_regex) or
                self._matches_patterns(filename, self._exclude_regex)
            )

            return not is_excluded  # Include only if not explicitly excluded

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "mode": self.mode.value,
            # Mode-specific patterns
            "blacklist_exclude": self.blacklist_exclude,
            "blacklist_include": self.blacklist_include,
            "whitelist_include": self.whitelist_include,
            "whitelist_exclude": self.whitelist_exclude,
            # Legacy fields for backward compatibility
            "exclude": self.exclude,
            "include": self.include,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict, config_path: Optional[Path] = None) -> FilterConfig:
        """Create FilterConfig from dictionary with migration from v1 to v2."""
        mode_str = data.get("mode", "blacklist")
        try:
            mode = FilterMode(mode_str)
        except ValueError:
            logger.warning(sm("Invalid filter mode, defaulting to blacklist", mode=mode_str))
            mode = FilterMode.BLACKLIST

        version = data.get("version", 1)

        # Migration from version 1 to version 2
        if version == 1:
            logger.info(sm("Migrating filter config from v1 to v2", path=str(config_path) if config_path else "unknown"))

            # In v1, we only had exclude/include without mode-specific storage
            # Migrate based on current mode
            old_exclude = data.get("exclude", [])
            old_include = data.get("include", [])

            # Default whitelist patterns (common file types)
            default_whitelist = [
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

            if mode == FilterMode.BLACKLIST:
                # Old patterns were for blacklist mode
                blacklist_exclude = old_exclude
                blacklist_include = old_include
                # Initialize whitelist with sensible defaults
                whitelist_include = default_whitelist
                whitelist_exclude = []
            else:
                # Old patterns were for whitelist mode
                whitelist_include = old_include if old_include else default_whitelist
                whitelist_exclude = old_exclude
                # Initialize blacklist with defaults
                blacklist_exclude = DEFAULT_EXCLUDE_PATTERNS.copy()
                blacklist_include = DEFAULT_INCLUDE_PATTERNS.copy()

            return cls(
                version=2,  # Upgrade to v2
                mode=mode,
                blacklist_exclude=blacklist_exclude,
                blacklist_include=blacklist_include,
                whitelist_include=whitelist_include,
                whitelist_exclude=whitelist_exclude,
                config_path=config_path,
            )

        # Version 2+ format
        return cls(
            version=version,
            mode=mode,
            blacklist_exclude=data.get("blacklist_exclude", []),
            blacklist_include=data.get("blacklist_include", []),
            whitelist_include=data.get("whitelist_include", []),
            whitelist_exclude=data.get("whitelist_exclude", []),
            config_path=config_path,
        )

    @classmethod
    def from_json(cls, json_str: str, config_path: Optional[Path] = None) -> FilterConfig:
        """Create FilterConfig from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data, config_path)
        except json.JSONDecodeError as e:
            logger.error(sm("Failed to parse filter config JSON", error=str(e)))
            raise ValueError(f"Invalid JSON in filter config: {e}")

    @classmethod
    def load_from_file(cls, path: Path) -> Optional[FilterConfig]:
        """
        Load FilterConfig from a .cosmaconfig file.

        Args:
            path: Path to the config file

        Returns:
            FilterConfig if file exists and is valid, None otherwise
        """
        if not path.exists():
            logger.debug(sm("Config file not found", path=str(path)))
            return None

        try:
            content = path.read_text(encoding="utf-8")
            config = cls.from_json(content, config_path=path)
            logger.info(sm("Loaded filter config", path=str(path), mode=config.mode.value,
                          exclude_count=len(config.exclude), include_count=len(config.include)))
            return config
        except Exception as e:
            logger.error(sm("Failed to load filter config", path=str(path), error=str(e)))
            return None

    def save_to_file(self, path: Optional[Path] = None) -> bool:
        """
        Save FilterConfig to a .cosmaconfig file.

        Args:
            path: Path to save to (uses self.config_path if not provided)

        Returns:
            True if saved successfully, False otherwise
        """
        save_path = path or self.config_path
        if save_path is None:
            logger.error(sm("No path specified for saving filter config"))
            return False

        try:
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_path.write_text(self.to_json(), encoding="utf-8")
            self.config_path = save_path
            logger.info(sm("Saved filter config", path=str(save_path)))
            return True
        except Exception as e:
            logger.error(sm("Failed to save filter config", path=str(save_path), error=str(e)))
            return False

    @classmethod
    def get_default_global_path(cls) -> Path:
        """Get the default path for global config.

        Returns path in app data directory:
        - macOS: ~/Library/Application Support/cosma/filter.json
        - Linux: ~/.local/share/cosma/filter.json
        - Windows: C:/Users/<user>/AppData/Local/cosma/filter.json
        """
        return Path(APP_DIRS.user_data_dir) / GLOBAL_CONFIG_FILENAME

    @classmethod
    def load_global(cls) -> FilterConfig:
        """
        Load global filter config, creating default if not exists.

        Returns:
            FilterConfig instance (default if file doesn't exist)
        """
        global_path = cls.get_default_global_path()

        config = cls.load_from_file(global_path)
        if config is not None:
            return config

        # Create default config with sensible defaults for both modes
        logger.info(sm("Creating default global filter config", path=str(global_path)))

        # Default whitelist patterns (common document and media types)
        default_whitelist_include = [
            # Documents
            "*.pdf", "*.docx", "*.doc", "*.pptx", "*.ppt", "*.xlsx", "*.xls",
            "*.odt", "*.ods", "*.odp", "*.pages", "*.numbers", "*.key",
            # Images
            "*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.heic", "*.gif", "*.webp",
            "*.svg", "*.ico", "*.psd",
            # Audio
            "*.mp3", "*.wav", "*.aac", "*.m4a", "*.flac", "*.ogg",
            # Video
            "*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm",
            # Text and Web
            "*.html", "*.htm", "*.txt", "*.csv", "*.json", "*.xml", "*.md",
            "*.rtf", "*.log",
            # Config
            "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg",
            # Technical docs
            "*.tex", "*.rst", "*.adoc",
            # Archives
            "*.zip", "*.epub", "*.rar", "*.7z", "*.tar", "*.gz",
            # Code
            "*.py", "*.js", "*.java", "*.cpp", "*.c", "*.go", "*.rs", "*.sql", "*.sh",
            # Email
            "*.eml", "*.msg",
            # Design/CAD
            "*.dwg", "*.dxf", "*.skp",
        ]

        config = cls(
            version=2,
            mode=FilterMode.WHITELIST,  # Default to whitelist for new users
            blacklist_exclude=DEFAULT_EXCLUDE_PATTERNS.copy(),
            blacklist_include=DEFAULT_INCLUDE_PATTERNS.copy(),
            whitelist_include=default_whitelist_include,
            whitelist_exclude=[],
            config_path=global_path,
        )
        config.save_to_file()
        return config

    @classmethod
    def load_for_directory(cls, directory: Path, global_config: Optional[FilterConfig] = None) -> FilterConfig:
        """
        Load merged config for a specific directory.

        Merges global config with per-folder .cosmaconfig if present.
        Per-folder patterns extend global patterns.
        Per-folder mode is ignored (global mode always applies).

        Args:
            directory: Directory to load config for
            global_config: Optional pre-loaded global config

        Returns:
            Merged FilterConfig
        """
        # Load global config if not provided
        if global_config is None:
            global_config = cls.load_global()

        # Check for per-folder config (.cosmaconfig in the watched directory)
        folder_config_path = directory / LOCAL_CONFIG_FILENAME
        folder_config = cls.load_from_file(folder_config_path)

        if folder_config is None:
            # No per-folder config, use global
            logger.debug(sm("Using global config for directory", directory=str(directory)))
            return global_config

        # Merge configs: per-folder extends global
        # Mode comes from global only
        merged_exclude = list(global_config.exclude)
        merged_include = list(global_config.include)

        # Add per-folder patterns (avoid duplicates)
        for pattern in folder_config.exclude:
            if pattern not in merged_exclude:
                merged_exclude.append(pattern)

        for pattern in folder_config.include:
            if pattern not in merged_include:
                merged_include.append(pattern)

        # Create merged config with mode-specific fields set correctly
        if global_config.mode == FilterMode.BLACKLIST:
            merged = cls(
                mode=global_config.mode,
                blacklist_exclude=merged_exclude,
                blacklist_include=merged_include,
                whitelist_include=list(global_config.whitelist_include),
                whitelist_exclude=list(global_config.whitelist_exclude),
                config_path=folder_config_path,
            )
        else:  # WHITELIST
            merged = cls(
                mode=global_config.mode,
                whitelist_exclude=merged_exclude,
                whitelist_include=merged_include,
                blacklist_exclude=list(global_config.blacklist_exclude),
                blacklist_include=list(global_config.blacklist_include),
                config_path=folder_config_path,
            )

        logger.info(sm("Merged filter config for directory",
                      directory=str(directory),
                      mode=merged.mode.value,
                      exclude_count=len(merged.exclude),
                      include_count=len(merged.include)))

        return merged

    def merge_with(self, other: FilterConfig) -> FilterConfig:
        """
        Create a new config that merges this config with another.

        The other config's patterns extend this config's patterns.
        Mode is kept from this config (self).

        Args:
            other: Config to merge with

        Returns:
            New merged FilterConfig
        """
        merged_exclude = list(self.exclude)
        merged_include = list(self.include)

        for pattern in other.exclude:
            if pattern not in merged_exclude:
                merged_exclude.append(pattern)

        for pattern in other.include:
            if pattern not in merged_include:
                merged_include.append(pattern)

        return FilterConfig(
            mode=self.mode,
            exclude=merged_exclude,
            include=merged_include,
            config_path=other.config_path or self.config_path,
        )


class FilterConfigManager:
    """
    Manages filter configurations for multiple directories.

    Caches loaded configs and handles reloading on config file changes.
    """

    def __init__(self):
        self._global_config: Optional[FilterConfig] = None
        self._directory_configs: dict[Path, FilterConfig] = {}

    @property
    def global_config(self) -> FilterConfig:
        """Get the global config, loading if necessary."""
        if self._global_config is None:
            self._global_config = FilterConfig.load_global()
        return self._global_config

    def reload_global(self) -> FilterConfig:
        """Force reload of global config."""
        self._global_config = FilterConfig.load_global()
        # Invalidate all directory configs since they depend on global
        self._directory_configs.clear()
        return self._global_config

    def get_config_for_directory(self, directory: Path) -> FilterConfig:
        """
        Get the merged config for a directory.

        Uses cached config if available.
        """
        directory = directory.resolve()

        if directory not in self._directory_configs:
            self._directory_configs[directory] = FilterConfig.load_for_directory(
                directory, self.global_config
            )

        return self._directory_configs[directory]

    def reload_directory(self, directory: Path) -> FilterConfig:
        """Force reload of config for a specific directory."""
        directory = directory.resolve()
        self._directory_configs[directory] = FilterConfig.load_for_directory(
            directory, self.global_config
        )
        return self._directory_configs[directory]

    def invalidate_directory(self, directory: Path):
        """Remove cached config for a directory."""
        directory = directory.resolve()
        self._directory_configs.pop(directory, None)

    def update_global_config(
        self,
        mode: Optional[FilterMode] = None,
        exclude: Optional[list[str]] = None,
        include: Optional[list[str]] = None,
        # New mode-specific parameters
        blacklist_exclude: Optional[list[str]] = None,
        blacklist_include: Optional[list[str]] = None,
        whitelist_include: Optional[list[str]] = None,
        whitelist_exclude: Optional[list[str]] = None,
    ) -> FilterConfig:
        """
        Update the global config.

        Args:
            mode: New filter mode (or keep existing)
            exclude: Legacy exclude patterns (deprecated - use mode-specific params)
            include: Legacy include patterns (deprecated - use mode-specific params)
            blacklist_exclude: Patterns to exclude in blacklist mode
            blacklist_include: Patterns to include in blacklist mode
            whitelist_include: Patterns to include in whitelist mode
            whitelist_exclude: Patterns to exclude in whitelist mode

        Returns:
            Updated global config
        """
        config = self.global_config

        # Determine new mode
        new_mode = mode if mode is not None else config.mode

        # Handle mode-specific updates
        if blacklist_exclude is not None or blacklist_include is not None or \
           whitelist_include is not None or whitelist_exclude is not None:
            # Using new API with mode-specific patterns
            new_config = FilterConfig(
                mode=new_mode,
                blacklist_exclude=blacklist_exclude if blacklist_exclude is not None else config.blacklist_exclude,
                blacklist_include=blacklist_include if blacklist_include is not None else config.blacklist_include,
                whitelist_include=whitelist_include if whitelist_include is not None else config.whitelist_include,
                whitelist_exclude=whitelist_exclude if whitelist_exclude is not None else config.whitelist_exclude,
                config_path=config.config_path,
            )
        else:
            # Using legacy API (exclude/include) - apply to current mode's patterns
            new_blacklist_exclude = config.blacklist_exclude
            new_blacklist_include = config.blacklist_include
            new_whitelist_include = config.whitelist_include
            new_whitelist_exclude = config.whitelist_exclude

            if exclude is not None or include is not None:
                if new_mode == FilterMode.BLACKLIST:
                    new_blacklist_exclude = exclude if exclude is not None else config.blacklist_exclude
                    new_blacklist_include = include if include is not None else config.blacklist_include
                else:  # WHITELIST
                    new_whitelist_exclude = exclude if exclude is not None else config.whitelist_exclude
                    new_whitelist_include = include if include is not None else config.whitelist_include

            new_config = FilterConfig(
                mode=new_mode,
                blacklist_exclude=new_blacklist_exclude,
                blacklist_include=new_blacklist_include,
                whitelist_include=new_whitelist_include,
                whitelist_exclude=new_whitelist_exclude,
                config_path=config.config_path,
            )

        new_config.save_to_file()
        self._global_config = new_config

        # Invalidate all directory configs
        self._directory_configs.clear()

        return new_config

    def should_include_file(self, file_path: Path, base_directory: Path) -> bool:
        """
        Check if a file should be included based on filter config.

        Args:
            file_path: Path to the file
            base_directory: The watched directory root

        Returns:
            True if file should be indexed
        """
        config = self.get_config_for_directory(base_directory)
        return config.should_include(file_path, base_directory)
