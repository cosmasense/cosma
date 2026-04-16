"""Unit tests for SettingsManager."""

import tempfile
from pathlib import Path

import pytest

from cosma_backend.settings import (
    SettingsManager,
    Settings,
    EmbedderConfig,
    SummarizerConfig,
    ParserConfig,
    _coerce,
    _from_dict,
    _get_by_path,
    _set_by_path,
)


@pytest.mark.unit
class TestCoerce:
    """Test type coercion helper."""

    def test_coerce_str_to_int(self):
        assert _coerce("42", int) == 42

    def test_coerce_str_to_bool_true(self):
        assert _coerce("true", bool) is True
        assert _coerce("True", bool) is True
        assert _coerce("1", bool) is True
        assert _coerce("yes", bool) is True

    def test_coerce_str_to_bool_false(self):
        assert _coerce("false", bool) is False
        assert _coerce("0", bool) is False
        assert _coerce("no", bool) is False

    def test_coerce_same_type(self):
        assert _coerce(42, int) == 42
        assert _coerce("hello", str) == "hello"
        assert _coerce(True, bool) is True

    def test_coerce_int_to_str(self):
        assert _coerce(42, str) == "42"


@pytest.mark.unit
class TestFromDict:
    """Test recursive dict -> dataclass conversion."""

    def test_partial_dict(self):
        s = _from_dict(Settings, {"embedder": {"provider": "online"}})
        assert s.embedder.provider == "online"
        # Other fields keep defaults
        assert s.embedder.model == "text-embedding-3-small"
        assert s.summarizer.provider == "llamacpp"

    def test_nested_dict(self):
        s = _from_dict(Settings, {
            "summarizer": {"ollama": {"model": "llama3", "host": "http://myhost:11434"}}
        })
        assert s.summarizer.ollama.model == "llama3"
        assert s.summarizer.ollama.host == "http://myhost:11434"

    def test_empty_dict(self):
        s = _from_dict(Settings, {})
        assert s == Settings()


@pytest.mark.unit
class TestPathHelpers:
    """Test dotted-path get/set on dataclass tree."""

    def test_get_by_path(self):
        s = Settings()
        assert _get_by_path(s, "embedder.provider") == "local"
        assert _get_by_path(s, "summarizer.ollama.model") == "qwen3-vl:2b-instruct"

    def test_get_by_path_unknown_raises(self):
        s = Settings()
        with pytest.raises(KeyError):
            _get_by_path(s, "embedder.nonexistent")

    def test_set_by_path(self):
        s = Settings()
        _set_by_path(s, "embedder.provider", "online")
        assert s.embedder.provider == "online"

    def test_set_by_path_coerces_type(self):
        s = Settings()
        _set_by_path(s, "embedder.dimensions", "1024")
        assert s.embedder.dimensions == 1024
        assert isinstance(s.embedder.dimensions, int)

    def test_set_by_path_unknown_raises(self):
        s = Settings()
        with pytest.raises(KeyError):
            _set_by_path(s, "embedder.nonexistent", "value")


@pytest.mark.unit
class TestSettingsManager:
    """Test SettingsManager load/save/get/set."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temp dir simulating user_config_dir."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return config_dir

    @pytest.fixture
    def manager(self, temp_config_dir, isolated_app_dirs):
        """Create a SettingsManager with a temp config dir.

        Uses the shared `isolated_app_dirs` shim so nothing escapes to the real
        user dirs even before `_toml_path` is overridden.
        """
        mgr = SettingsManager(isolated_app_dirs)
        # Override the TOML path to use temp dir
        mgr._toml_path = temp_config_dir / "settings.toml"
        return mgr

    def test_load_creates_toml_when_missing(self, manager):
        """Loading with no existing TOML should create one with defaults."""
        settings = manager.load()
        assert manager.toml_path.exists()
        assert settings.embedder.provider == "local"
        assert settings.summarizer.provider == "llamacpp"
        assert settings.embedder.dimensions == 512

    def test_load_returns_defaults(self, manager):
        """All fields should match defaults on fresh load."""
        settings = manager.load()
        defaults = Settings()
        assert settings.embedder.provider == defaults.embedder.provider
        assert settings.summarizer.provider == defaults.summarizer.provider
        assert settings.parser.spotlight_enabled == defaults.parser.spotlight_enabled

    def test_save_and_reload(self, manager):
        """Settings should round-trip through save/load."""
        manager.load()
        manager.set_by_path("summarizer.provider", "ollama")
        manager.set_by_path("embedder.dimensions", 1024)

        # Create a new manager pointing at the same file
        mgr2 = SettingsManager(manager._dirs)
        mgr2._toml_path = manager.toml_path
        settings = mgr2.load()

        assert settings.summarizer.provider == "ollama"
        assert settings.embedder.dimensions == 1024

    def test_get_by_path_known_key(self, manager):
        manager.load()
        assert manager.get_by_path("embedder.provider") == "local"

    def test_get_by_path_unknown_key_raises(self, manager):
        manager.load()
        with pytest.raises(KeyError):
            manager.get_by_path("embedder.nonexistent")

    def test_set_by_path_unknown_key_raises(self, manager):
        manager.load()
        with pytest.raises(KeyError):
            manager.set_by_path("nonexistent.key", "value")

    def test_set_by_path_coerces_type(self, manager):
        manager.load()
        manager.set_by_path("embedder.dimensions", "256")
        assert manager.get_by_path("embedder.dimensions") == 256
        assert isinstance(manager.get_by_path("embedder.dimensions"), int)

    def test_set_bool_coercion(self, manager):
        manager.load()
        manager.set_by_path("summarizer.llamacpp.verbose", "true")
        assert manager.get_by_path("summarizer.llamacpp.verbose") is True

    def test_update_bulk(self, manager):
        manager.load()
        updated = manager.update({
            "summarizer.provider": "ollama",
            "summarizer.ollama.model": "llama3",
            "embedder.dimensions": 256,
        })
        assert updated["summarizer"]["provider"] == "ollama"
        assert updated["summarizer"]["ollama"]["model"] == "llama3"
        assert updated["embedder"]["dimensions"] == 256

    def test_update_unknown_key_raises(self, manager):
        manager.load()
        with pytest.raises(KeyError):
            manager.update({"bad.key": "value"})

    def test_to_dict_grouped(self, manager):
        manager.load()
        grouped = manager.to_dict()
        assert "embedder" in grouped
        assert "summarizer" in grouped
        assert "parser" in grouped
        assert grouped["embedder"]["provider"] == "local"
        assert grouped["summarizer"]["provider"] == "llamacpp"

    def test_defaults_static(self):
        defaults = SettingsManager.defaults()
        assert "embedder" in defaults
        assert defaults["embedder"]["provider"] == "local"

    def test_toml_nested_structure(self, manager):
        """TOML file should have nested sections."""
        manager.load()
        manager.set_by_path("summarizer.ollama.model", "custom-model")

        import tomllib
        with open(manager.toml_path, "rb") as f:
            data = tomllib.load(f)

        assert data["summarizer"]["ollama"]["model"] == "custom-model"
        assert "embedder" in data
        assert "parser" in data


@pytest.mark.unit
class TestSettingsValidation:
    """Test value validation for settings with range constraints."""

    def test_cooldown_seconds_zero_raises(self):
        s = Settings()
        with pytest.raises(ValueError, match="cooldown_seconds"):
            _set_by_path(s, "queue.cooldown_seconds", 0)

    def test_cooldown_seconds_negative_raises(self):
        s = Settings()
        with pytest.raises(ValueError):
            _set_by_path(s, "queue.cooldown_seconds", -5)

    def test_cooldown_seconds_valid(self):
        s = Settings()
        _set_by_path(s, "queue.cooldown_seconds", 1)
        assert s.queue.cooldown_seconds == 1

    def test_max_concurrency_zero_raises(self):
        s = Settings()
        with pytest.raises(ValueError, match="max_concurrency"):
            _set_by_path(s, "queue.max_concurrency", 0)

    def test_max_concurrency_valid(self):
        s = Settings()
        _set_by_path(s, "queue.max_concurrency", 4)
        assert s.queue.max_concurrency == 4

    def test_max_retries_negative_raises(self):
        s = Settings()
        with pytest.raises(ValueError, match="max_retries"):
            _set_by_path(s, "queue.max_retries", -1)

    def test_max_retries_zero_valid(self):
        s = Settings()
        _set_by_path(s, "queue.max_retries", 0)
        assert s.queue.max_retries == 0

    def test_check_interval_seconds_too_low_raises(self):
        s = Settings()
        with pytest.raises(ValueError, match="check_interval_seconds"):
            _set_by_path(s, "scheduler.check_interval_seconds", 3)

    def test_check_interval_seconds_valid(self):
        s = Settings()
        _set_by_path(s, "scheduler.check_interval_seconds", 10)
        assert s.scheduler.check_interval_seconds == 10

    def test_unvalidated_field_accepts_any_value(self):
        """Fields without validation rules should accept any value."""
        s = Settings()
        _set_by_path(s, "embedder.dimensions", 1)
        assert s.embedder.dimensions == 1
