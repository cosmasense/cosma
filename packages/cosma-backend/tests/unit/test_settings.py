"""Unit tests for SettingsManager."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from platformdirs import PlatformDirs

from cosma_backend.settings import SettingsManager, SETTINGS_SCHEMA, _coerce


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
class TestSettingsManager:
    """Test SettingsManager load/save/get/set."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temp dir simulating user_config_dir."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return config_dir

    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create a SettingsManager with a temp config dir."""
        dirs = PlatformDirs("cosma-test", ensure_exists=True)
        mgr = SettingsManager(dirs)
        # Override the TOML path to use temp dir
        mgr._toml_path = temp_config_dir / "settings.toml"
        return mgr

    def test_load_creates_toml_when_missing(self, manager):
        """Loading with no existing TOML should create one with defaults."""
        settings = manager.load()
        assert manager.toml_path.exists()
        # Check some defaults
        assert settings["EMBEDDING_PROVIDER"] == "local"
        assert settings["AI_PROVIDER"] == "auto"
        assert settings["EMBEDDING_DIMENSIONS"] == 512

    def test_load_returns_defaults(self, manager):
        """All schema keys should be present with their defaults."""
        settings = manager.load()
        for key, schema in SETTINGS_SCHEMA.items():
            assert key in settings, f"Missing key: {key}"
            assert settings[key] == schema["default"], f"Wrong default for {key}"

    def test_save_and_reload(self, manager):
        """Settings should round-trip through save/load."""
        manager.load()
        manager.set("AI_PROVIDER", "ollama")
        manager.set("EMBEDDING_DIMENSIONS", 1024)

        # Create a new manager pointing at the same file
        mgr2 = SettingsManager(manager._dirs)
        mgr2._toml_path = manager.toml_path
        settings = mgr2.load()

        assert settings["AI_PROVIDER"] == "ollama"
        assert settings["EMBEDDING_DIMENSIONS"] == 1024

    def test_get_known_key(self, manager):
        manager.load()
        assert manager.get("EMBEDDING_PROVIDER") == "local"

    def test_get_unknown_key_raises(self, manager):
        manager.load()
        with pytest.raises(KeyError):
            manager.get("NONEXISTENT_KEY")

    def test_set_unknown_key_raises(self, manager):
        manager.load()
        with pytest.raises(KeyError):
            manager.set("NONEXISTENT_KEY", "value")

    def test_set_coerces_type(self, manager):
        manager.load()
        manager.set("EMBEDDING_DIMENSIONS", "256")
        assert manager.get("EMBEDDING_DIMENSIONS") == 256
        assert isinstance(manager.get("EMBEDDING_DIMENSIONS"), int)

    def test_set_bool_coercion(self, manager):
        manager.load()
        manager.set("LLAMACPP_VERBOSE", "true")
        assert manager.get("LLAMACPP_VERBOSE") is True

    def test_update_bulk(self, manager):
        manager.load()
        updated = manager.update({
            "AI_PROVIDER": "ollama",
            "OLLAMA_MODEL": "llama3",
            "EMBEDDING_DIMENSIONS": 256,
        })
        assert updated["AI_PROVIDER"] == "ollama"
        assert updated["OLLAMA_MODEL"] == "llama3"
        assert updated["EMBEDDING_DIMENSIONS"] == 256

    def test_update_unknown_key_raises(self, manager):
        manager.load()
        with pytest.raises(KeyError):
            manager.update({"BAD_KEY": "value"})

    def test_to_dict_grouped(self, manager):
        manager.load()
        grouped = manager.to_dict()
        assert "embedder" in grouped
        assert "summarizer" in grouped
        assert "parser" in grouped
        assert grouped["embedder"]["provider"] == "local"
        assert grouped["summarizer"]["provider"] == "auto"

    def test_to_flat_dict(self, manager):
        manager.load()
        flat = manager.to_flat_dict()
        for key in SETTINGS_SCHEMA:
            assert key in flat

    def test_defaults_static(self):
        defaults = SettingsManager.defaults()
        assert "embedder" in defaults
        assert defaults["embedder"]["provider"] == "local"

    def test_flat_defaults_static(self):
        flat = SettingsManager.flat_defaults()
        assert flat["EMBEDDING_PROVIDER"] == "local"
        assert flat["AI_PROVIDER"] == "auto"

    def test_env_var_overrides_toml(self, manager, monkeypatch):
        """Env vars with COSMA_ prefix should override TOML values."""
        # First load with defaults (creates TOML)
        manager.load()

        # Set env var and reload
        monkeypatch.setenv("COSMA_AI_PROVIDER", "ollama")
        settings = manager.load()
        assert settings["AI_PROVIDER"] == "ollama"

    def test_env_var_int_coercion(self, manager, monkeypatch):
        """Env var strings should be coerced to the correct type."""
        monkeypatch.setenv("COSMA_EMBEDDING_DIMENSIONS", "1024")
        settings = manager.load()
        assert settings["EMBEDDING_DIMENSIONS"] == 1024
        assert isinstance(settings["EMBEDDING_DIMENSIONS"], int)

    def test_env_var_bool_coercion(self, manager, monkeypatch):
        """Env var boolean strings should be coerced correctly."""
        monkeypatch.setenv("COSMA_LLAMACPP_VERBOSE", "true")
        settings = manager.load()
        assert settings["LLAMACPP_VERBOSE"] is True

    def test_toml_nested_structure(self, manager):
        """TOML file should have nested sections."""
        manager.load()
        manager.set("OLLAMA_MODEL", "custom-model")

        import tomllib
        with open(manager.toml_path, "rb") as f:
            data = tomllib.load(f)

        assert data["summarizer"]["ollama"]["model"] == "custom-model"
        assert "embedder" in data
        assert "parser" in data
