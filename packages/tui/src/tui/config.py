"""
Configuration management for EZF using platformdirs
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from platformdirs import user_config_dir


class Config:
    """Configuration manager for EZF"""
    
    def __init__(self):
        self.config_dir = Path(user_config_dir("ezf", ensure_exists=True))
        self.config_file = self.config_dir / "config.json"
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}")
                self._config = {}
        else:
            self._config = {}
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Error saving config: {e}")
    
    def is_first_run(self) -> bool:
        """Check if this is the first run (no config file exists)"""
        return not self.config_file.exists()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self._config[key] = value
        self._save_config()
    
    def get_theme(self) -> str:
        """Get the configured theme"""
        return self.get('theme', 'textual-dark')
    
    def set_theme(self, theme: str) -> None:
        """Set the theme"""
        self.set('theme', theme)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the full configuration as a dictionary"""
        return self._config.copy()


# Global config instance
_config = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config