"""Tests for config loading and device resolution (src/utils/config.py).

Tests verify:
- YAML config loads correctly
- Missing config file raises FileNotFoundError
- get_device returns valid torch.device
- Auto device detection works
- Explicit device override works
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.utils.config import get_device, load_config


# ──────────────────────────────────────────────
# load_config
# ──────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        """Loads a valid YAML config into dict."""
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text("model:\n  name: test\n  dim: 256\n")
        config = load_config(str(cfg_file))
        assert config["model"]["name"] == "test"
        assert config["model"]["dim"] == 256

    def test_missing_file_raises(self):
        """Nonexistent config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_loads_real_default_config(self):
        """The actual default.yaml loads without error."""
        config = load_config("configs/default.yaml")
        assert "model" in config
        assert "training" in config
        assert "data" in config

    def test_returns_dict(self, tmp_path):
        """Returns a plain dict, not None."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("key: value\n")
        result = load_config(str(cfg_file))
        assert isinstance(result, dict)


# ──────────────────────────────────────────────
# get_device
# ──────────────────────────────────────────────

class TestGetDevice:
    def test_explicit_cpu(self):
        """Explicit 'cpu' returns cpu device."""
        config = {"hardware": {"device": "cpu"}}
        device = get_device(config)
        assert device == torch.device("cpu")

    def test_auto_returns_valid_device(self):
        """Auto detection returns a valid device type."""
        config = {"hardware": {"device": "auto"}}
        device = get_device(config)
        assert device.type in ("cpu", "cuda", "mps")

    def test_missing_hardware_section(self):
        """Missing hardware section defaults to auto → valid device."""
        config = {}
        device = get_device(config)
        assert device.type in ("cpu", "cuda", "mps")

    def test_returns_torch_device(self):
        """Return type is torch.device."""
        config = {"hardware": {"device": "cpu"}}
        device = get_device(config)
        assert isinstance(device, torch.device)
