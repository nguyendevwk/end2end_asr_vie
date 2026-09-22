"""Test bootstrap: stub GPU-only deps so service modules import anywhere."""

import sys
from unittest.mock import MagicMock

# Create a torch mock with a proper Tensor class so scipy's issubclass checks work
_torch_mock = MagicMock()
_torch_mock.Tensor = type("Tensor", (), {})  # Real class for issubclass()
for _name in ("torch", "qwen_asr", "qwen_tts"):
    sys.modules.setdefault(_name, _torch_mock if _name == "torch" else MagicMock())
