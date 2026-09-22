"""Test bootstrap: stub GPU-only deps so service modules import anywhere."""

import sys
from unittest.mock import MagicMock

for _name in ("torch", "qwen_asr", "qwen_tts"):
    sys.modules.setdefault(_name, MagicMock())
