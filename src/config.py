from __future__ import annotations

import os
from pathlib import Path


APP_TITLE = "Local LLM Compare"
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_RUN_TIMEOUT_SECONDS = 600
DEFAULT_MODEL_PULL_TIMEOUT_SECONDS = 3600
DEFAULT_TEMPERATURE = 0.70
STYLE_PATH = Path("css/style.css")
