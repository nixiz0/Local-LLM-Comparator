# Agent Guide

Use this repo as a Docker-friendly `uv` + Streamlit app for comparing local
Ollama models. The target user is non-technical, so keep the interface,
documentation, and launch path simple, organized, and easy to run.

## Default Workflow

- Sync dependencies with `uv sync`.
- Start Ollama separately with `ollama serve` when local model calls are needed.
- Run the app with `uv run streamlit run streamlit_app.py`.
- For non-technical user runs, prefer the Docker launchers. Auto mode should use
  an isolated Ollama container stack by default, not the user's host Ollama:
  - Windows: `run.bat`
  - macOS/Linux: `./run.sh`
- Keep the Docker app image on `uv`; do not add `requirements.txt` unless the
  user explicitly asks for a pip-based image.
- Keep Docker Compose files layered:
  - `docker/compose.yaml`: base app container
  - `docker/compose.ollama.yaml`: app + Ollama CPU container
  - `docker/compose.nvidia.yaml`: NVIDIA GPU override
  - `docker/compose.amd.yaml`: AMD ROCm override for Linux
- Use Bash-style commands in docs unless the user explicitly asks for Windows
  or PowerShell examples.
- Keep `streamlit_app.py` as a small entrypoint; application logic belongs in
  `src/`.
- Prefer standard-library HTTP helpers unless a dependency is already justified.

## Project Shape

- `streamlit_app.py`: small Streamlit entrypoint
- `docker/Dockerfile`: lightweight app container image
- `docker/compose.yaml`: base Docker Compose file for app
- `docker/compose.ollama.yaml`: optional Ollama container
- `docker/compose.nvidia.yaml`: NVIDIA GPU compose override
- `docker/compose.amd.yaml`: AMD ROCm compose override
- `run.sh`: macOS/Linux Docker launcher
- `run.bat`: Windows Docker launcher
- `.streamlit/config.toml`: Streamlit runtime/theme settings
- `css/style.css`: Streamlit visual theme
- `src/config.py`: app constants
- `src/ui.py`: Streamlit views, widgets, and run orchestration
- `src/benchmarks.py`: predefined benchmark cases and prompt templates
- `src/ollama_client.py`: Ollama API calls and result schema
- `src/metrics.py`: aggregate tables and leaderboards
- `src/exports.py`: CSV, JSON, Markdown exports
- `src/i18n.py`: English/French translations

## Guard Rails

- Do not add notebook-specific tooling unless explicitly requested.
- Do not reintroduce marimo files unless the user explicitly asks for marimo.
- Keep UI wording direct and non-technical for non-developer users.
- Keep the cyan futuristic visual direction in `css/style.css`.
- Keep English as the default language and maintain French translations for
  visible UI text.
- Use Streamlit's current `width` API instead of deprecated
  `use_container_width`.
- Preserve result metadata for comparisons, including model, case, run,
  temperature, top-p, seed, thinking state, latency, token counts, and
  tokens/sec.
- Keep custom benchmark definitions serializable so reports can be exported.
- Validate with `uv run python -m py_compile streamlit_app.py src/*.py` and,
  when possible, start Streamlit locally.
