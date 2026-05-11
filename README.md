# 🧪 Local LLM Comparator

Compare local Ollama models in a simple Streamlit app.

Local LLM Compare helps you run the same prompts against multiple local models,
compare their answers, understand speed and reliability metrics, and export the
results. It is designed to be useful for non-technical users as well as people
testing models more deeply.

## ✨ What It Does

- Compare several Ollama models side by side
- Use bilingual benchmark prompts in English and French
- Test custom prompts, tool calling, and vision models
- Read plain-language result analysis for non-specialists
- Track latency, generation speed, token counts, errors, repeats, and settings
- See the original system instruction and user prompt beside each output
- Export results as CSV, JSON, or Markdown
- Run with Docker by default, using an isolated Ollama container

## 🚀 Quick Start

Install Docker first:

- Windows/macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Linux: Docker Engine with the Compose plugin

Then run the launcher for your system.

Windows:

```powershell
run.bat
```

macOS/Linux:

```bash
sh run.sh
```

Open:

```text
http://localhost:8501
```

The default launch starts both the app and a separate Ollama server for this
project. Downloaded models stay in the Docker volume
`local-llm-compare_ollama`, separate from your normal Ollama install.

## 📦 First Model

After the app opens, use the sidebar to download a model by name.

Good small first choices:

```text
qwen3:4b
gemma3:4b
gemma4:e2b
```

Then choose models, pick predefined tests or write your own prompt, and run the
comparison.

## 🧭 Benchmark Coverage

Predefined tests cover:

- summarization
- structured extraction
- classification
- reasoning
- data analysis
- instruction following
- translation
- code generation
- tool calling
- vision review

Each predefined benchmark has English and French versions, plus metadata such
as language, difficulty, comparison focus, and expected output.

## 📊 Understanding Results

The Results page includes two levels:

- **Simple view**: quick winners, plain-language comparison, and metric guide
- **Advanced view**: detailed tables, charts, per-run metrics, and raw outputs

Important: speed metrics do not prove answer quality. Always read the answers
and prompts when choosing a model.

## ⚙️ Run Modes

Auto mode is the default and is best for most users.

| Mode | Command | Use when |
| --- | --- | --- |
| Auto | `run.bat` or `sh run.sh` | You want the easiest isolated setup |
| CPU | `run.bat cpu` or `sh run.sh cpu` | You want the most compatible Docker setup |
| NVIDIA | `run.bat nvidia` or `sh run.sh nvidia` | You have NVIDIA GPU Docker support |
| AMD Linux | `sh run.sh amd` | You have Linux AMD ROCm support |
| Native | `run.bat native` or `sh run.sh native` | You want to use Ollama already installed on your machine |

Native mode requires Ollama from <https://ollama.com/download>.

## 🐳 Advanced: Manual Docker Commands

The launchers are recommended, but you can run Docker Compose manually.

App + isolated Ollama CPU container:

```bash
docker compose -f docker/compose.yaml -f docker/compose.ollama.yaml up --build
```

App + isolated Ollama with NVIDIA override:

```bash
docker compose -f docker/compose.yaml -f docker/compose.ollama.yaml -f docker/compose.nvidia.yaml up --build
```

App + isolated Ollama with AMD ROCm override on Linux:

```bash
docker compose -f docker/compose.yaml -f docker/compose.ollama.yaml -f docker/compose.amd.yaml up --build
```

App only, using an Ollama server you manage yourself:

```bash
docker compose -f docker/compose.yaml up --build
```

Stop the containers:

```bash
docker compose -f docker/compose.yaml -f docker/compose.ollama.yaml down --remove-orphans
```

Pull a model into the isolated Ollama Docker volume:

```bash
docker compose -f docker/compose.yaml -f docker/compose.ollama.yaml exec ollama ollama pull qwen3:4b
```

## 🛠️ Advanced: Local Development

Requirements:

- Python `3.13`
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/download/)

Run locally:

```bash
uv sync
ollama serve
uv run streamlit run streamlit_app.py
```

Validate changes:

```bash
uv run python -m py_compile streamlit_app.py src/*.py
```

## 🗂️ Project Layout

```text
streamlit_app.py        App entrypoint
css/style.css           Streamlit visual theme
src/ui.py               Views, widgets, and run orchestration
src/benchmarks.py       Predefined tests and prompt templates
src/ollama_client.py    Ollama API calls and result records
src/metrics.py          Result summaries and leaderboards
src/exports.py          CSV, JSON, and Markdown exports
src/i18n.py             English/French UI text
docker/                 Dockerfile and Compose files
run.bat                 Windows launcher
run.sh                  macOS/Linux launcher
```

## 💡 Why This Project

Local models are easier to trust when you can compare them with the same tests,
same settings, and visible prompts. This app keeps that workflow simple:
choose models, run tests, read outputs, inspect metrics, export the report.

---

## Author

- [@nixiz0](https://github.com/nixiz0)

---

## License

This project is licensed under the MIT License.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software, including for commercial use, provided that the original copyright notice and license are included.

Copyright (c) 2026 Hey Initium
