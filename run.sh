#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

MODE="${1:-auto}"
REQUESTED_MODE="$MODE"
APP_PORT="${APP_PORT:-8501}"
APP_URL="${APP_URL:-http://localhost:${APP_PORT}}"
HOST_OLLAMA_URL="${HOST_OLLAMA_URL:-http://127.0.0.1:11434}"
COMPOSE_DIR="docker"

info() {
    printf '%s\n' "$*"
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

has_command() {
    command -v "$1" >/dev/null 2>&1
}

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    elif has_command docker-compose; then
        docker-compose "$@"
    else
        fail "Docker Compose is not installed. Install Docker Desktop or Docker Engine with the Compose plugin."
    fi
}

usage() {
    cat <<'EOF'
Usage:
  sh run.sh [auto|native|cpu|nvidia|amd]

Modes:
  auto     use the containerized Ollama stack; selects nvidia, amd, then cpu when supported.
  native   run only the app in Docker and use Ollama from the host.
  cpu      run app + Ollama CPU containers.
  nvidia   run app + Ollama container with NVIDIA GPU support.
  amd      run app + Ollama ROCm container for AMD GPUs on Linux.
EOF
}

if [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
    usage
    exit 0
fi

case "$MODE" in
    auto|native|cpu|nvidia|amd)
        ;;
    *)
        usage
        fail "Unknown mode: $MODE"
        ;;
esac

has_command docker || fail "Docker is not installed. Install Docker Desktop or Docker Engine first."
docker info >/dev/null 2>&1 || fail "Docker is not running. Start Docker and run this script again."

http_ok() {
    url="$1"
    if has_command curl; then
        curl -fsS "$url/api/tags" >/dev/null 2>&1
    elif has_command wget; then
        wget -qO- "$url/api/tags" >/dev/null 2>&1
    else
        return 1
    fi
}

start_native_ollama() {
    if http_ok "$HOST_OLLAMA_URL"; then
        return 0
    fi

    has_command ollama || fail "Ollama is not installed or not in PATH. Install it from https://ollama.com/download, then run this script again."

    mkdir -p .docker
    info "Starting native Ollama on the host..."
    nohup ollama serve > .docker/ollama-native.log 2>&1 &
    sleep 5

    if ! http_ok "$HOST_OLLAMA_URL"; then
        fail "Ollama did not answer at $HOST_OLLAMA_URL. Check .docker/ollama-native.log."
    fi
}

detect_mode() {
    if [ "$MODE" != "auto" ]; then
        printf '%s\n' "$MODE"
        return 0
    fi

    os_name="$(uname -s)"
    case "$os_name" in
        Darwin*)
            printf '%s\n' "cpu"
            ;;
        Linux*)
            if has_command nvidia-smi; then
                printf '%s\n' "nvidia"
            elif [ -e /dev/kfd ] && [ -e /dev/dri ]; then
                printf '%s\n' "amd"
            else
                printf '%s\n' "cpu"
            fi
            ;;
        *)
            printf '%s\n' "cpu"
            ;;
    esac
}

open_browser() {
    if [ "${NO_BROWSER:-}" = "1" ]; then
        return 0
    fi

    os_name="$(uname -s)"
    case "$os_name" in
        Darwin*)
            open "$APP_URL" >/dev/null 2>&1 || true
            ;;
        Linux*)
            if has_command xdg-open; then
                xdg-open "$APP_URL" >/dev/null 2>&1 || true
            fi
            ;;
    esac
}

MODE="$(detect_mode)"
FILES="-f ${COMPOSE_DIR}/compose.yaml"

case "$MODE" in
    native)
        export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
        start_native_ollama
        info "Mode: native Ollama on host"
        ;;
    cpu)
        FILES="$FILES -f ${COMPOSE_DIR}/compose.ollama.yaml"
        info "Mode: Ollama CPU container"
        ;;
    nvidia)
        FILES="$FILES -f ${COMPOSE_DIR}/compose.ollama.yaml -f ${COMPOSE_DIR}/compose.nvidia.yaml"
        info "Mode: Ollama NVIDIA GPU container"
        info "This requires NVIDIA drivers and the NVIDIA Container Toolkit."
        ;;
    amd)
        FILES="$FILES -f ${COMPOSE_DIR}/compose.ollama.yaml -f ${COMPOSE_DIR}/compose.amd.yaml"
        info "Mode: Ollama AMD ROCm container"
        info "This requires Linux, ROCm-compatible AMD hardware, and access to /dev/kfd and /dev/dri."
        ;;
    *)
        usage
        fail "Unknown mode: $MODE"
        ;;
esac

info "Building and starting containers..."
if ! compose $FILES up --build -d; then
    if [ "$REQUESTED_MODE" = "auto" ] && { [ "$MODE" = "nvidia" ] || [ "$MODE" = "amd" ]; }; then
        info "GPU container mode failed. Retrying with the CPU Ollama container."
        MODE="cpu"
        FILES="-f ${COMPOSE_DIR}/compose.yaml -f ${COMPOSE_DIR}/compose.ollama.yaml"
        compose $FILES up --build -d
    else
        fail "Docker Compose failed to start."
    fi
fi

if [ "$MODE" = "nvidia" ]; then
    if compose $FILES exec -T ollama nvidia-smi >/dev/null 2>&1; then
        info "NVIDIA GPU is visible inside the Ollama container."
    else
        info "Warning: the Ollama container started, but NVIDIA GPU access was not visible inside it."
        info "Check Docker GPU support and NVIDIA Container Toolkit, then try: sh run.sh nvidia"
        info "Diagnostic command: docker compose $FILES exec -T ollama nvidia-smi"
    fi
fi

open_browser

info "App: $APP_URL"
info "Use 'docker compose $FILES logs -f' to view logs."
