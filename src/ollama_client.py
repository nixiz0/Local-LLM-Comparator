from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from collections.abc import Iterator
from typing import Any

from .benchmarks import BenchmarkCase


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 10,
) -> tuple[dict[str, Any], float]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def list_models(base_url: str, timeout: int = 5) -> list[dict[str, Any]]:
    body, _ = _json_request(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    return body.get("models", [])


def show_model(base_url: str, model: str, timeout: int = 15) -> dict[str, Any]:
    body, _ = _json_request(
        f"{base_url.rstrip('/')}/api/show",
        method="POST",
        payload={"model": model, "verbose": False},
        timeout=timeout,
    )
    return body


def list_running_models(base_url: str, timeout: int = 5) -> list[dict[str, Any]]:
    body, _ = _json_request(f"{base_url.rstrip('/')}/api/ps", timeout=timeout)
    return body.get("models", [])


def pull_model(
    base_url: str,
    model_name: str,
    timeout: int = 3600,
) -> Iterator[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=json.dumps({"name": model_name}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                yield json.loads(line)


def model_inventory(base_url: str, timeout: int = 5) -> tuple[list[dict[str, Any]], str | None]:
    try:
        models = list_models(base_url, timeout=timeout)
    except Exception as exc:
        return [], str(exc)

    inventory: list[dict[str, Any]] = []
    for model in models:
        name = model.get("name") or model.get("model")
        details = model.get("details") or {}
        show_payload: dict[str, Any] = {}
        capabilities: list[str] = []
        model_info: dict[str, Any] = {}
        if name:
            try:
                show_payload = show_model(base_url, name, timeout=timeout)
                capabilities = show_payload.get("capabilities") or []
                model_info = show_payload.get("model_info") or {}
                details = show_payload.get("details") or details
            except Exception:
                show_payload = {}

        inventory.append(
            {
                "name": name or "-",
                "size_gb": round((model.get("size") or 0) / (1024**3), 2),
                "family": details.get("family") or "-",
                "families": ", ".join(details.get("families") or []) or "-",
                "parameters": details.get("parameter_size") or "-",
                "quantization": details.get("quantization_level") or "-",
                "format": details.get("format") or "-",
                "capabilities": ", ".join(capabilities) if capabilities else "-",
                "vision": "vision" in capabilities,
                "thinking_hint": _thinking_hint(name or "", capabilities, show_payload),
                "thinking_policy": _thinking_policy(
                    name or "", capabilities, show_payload
                ),
                "context": _first_model_info_value(model_info, "context_length"),
                "architecture": _first_model_info_value(model_info, "architecture"),
                "modified_at": model.get("modified_at") or show_payload.get("modified_at") or "-",
            }
        )
    return inventory, None


def _first_model_info_value(model_info: dict[str, Any], suffix: str) -> Any:
    for key, value in model_info.items():
        if key.endswith(suffix):
            return value
    return "-"


def _thinking_hint(model_name: str, capabilities: list[str], show_payload: dict[str, Any]) -> str:
    policy = _thinking_policy(model_name, capabilities, show_payload)
    if policy in {"switchable", "always_on"}:
        return "yes"
    if policy == "likely_switchable":
        return "likely"
    return "unknown"


def _thinking_policy(
    model_name: str,
    capabilities: list[str],
    show_payload: dict[str, Any],
) -> str:
    lowered = model_name.lower()
    if "deepseek-r1" in lowered:
        return "always_on"
    if any(token in lowered for token in ["qwen3", "deepseek-r1", "deepseek-v3.1", "gpt-oss"]):
        return "switchable"
    if "thinking" in capabilities:
        return "switchable"
    parameters = (show_payload.get("parameters") or "").lower()
    if "think" in parameters:
        return "likely_switchable"
    return "none"


def encode_uploaded_images(uploaded_files: list[Any]) -> list[str]:
    encoded: list[str] = []
    for uploaded_file in uploaded_files:
        encoded.append(base64.b64encode(uploaded_file.getvalue()).decode("utf-8"))
    return encoded


def run_case(
    *,
    base_url: str,
    model: str,
    case: BenchmarkCase,
    options: dict[str, Any],
    repeats: int,
    timeout: int,
    think: bool | str | None = None,
    images: list[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_index in range(1, repeats + 1):
        try:
            payload, endpoint = _build_payload(
                model=model,
                case=case,
                options=options,
                think=think,
                images=images,
            )
            response, wall_seconds = _json_request(
                f"{base_url.rstrip('/')}{endpoint}",
                method="POST",
                payload=payload,
                timeout=timeout,
            )
            records.append(
                _result_record(
                    model=model,
                    case=case,
                    run_index=run_index,
                    options=options,
                    response=response,
                    wall_seconds=wall_seconds,
                    error=None,
                )
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            records.append(_error_record(model, case, run_index, f"HTTP error: {detail or exc.reason}", options))
        except Exception as exc:
            records.append(_error_record(model, case, run_index, f"Run failed: {exc}", options))
    return records


def _build_payload(
    *,
    model: str,
    case: BenchmarkCase,
    options: dict[str, Any],
    think: bool | str | None,
    images: list[str] | None,
) -> tuple[dict[str, Any], str]:
    if case.mode == "chat_tools":
        messages = []
        if case.system:
            messages.append({"role": "system", "content": case.system})
        messages.append({"role": "user", "content": case.prompt})
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": messages,
            "tools": case.tools or [],
            "options": options,
        }
        if think is not None:
            payload["think"] = think
        return payload, "/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "prompt": case.prompt,
        "system": case.system,
        "options": options,
    }
    if think is not None:
        payload["think"] = think
    if images:
        payload["images"] = images
    return payload, "/api/generate"


def _result_record(
    *,
    model: str,
    case: BenchmarkCase,
    run_index: int,
    options: dict[str, Any],
    response: dict[str, Any],
    wall_seconds: float,
    error: str | None,
) -> dict[str, Any]:
    message = response.get("message") or {}
    tool_name, tool_arguments = _extract_tool_call(message)
    response_text = response.get("response") or message.get("content") or ""
    thinking = response.get("thinking") or message.get("thinking") or ""
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    tokens_per_second = None
    if eval_count and eval_duration:
        seconds = eval_duration / 1_000_000_000
        tokens_per_second = eval_count / seconds if seconds else None

    expected_tool = case.expected_tool
    return {
        "model": model,
        "case_id": case.id,
        "title": case.title,
        "category": case.category,
        "language": case.language,
        "difficulty": case.difficulty,
        "focus": case.focus,
        "expected_output": case.expected_output,
        "mode": case.mode,
        "run": run_index,
        "temperature": options.get("temperature"),
        "top_p": options.get("top_p"),
        "seed": options.get("seed"),
        "error": error,
        "response": response_text.strip() or "(empty response)",
        "thinking": thinking.strip(),
        "thinking_present": bool(thinking.strip()),
        "tool_name": tool_name,
        "tool_arguments": json.dumps(tool_arguments, ensure_ascii=False) if tool_arguments else "",
        "expected_tool": expected_tool or "",
        "tool_match": tool_name == expected_tool if expected_tool else None,
        "wall_latency_s": wall_seconds,
        "total_duration_s": _ns_to_s(response.get("total_duration")),
        "load_duration_s": _ns_to_s(response.get("load_duration")),
        "prompt_eval_duration_s": _ns_to_s(response.get("prompt_eval_duration")),
        "eval_duration_s": _ns_to_s(eval_duration),
        "prompt_eval_tokens": response.get("prompt_eval_count"),
        "eval_tokens": eval_count,
        "tokens_per_second": tokens_per_second,
        "done_reason": response.get("done_reason") or message.get("done_reason") or "",
        "raw_response": response,
        "case": asdict(case),
    }


def _error_record(
    model: str,
    case: BenchmarkCase,
    run_index: int,
    error: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    return {
        "model": model,
        "case_id": case.id,
        "title": case.title,
        "category": case.category,
        "language": case.language,
        "difficulty": case.difficulty,
        "focus": case.focus,
        "expected_output": case.expected_output,
        "mode": case.mode,
        "run": run_index,
        "temperature": options.get("temperature"),
        "top_p": options.get("top_p"),
        "seed": options.get("seed"),
        "error": error,
        "response": error,
        "thinking": "",
        "thinking_present": False,
        "tool_name": "",
        "tool_arguments": "",
        "expected_tool": case.expected_tool or "",
        "tool_match": None,
        "wall_latency_s": None,
        "total_duration_s": None,
        "load_duration_s": None,
        "prompt_eval_duration_s": None,
        "eval_duration_s": None,
        "prompt_eval_tokens": None,
        "eval_tokens": None,
        "tokens_per_second": None,
        "done_reason": "error",
        "raw_response": None,
        "case": asdict(case),
    }


def _extract_tool_call(message: dict[str, Any]) -> tuple[str, Any]:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return "", None
    function_payload = (tool_calls[0] or {}).get("function") or {}
    return function_payload.get("name") or "", function_payload.get("arguments")


def _ns_to_s(value: int | float | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000_000
