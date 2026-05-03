from __future__ import annotations

from statistics import mean, stdev
from typing import Any

import pandas as pd


DISPLAY_COLUMNS = [
    "model",
    "case_id",
    "test",
    "category",
    "language",
    "difficulty",
    "focus",
    "expected_output",
    "run",
    "temperature",
    "top_p",
    "seed",
    "wall_latency_s",
    "total_duration_s",
    "load_duration_s",
    "prompt_eval_tokens",
    "eval_tokens",
    "tokens_per_second",
    "thinking_present",
    "tool_name",
    "tool_match",
    "done_reason",
]


FINAL_BOARD_COLUMNS = [
    "model",
    "case_id",
    "test",
    "category",
    "language",
    "difficulty",
    "focus",
    "runs",
    "errors",
    "latency_mean_s",
    "tokens_per_second_mean",
    "eval_tokens_mean",
    "temperature",
    "top_p",
    "seed",
]


def results_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    df = pd.DataFrame(results)
    if "test" not in df.columns:
        df["test"] = df.get("title")
    for column in DISPLAY_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[DISPLAY_COLUMNS]


def aggregate_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(
            columns=[
                "model",
                "case_id",
                "test",
                "category",
                "language",
                "difficulty",
                "focus",
                "expected_output",
                "runs",
                "errors",
                "unique_outputs",
                "temperature",
                "top_p",
                "seed",
                "latency_mean_s",
                "latency_stdev_s",
                "total_mean_s",
                "load_mean_s",
                "tokens_per_second_mean",
                "eval_tokens_mean",
                "tool_match_rate",
                "thinking_runs",
            ]
        )

    rows = []
    grouped: dict[tuple[str, str, Any, Any, Any], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(
            (
                result["model"],
                result["case_id"],
                result.get("temperature"),
                result.get("top_p"),
                result.get("seed"),
            ),
            [],
        ).append(result)

    for (model, case_id, temperature, top_p, seed), group in grouped.items():
        latencies = _values(group, "wall_latency_s")
        total_durations = _values(group, "total_duration_s")
        load_durations = _values(group, "load_duration_s")
        speeds = _values(group, "tokens_per_second")
        eval_tokens = _values(group, "eval_tokens")
        tool_matches = [row["tool_match"] for row in group if row.get("tool_match") is not None]
        rows.append(
            {
                "model": model,
                "case_id": case_id,
                "test": group[0].get("title", case_id),
                "category": group[0]["category"],
                "language": group[0].get("language"),
                "difficulty": group[0].get("difficulty"),
                "focus": group[0].get("focus"),
                "expected_output": group[0].get("expected_output"),
                "runs": len(group),
                "errors": sum(bool(row.get("error")) for row in group),
                "unique_outputs": len({row.get("response", "") for row in group}),
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "latency_mean_s": _round_mean(latencies),
                "latency_stdev_s": round(stdev(latencies), 3) if len(latencies) > 1 else 0.0 if latencies else None,
                "total_mean_s": _round_mean(total_durations),
                "load_mean_s": _round_mean(load_durations),
                "tokens_per_second_mean": _round_mean(speeds, digits=2),
                "eval_tokens_mean": _round_mean(eval_tokens, digits=1),
                "tool_match_rate": (
                    f"{sum(bool(value) for value in tool_matches)}/{len(tool_matches)}"
                    if tool_matches
                    else "-"
                ),
                "thinking_runs": sum(bool(row.get("thinking_present")) for row in group),
            }
        )

    return pd.DataFrame(rows).sort_values(["category", "case_id", "model"])


def final_results_board(results: list[dict[str, Any]]) -> pd.DataFrame:
    summary = aggregate_results(results)
    if summary.empty:
        return pd.DataFrame(columns=FINAL_BOARD_COLUMNS)
    for column in FINAL_BOARD_COLUMNS:
        if column not in summary.columns:
            summary[column] = None
    return summary[FINAL_BOARD_COLUMNS].sort_values(
        ["errors", "latency_mean_s", "tokens_per_second_mean"],
        ascending=[True, True, False],
        na_position="last",
    )


def leaderboard(results: list[dict[str, Any]]) -> pd.DataFrame:
    summary = aggregate_results(results)
    if summary.empty:
        return summary
    model_rows = []
    for model, group in summary.groupby("model"):
        model_rows.append(
            {
                "model": model,
                "cases": len(group),
                "runs": int(group["runs"].sum()),
                "errors": int(group["errors"].sum()),
                "avg_latency_s": round(group["latency_mean_s"].dropna().mean(), 3)
                if group["latency_mean_s"].notna().any()
                else None,
                "avg_tokens_per_second": round(group["tokens_per_second_mean"].dropna().mean(), 2)
                if group["tokens_per_second_mean"].notna().any()
                else None,
                "avg_eval_tokens": round(group["eval_tokens_mean"].dropna().mean(), 1)
                if group["eval_tokens_mean"].notna().any()
                else None,
            }
        )
    return pd.DataFrame(model_rows).sort_values(
        ["errors", "avg_latency_s"], na_position="last"
    )


def _values(group: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in group if row.get(key) is not None]


def _round_mean(values: list[float], digits: int = 3) -> float | None:
    return round(mean(values), digits) if values else None
