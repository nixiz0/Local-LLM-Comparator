from __future__ import annotations

import json
from typing import Any

import pandas as pd


def results_to_json(results: list[dict[str, Any]]) -> str:
    return json.dumps(results, indent=2, ensure_ascii=False, default=str)


def results_to_csv(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    export_rows = []
    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key not in {"raw_response", "case"}
        }
        export_rows.append(row)
    return pd.DataFrame(export_rows).to_csv(index=False)


def results_to_markdown(results: list[dict[str, Any]], summary: pd.DataFrame) -> str:
    lines = ["# Local LLM Evaluation Report", ""]
    if not summary.empty:
        lines.extend(["## Summary", "", _dataframe_to_markdown(summary), ""])
    lines.append("## Runs")
    for result in results:
        lines.extend(
            [
                "",
                f"### {result['model']} | {result['title']} | run {result['run']}",
                "",
                f"- Category: `{result['category']}`",
                f"- Language: `{result.get('language')}`",
                f"- Difficulty: `{result.get('difficulty')}`",
                f"- Comparison focus: `{result.get('focus')}`",
                f"- Expected output: `{result.get('expected_output')}`",
                f"- Temperature: `{result.get('temperature')}`",
                f"- Top-p: `{result.get('top_p')}`",
                f"- Seed: `{result.get('seed')}`",
                f"- Latency: `{result.get('wall_latency_s')}` seconds",
                f"- Tokens/sec: `{result.get('tokens_per_second')}`",
                f"- Eval tokens: `{result.get('eval_tokens')}`",
                f"- Thinking present: `{result.get('thinking_present')}`",
                "",
                "#### Response",
                "",
                result.get("response", ""),
            ]
        )
        if result.get("thinking"):
            lines.extend(["", "#### Thinking", "", result["thinking"]])
    return "\n".join(lines)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_clean_cell(row[column]) for column in df.columns) + " |")
    return "\n".join([header, separator, *rows])


def _clean_cell(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")
