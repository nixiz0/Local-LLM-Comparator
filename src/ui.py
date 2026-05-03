from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .benchmarks import (
    BENCHMARK_CASES,
    benchmark_languages,
    cases_by_categories,
    cases_by_ids,
    categories,
    custom_case,
    default_tool_schema_text,
)
from .config import (
    DEFAULT_MODEL_PULL_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    DEFAULT_TEMPERATURE,
)
from .exports import results_to_csv, results_to_json, results_to_markdown
from .i18n import (
    LANGUAGE_OPTIONS,
    benchmark_language_label,
    case_label,
    case_title,
    category_label,
    column_label,
    difficulty_label,
    language_from_label,
    localized_prompt_templates,
    metric_guide_rows,
    t,
)
from .metrics import aggregate_results, final_results_board, leaderboard, results_dataframe
from .ollama_client import encode_uploaded_images, model_inventory, pull_model, run_case


def init_state() -> None:
    st.session_state.setdefault("results", [])
    st.session_state.setdefault("refresh_token", 0)


def load_css(path: Path) -> None:
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def language_selector() -> str:
    selected_language = st.sidebar.selectbox(
        "Language / Langue",
        list(LANGUAGE_OPTIONS),
        index=0,
        key="language_selector",
    )
    return language_from_label(selected_language)


def render_hero(lang: str) -> None:
    st.markdown(
        f"""
        <section class="app-hero">
            <h1 class="app-title">{t(lang, "hero_title")} <span>{t(lang, "hero_accent")}</span></h1>
            <p class="app-subtitle">{t(lang, "hero_subtitle")}</p>
            <div class="app-strip">
                <span class="app-chip">{t(lang, "chip_ollama")}</span>
                <span class="app-chip">{t(lang, "chip_multi")}</span>
                <span class="app-chip">{t(lang, "chip_vision")}</span>
                <span class="app-chip">{t(lang, "chip_export")}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(lang: str) -> dict[str, Any]:
    st.sidebar.header(t(lang, "connection"))
    base_url = st.sidebar.text_input(
        t(lang, "ollama_url"),
        value=DEFAULT_OLLAMA_URL,
        help=t(lang, "ollama_help"),
    ).rstrip("/")

    if st.sidebar.button(t(lang, "refresh_models"), width="stretch"):
        st.session_state.refresh_token += 1

    _model_download_form(base_url, lang)

    st.sidebar.header(t(lang, "run_settings"))
    repeats = st.sidebar.number_input(
        t(lang, "repeats"),
        min_value=1,
        max_value=20,
        value=1,
        help=t(lang, "repeats_help"),
    )
    temperature = st.sidebar.slider(
        t(lang, "temperature"),
        0.0,
        2.0,
        DEFAULT_TEMPERATURE,
        0.05,
        format="%.2f",
        help=t(lang, "temperature_help"),
    )
    top_p = st.sidebar.slider(
        t(lang, "top_p"),
        0.05,
        1.0,
        0.9,
        0.05,
        help=t(lang, "top_p_help"),
    )
    seed_text = st.sidebar.text_input(
        t(lang, "seed"),
        value="",
        placeholder=t(lang, "seed_placeholder"),
        help=t(lang, "seed_help"),
    )

    options: dict[str, Any] = {"temperature": temperature, "top_p": top_p}
    if seed_text.strip():
        try:
            options["seed"] = int(seed_text.strip())
        except ValueError:
            st.sidebar.warning(t(lang, "seed_warning"))

    return {
        "base_url": base_url,
        "repeats": int(repeats),
        "timeout": DEFAULT_RUN_TIMEOUT_SECONDS,
        "options": options,
        "thinking_by_model": {},
        "refresh_token": st.session_state.refresh_token,
        "lang": lang,
    }


@st.cache_data(show_spinner=False)
def cached_inventory(base_url: str, refresh_token: int) -> tuple[list[dict[str, Any]], str | None]:
    return model_inventory(base_url)


def load_inventory(base_url: str, refresh_token: int, lang: str) -> pd.DataFrame:
    with st.spinner(t(lang, "reading_models")):
        rows, error = cached_inventory(base_url, refresh_token)
    if error:
        st.warning(t(lang, "ollama_unreachable", base_url=base_url, error=error))
    return pd.DataFrame(rows)


def render_tabs(inventory_df: pd.DataFrame, settings: dict[str, Any], lang: str) -> None:
    available_models = inventory_df["name"].tolist() if not inventory_df.empty else []
    tabs = st.tabs(
        [
            t(lang, "tab_models"),
            t(lang, "tab_predefined"),
            t(lang, "tab_custom"),
            t(lang, "tab_vision"),
            t(lang, "tab_results"),
        ]
    )
    with tabs[0]:
        models_tab(inventory_df, settings["base_url"], lang)
    with tabs[1]:
        predefined_tab(available_models, inventory_df, settings, lang)
    with tabs[2]:
        custom_tab(available_models, inventory_df, settings, lang)
    with tabs[3]:
        vision_tab(available_models, inventory_df, settings, lang)
    with tabs[4]:
        results_tab(lang)


def _model_download_form(base_url: str, lang: str) -> None:
    with st.sidebar.expander(t(lang, "download_model"), expanded=False):
        model_name = st.text_input(
            t(lang, "model_to_download"),
            placeholder="qwen3:4b",
            key="model_to_download",
        )
        if st.button(t(lang, "download_model_button"), width="stretch"):
            requested_model = model_name.strip()
            if not requested_model:
                st.warning(t(lang, "download_model_empty"))
                return
            _download_model(base_url, requested_model, lang)


def _download_model(base_url: str, model_name: str, lang: str) -> None:
    progress = st.progress(0, text=t(lang, "download_model_start", model=model_name))
    status = st.empty()
    try:
        last_percent = 0
        for event in pull_model(
            base_url,
            model_name,
            timeout=DEFAULT_MODEL_PULL_TIMEOUT_SECONDS,
        ):
            percent = _pull_percent(event)
            if percent is not None:
                last_percent = max(last_percent, percent)
                progress.progress(
                    min(last_percent, 100) / 100,
                    text=t(lang, "download_model_progress", model=model_name, percent=last_percent),
                )
            if event.get("status"):
                status.caption(str(event["status"]))
        progress.progress(1.0, text=t(lang, "download_model_done", model=model_name))
        st.session_state.refresh_token += 1
        cached_inventory.clear()
    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(t(lang, "download_model_failed", model=model_name, error=exc))


def _pull_percent(event: dict[str, Any]) -> int | None:
    total = event.get("total")
    completed = event.get("completed")
    if not total or completed is None:
        return None
    return int((int(completed) * 100) / max(int(total), 1))


def models_tab(inventory_df: pd.DataFrame, base_url: str, lang: str) -> None:
    left, right = st.columns([2, 1])
    with left:
        st.subheader(t(lang, "model_browser"))
        if inventory_df.empty:
            st.info(t(lang, "empty_models"))
            st.code("ollama serve\nollama pull qwen3:4b", language="powershell")
            return
        display_cols = [
            "name",
            "size_gb",
            "family",
            "parameters",
            "quantization",
            "capabilities",
            "thinking_hint",
            "thinking_policy",
            "context",
        ]
        dataframe(inventory_df[display_cols], lang)

    with right:
        st.subheader(t(lang, "status"))
        st.metric(t(lang, "models"), len(inventory_df))
        st.metric(t(lang, "vision_capable"), int(inventory_df["vision"].sum()) if "vision" in inventory_df else 0)
        likely_thinking = inventory_df["thinking_hint"].isin(["yes", "likely"]).sum() if "thinking_hint" in inventory_df else 0
        st.metric(t(lang, "thinking_hints"), int(likely_thinking))
        st.caption(t(lang, "connected_to", base_url=base_url))

    with st.expander(t(lang, "full_metadata"), expanded=False):
        dataframe(inventory_df, lang)


def predefined_tab(
    available_models: list[str],
    inventory_df: pd.DataFrame,
    settings: dict[str, Any],
    lang: str,
) -> None:
    st.subheader(t(lang, "tab_predefined"))
    if not require_models(available_models, lang):
        return

    selected_models = model_multiselect(
        key="predefined_models",
        available_models=available_models,
        inventory_df=inventory_df,
        lang=lang,
    )
    thinking_by_model = thinking_controls(selected_models, inventory_df, lang, key="predefined_models")
    selected_cases = _selected_predefined_cases(lang)
    dataframe(_cases_dataframe(selected_cases, lang), lang)

    if st.button(t(lang, "run_predefined"), type="primary", width="stretch"):
        if not selected_cases:
            st.error(t(lang, "no_case_selected"))
            return
        run_plan = predefined_run_plan(selected_cases, selected_models, inventory_df, lang)
        run_plan_and_store(
            base_url=settings["base_url"],
            run_plan=run_plan,
            settings={**settings, "thinking_by_model": thinking_by_model},
            images=None,
        )


def custom_tab(
    available_models: list[str],
    inventory_df: pd.DataFrame,
    settings: dict[str, Any],
    lang: str,
) -> None:
    st.subheader(t(lang, "tab_custom"))
    if not require_models(available_models, lang):
        return

    selected_models = model_multiselect(
        key="custom_models",
        available_models=available_models,
        inventory_df=inventory_df,
        lang=lang,
    )
    thinking_by_model = thinking_controls(selected_models, inventory_df, lang, key="custom_models")
    case = _custom_case_form(lang)

    if st.button(t(lang, "run_custom"), type="primary", width="stretch"):
        if case is None:
            return
        run_plan_and_store(
            base_url=settings["base_url"],
            run_plan=[(selected_models, [case])],
            settings={**settings, "thinking_by_model": thinking_by_model},
            images=None,
        )


def vision_tab(
    available_models: list[str],
    inventory_df: pd.DataFrame,
    settings: dict[str, Any],
    lang: str,
) -> None:
    st.subheader(t(lang, "vision_tests"))
    if not require_models(available_models, lang):
        return

    vision_models = vision_model_names(inventory_df)
    if not vision_models:
        st.info(t(lang, "no_vision_models"))
        dataframe(inventory_df[["name", "capabilities"]], lang)
        return

    selected_models = st.multiselect(
        t(lang, "vision_models"),
        vision_models,
        default=vision_models[:1],
        key="vision_models",
    )
    thinking_by_model = thinking_controls(selected_models, inventory_df, lang, key="vision_models")
    uploaded_files = st.file_uploader(
        t(lang, "upload_images"),
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    system = st.text_area(
        t(lang, "system_prompt"),
        value=t(lang, "vision_default_system"),
        height=100,
        key=f"vision_system_{lang}",
    )
    prompt = st.text_area(
        t(lang, "vision_prompt"),
        value=t(lang, "vision_default_prompt"),
        height=150,
        key=f"vision_prompt_{lang}",
    )
    if st.button(t(lang, "run_vision"), type="primary", width="stretch"):
        if not uploaded_files:
            st.error(t(lang, "upload_required"))
            return
        case = custom_case(
            title=t(lang, "vision_tests"),
            mode="vision_generate",
            system=system,
            prompt=prompt,
            language=lang,
        )
        run_plan_and_store(
            base_url=settings["base_url"],
            run_plan=[(selected_models, [case])],
            settings={**settings, "thinking_by_model": thinking_by_model},
            images=encode_uploaded_images(uploaded_files),
        )


def results_tab(lang: str) -> None:
    st.subheader(t(lang, "results"))
    results = st.session_state.results
    if not results:
        st.info(t(lang, "no_results"))
        return

    summary = aggregate_results(results)
    board = leaderboard(results)
    final_board = final_results_board(results)
    _result_metrics(results, board, lang)
    _render_quick_read(board, lang)
    _render_simple_comparison(board, lang)
    _render_metric_guide(lang)

    st.markdown(f"### {t(lang, 'final_board')}")
    st.caption(t(lang, "final_board_help"))
    dataframe(_board_for_display(final_board, lang, compact=True), lang)

    with st.expander(t(lang, "advanced_results"), expanded=False):
        st.caption(t(lang, "advanced_results_help"))
        st.markdown(f"#### {t(lang, 'leaderboard')}")
        dataframe(board, lang)
        _result_charts(_board_for_display(summary, lang), lang)
        st.markdown(f"#### {t(lang, 'aggregate')}")
        dataframe(_board_for_display(summary, lang), lang)
        st.markdown(f"#### {t(lang, 'per_run')}")
        dataframe(_board_for_display(results_dataframe(results), lang), lang)
    _render_outputs(results, lang)
    _result_exports(results, summary, lang)


def _selected_predefined_cases(lang: str) -> list[Any]:
    selected_languages = _selected_benchmark_languages(lang)
    text_cases = [
        case
        for case in BENCHMARK_CASES
        if not case.requires_vision and case.language in selected_languages
    ]
    selection_modes = {
        t(lang, "by_category"): "category",
        t(lang, "specific_prompts"): "prompts",
    }
    mode_label = st.radio(t(lang, "selection_mode"), list(selection_modes), horizontal=True)
    if selection_modes[mode_label] == "category":
        text_categories = [
            category for category in categories()
            if any(case.category == category for case in text_cases)
        ]
        category_lookup = {category_label(lang, code): code for code in text_categories}
        default_categories = [
            category_label(lang, category)
            for category in ["summarization", "structured_extraction", "reasoning"]
            if category in text_categories
        ]
        selected_labels = st.multiselect(
            t(lang, "categories"),
            list(category_lookup),
            default=default_categories,
        )
        selected_cases = cases_by_categories(
            [category_lookup[label] for label in selected_labels],
            selected_languages,
        )
        return [case for case in selected_cases if not case.requires_vision]

    case_options = {
        (
            f"{benchmark_language_label(lang, case.language)} | "
            f"{category_label(lang, case.category)} | {case_title(lang, case)}"
        ): case.id
        for case in text_cases
    }
    selected_labels = st.multiselect(
        t(lang, "prompts"),
        list(case_options),
        default=list(case_options)[:2],
    )
    return cases_by_ids([case_options[label] for label in selected_labels])


def _selected_benchmark_languages(lang: str) -> list[str]:
    language_lookup = {
        benchmark_language_label(lang, code): code for code in benchmark_languages()
    }
    default_codes = [lang] if lang in language_lookup.values() else ["en"]
    default_labels = [
        benchmark_language_label(lang, code)
        for code in default_codes
        if benchmark_language_label(lang, code) in language_lookup
    ]
    selected_labels = st.multiselect(
        t(lang, "benchmark_languages"),
        list(language_lookup),
        default=default_labels,
        help=t(lang, "benchmark_languages_help"),
    )
    return [language_lookup[label] for label in selected_labels]


def _custom_case_form(lang: str) -> Any | None:
    templates = localized_prompt_templates(lang)
    template_name = st.selectbox(t(lang, "prompt_template"), list(templates), index=0)
    template = templates[template_name]
    mode_options = mode_options_for(lang)
    mode_label = st.selectbox(t(lang, "mode"), list(mode_options), index=0)
    mode = mode_options[mode_label]
    title = st.text_input(t(lang, "test_name"), value=t(lang, "custom_prompt_title"))
    system = st.text_area(
        t(lang, "system_prompt"),
        value=template["system"],
        height=110,
        key=f"custom_system_{lang}_{template_name}",
    )
    prompt = st.text_area(
        t(lang, "user_prompt"),
        value=template["prompt"],
        height=220,
        key=f"custom_prompt_{lang}_{template_name}",
    )

    expected_tool = ""
    tool_schema_text = ""
    if mode == "chat_tools":
        expected_tool = st.text_input(t(lang, "expected_tool_name"), value="get_weather")
        tool_schema_text = st.text_area(
            t(lang, "tool_schema_json"),
            value=default_tool_schema_text(),
            height=260,
        )
    try:
        return custom_case(
            title=title,
            mode=mode,
            system=system,
            prompt=prompt,
            tool_schema_text=tool_schema_text,
            expected_tool=expected_tool,
            language=lang,
        )
    except json.JSONDecodeError as exc:
        st.error(t(lang, "invalid_tool_json", error=exc))
        return None


def _cases_dataframe(selected_cases: list[Any], lang: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": case.id,
                "category": category_label(lang, case.category),
                "language": benchmark_language_label(lang, case.language),
                "title": case_title(lang, case),
                "difficulty": difficulty_label(lang, case.difficulty),
                "focus": case.focus,
                "expected_output": case.expected_output,
                "mode": case.mode,
                "vision": case.requires_vision,
            }
            for case in selected_cases
        ]
    )


def _result_metrics(results: list[dict[str, Any]], board: pd.DataFrame, lang: str) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric(t(lang, "total_runs"), len(results))
    metric_cols[1].metric(t(lang, "models_tested"), len({row["model"] for row in results}))
    metric_cols[2].metric(t(lang, "errors"), sum(bool(row.get("error")) for row in results))
    avg_speed = board["avg_tokens_per_second"].dropna().mean() if not board.empty else None
    metric_cols[3].metric(
        t(lang, "avg_toks"),
        "-" if avg_speed is None or pd.isna(avg_speed) else round(avg_speed, 2),
    )


def _render_quick_read(board: pd.DataFrame, lang: str) -> None:
    if board.empty:
        return

    st.markdown(f"### {t(lang, 'quick_read')}")
    st.caption(t(lang, "quick_read_help"))
    insight_cols = st.columns(3)
    insight_cols[0].metric(
        t(lang, "fewest_errors"),
        _best_metric_value(board, "errors", lower_is_better=True, suffix=""),
    )
    insight_cols[1].metric(
        t(lang, "fastest_response"),
        _best_metric_value(board, "avg_latency_s", lower_is_better=True, suffix=" s"),
    )
    insight_cols[2].metric(
        t(lang, "fastest_generation"),
        _best_metric_value(board, "avg_tokens_per_second", lower_is_better=False, suffix=" tok/s"),
    )


def _render_simple_comparison(board: pd.DataFrame, lang: str) -> None:
    simple_board = _simple_results_board(board, lang)
    if simple_board.empty:
        return

    st.markdown(f"### {t(lang, 'simple_comparison')}")
    st.caption(t(lang, "simple_comparison_help"))
    dataframe(simple_board, lang)


def _render_metric_guide(lang: str) -> None:
    with st.expander(t(lang, "metric_guide"), expanded=False):
        st.caption(t(lang, "metric_guide_help"))
        dataframe(pd.DataFrame(metric_guide_rows(lang)), lang)
        st.info(t(lang, "metric_guide_note"))


def _best_metric_value(
    board: pd.DataFrame,
    column: str,
    *,
    lower_is_better: bool,
    suffix: str,
) -> str:
    if board.empty or column not in board:
        return "-"
    values = board[["model", column]].dropna()
    if values.empty:
        return "-"

    best_value = values[column].min() if lower_is_better else values[column].max()
    best_models = values.loc[values[column] == best_value, "model"].astype(str).tolist()
    model_text = ", ".join(best_models[:2])
    if len(best_models) > 2:
        model_text = f"{model_text} +{len(best_models) - 2}"
    return f"{model_text} ({_format_metric_number(best_value)}{suffix})"


def _simple_results_board(board: pd.DataFrame, lang: str) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()

    fastest_latency = _best_model(board, "avg_latency_s", lower_is_better=True)
    fastest_generation = _best_model(board, "avg_tokens_per_second", lower_is_better=False)
    rows = []
    for _, row in board.iterrows():
        errors = int(row.get("errors") or 0)
        model = str(row.get("model") or "")
        rows.append(
            {
                "model": model,
                "status": (
                    t(lang, "no_errors_status")
                    if errors == 0
                    else t(lang, "error_count_status", count=errors)
                ),
                "runs": row.get("runs"),
                "cases": row.get("cases"),
                "average_wait_s": _format_metric_value(row.get("avg_latency_s"), " s", lang),
                "writing_speed": _format_metric_value(row.get("avg_tokens_per_second"), " tok/s", lang),
                "answer_length": _format_metric_value(row.get("avg_eval_tokens"), " tokens", lang),
                "simple_read": _simple_read_for_model(
                    lang=lang,
                    model=model,
                    errors=errors,
                    fastest_latency=fastest_latency,
                    fastest_generation=fastest_generation,
                ),
            }
        )
    return pd.DataFrame(rows)


def _best_model(board: pd.DataFrame, column: str, *, lower_is_better: bool) -> str:
    if board.empty or column not in board:
        return ""
    values = board[["model", column]].dropna()
    if values.empty:
        return ""
    best_index = values[column].idxmin() if lower_is_better else values[column].idxmax()
    return str(values.loc[best_index, "model"])


def _simple_read_for_model(
    *,
    lang: str,
    model: str,
    errors: int,
    fastest_latency: str,
    fastest_generation: str,
) -> str:
    if errors:
        return t(lang, "review_errors_read")
    if model == fastest_latency and model == fastest_generation:
        return t(lang, "fastest_overall_read")
    if model == fastest_latency:
        return t(lang, "shortest_wait_read")
    if model == fastest_generation:
        return t(lang, "fastest_generation_read")
    return t(lang, "manual_quality_read")


def _format_metric_value(value: Any, suffix: str, lang: str) -> str:
    if value is None or pd.isna(value):
        return t(lang, "not_available")
    return f"{_format_metric_number(value)}{suffix}"


def _format_metric_number(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


def _result_charts(summary: pd.DataFrame, lang: str) -> None:
    if summary.empty:
        return
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown(f"#### {t(lang, 'latency_by_model')}")
        st.bar_chart(summary, x="model", y="latency_mean_s", color="category")
    with chart_cols[1]:
        st.markdown(f"#### {t(lang, 'speed_by_model')}")
        st.bar_chart(summary, x="model", y="tokens_per_second_mean", color="category")


def _board_for_display(df: pd.DataFrame, lang: str, *, compact: bool = False) -> pd.DataFrame:
    display = df.copy()
    if display.empty:
        return display.drop(columns=["case_id"], errors="ignore") if compact else display

    if "category" in display.columns:
        display["category"] = display["category"].apply(
            lambda value: category_label(lang, str(value)) if pd.notna(value) else value
        )
    if "language" in display.columns:
        display["language"] = display["language"].apply(
            lambda value: benchmark_language_label(lang, str(value)) if pd.notna(value) else value
        )
    if "difficulty" in display.columns:
        display["difficulty"] = display["difficulty"].apply(
            lambda value: difficulty_label(lang, str(value)) if pd.notna(value) else value
        )
    if "test" in display.columns and "case_id" in display.columns:
        display["test"] = display.apply(
            lambda row: case_label(
                lang,
                str(row.get("case_id") or ""),
                str(row.get("test") or row.get("title") or ""),
            ),
            axis=1,
        )
    elif "title" in display.columns and "case_id" in display.columns:
        display["title"] = display.apply(
            lambda row: case_label(
                lang,
                str(row.get("case_id") or ""),
                str(row.get("title") or ""),
            ),
            axis=1,
        )

    return display.drop(columns=["case_id"], errors="ignore") if compact else display


def _render_outputs(results: list[dict[str, Any]], lang: str) -> None:
    st.markdown(f"### {t(lang, 'outputs')}")
    for index, result in enumerate(reversed(results), start=1):
        label = f"{result['model']} | {result_title(result, lang)} | run {result['run']}"
        with st.expander(label, expanded=False):
            if result.get("error"):
                st.error(result["error"])
            st.caption(
                f"{t(lang, 'category')}: {category_label(lang, result['category'])} | "
                f"{t(lang, 'language')}: {benchmark_language_label(lang, result.get('language', 'custom'))} | "
                f"{t(lang, 'latency')}: {result.get('wall_latency_s')} s | "
                f"{t(lang, 'tokens_sec')}: {result.get('tokens_per_second')} | "
                f"{t(lang, 'temperature')}: {result.get('temperature')} | "
                f"{t(lang, 'top_p')}: {result.get('top_p')}"
            )
            _render_prompt_used(result, lang)
            if result.get("thinking"):
                st.markdown(f"#### {t(lang, 'thinking')}")
                st.code(result["thinking"])
            st.markdown(f"#### {t(lang, 'response')}")
            st.markdown(result.get("response") or "")
            if result.get("tool_name"):
                st.markdown(f"#### {t(lang, 'tool_call')}")
                st.json(
                    {
                        "tool_name": result.get("tool_name"),
                        "arguments": result.get("tool_arguments"),
                        "expected": result.get("expected_tool"),
                        "match": result.get("tool_match"),
                    }
                )
            with st.expander(t(lang, "raw_response"), expanded=False):
                st.json(result.get("raw_response"))


def _render_prompt_used(result: dict[str, Any], lang: str) -> None:
    case = result.get("case") if isinstance(result.get("case"), dict) else {}
    system = str(case.get("system") or result.get("system") or "").strip()
    prompt = str(case.get("prompt") or result.get("prompt") or "").strip()
    if not system and not prompt:
        return

    st.markdown(f"#### {t(lang, 'prompt_used')}")
    if system:
        st.markdown(f"**{t(lang, 'system_instruction')}**")
        st.code(system, language="text")
    if prompt:
        st.markdown(f"**{t(lang, 'user_prompt')}**")
        st.code(prompt, language="text")


def _result_exports(results: list[dict[str, Any]], summary: pd.DataFrame, lang: str) -> None:
    st.markdown(f"### {t(lang, 'export')}")
    export_cols = st.columns(4)
    export_cols[0].download_button(
        t(lang, "download_csv"),
        results_to_csv(results),
        file_name="llm-results.csv",
        mime="text/csv",
        width="stretch",
    )
    export_cols[1].download_button(
        t(lang, "download_json"),
        results_to_json(results),
        file_name="llm-results.json",
        mime="application/json",
        width="stretch",
    )
    export_cols[2].download_button(
        t(lang, "download_report"),
        results_to_markdown(results, summary),
        file_name="llm-report.md",
        mime="text/markdown",
        width="stretch",
    )
    if export_cols[3].button(t(lang, "clear_results"), width="stretch"):
        st.session_state.results = []
        st.rerun()


def dataframe(df: pd.DataFrame, lang: str) -> None:
    localized = df.rename(columns={column: column_label(lang, str(column)) for column in df.columns})
    localized.columns = _unique_columns([str(column) for column in localized.columns])
    st.dataframe(localized, width="stretch", hide_index=True)


def _unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for column in columns:
        count = seen.get(column, 0)
        seen[column] = count + 1
        unique.append(column if count == 0 else f"{column} {count + 1}")
    return unique


def mode_options_for(lang: str) -> dict[str, str]:
    return {
        t(lang, "text_generation"): "generate",
        t(lang, "tool_calling_mode"): "chat_tools",
    }


def result_title(result: dict[str, Any], lang: str) -> str:
    lookup = {case.id: case for case in BENCHMARK_CASES}
    case = lookup.get(result.get("case_id", ""))
    if case is not None:
        return case_title(lang, case)
    return str(result.get("title") or "")


def require_models(available_models: list[str], lang: str) -> bool:
    if available_models:
        return True
    st.info(t(lang, "empty_models"))
    return False


def vision_model_names(inventory_df: pd.DataFrame) -> list[str]:
    if inventory_df.empty or "vision" not in inventory_df:
        return []
    return inventory_df.loc[inventory_df["vision"], "name"].tolist()


def model_multiselect(
    *,
    key: str,
    available_models: list[str],
    inventory_df: pd.DataFrame,
    lang: str,
    default_count: int = 2,
) -> list[str]:
    filter_options = {
        t(lang, "filter_all"): "all",
        t(lang, "filter_thinking"): "thinking",
        t(lang, "filter_no_thinking"): "no_thinking",
        t(lang, "filter_switchable"): "switchable",
    }
    filter_label = st.radio(
        t(lang, "model_filter"),
        list(filter_options),
        horizontal=True,
        key=f"{key}_filter",
    )
    candidates = filtered_models(
        available_models,
        inventory_df,
        filter_options[filter_label],
    )
    if not candidates:
        st.info(t(lang, "no_models_for_filter"))
        return []

    default = candidates[: min(default_count, len(candidates))]
    return st.multiselect(
        t(lang, "models_to_compare"),
        candidates,
        default=default,
        key=f"{key}_{filter_options[filter_label]}",
    )


def filtered_models(
    available_models: list[str],
    inventory_df: pd.DataFrame,
    filter_mode: str,
) -> list[str]:
    if filter_mode == "all" or inventory_df.empty:
        return available_models

    selected = []
    for model in available_models:
        policy = thinking_policy_for_model(inventory_df, model)
        if filter_mode == "thinking" and policy in {"switchable", "likely_switchable", "always_on"}:
            selected.append(model)
        elif filter_mode == "switchable" and policy in {"switchable", "likely_switchable"}:
            selected.append(model)
        elif filter_mode == "no_thinking" and policy == "none":
            selected.append(model)
    return selected


def thinking_controls(
    selected_models: list[str],
    inventory_df: pd.DataFrame,
    lang: str,
    key: str,
) -> dict[str, bool | None]:
    if not selected_models:
        return {}

    switchable = [
        model
        for model in selected_models
        if thinking_policy_for_model(inventory_df, model) in {"switchable", "likely_switchable"}
    ]
    always_on = [
        model
        for model in selected_models
        if thinking_policy_for_model(inventory_df, model) == "always_on"
    ]
    no_config = [
        model for model in selected_models if model not in switchable and model not in always_on
    ]

    thinking_by_model: dict[str, bool | None] = {}
    with st.expander(t(lang, "thinking_behavior"), expanded=bool(switchable)):
        st.caption(t(lang, "thinking_help"))
        if switchable:
            st.info(t(lang, "thinking_switchable_note", models=", ".join(switchable)))
            thinking_options = {
                t(lang, "think_default"): None,
                t(lang, "think_enable"): True,
                t(lang, "think_disable"): False,
            }
            choice = st.selectbox(
                t(lang, "thinking_behavior"),
                list(thinking_options),
                key=f"{key}_thinking_behavior",
            )
            thinking_by_model.update({model: thinking_options[choice] for model in switchable})
        if always_on:
            st.caption(t(lang, "thinking_always_on_note", models=", ".join(always_on)))
        if no_config:
            st.caption(t(lang, "thinking_none_note", models=", ".join(no_config)))
    return thinking_by_model


def thinking_policy_for_model(inventory_df: pd.DataFrame, model: str) -> str:
    if inventory_df.empty or "name" not in inventory_df:
        return "none"
    row = inventory_df.loc[inventory_df["name"] == model]
    if row.empty:
        return "none"
    policy = row.iloc[0].get("thinking_policy", "none")
    if policy in {"switchable", "likely_switchable", "always_on", "none"}:
        return str(policy)
    hint = row.iloc[0].get("thinking_hint", "unknown")
    return "likely_switchable" if hint in {"yes", "likely"} else "none"


def predefined_run_plan(
    selected_cases: list[Any],
    selected_models: list[str],
    inventory_df: pd.DataFrame,
    lang: str,
) -> list[tuple[list[str], list[Any]]]:
    text_cases = [case for case in selected_cases if not case.requires_vision]
    vision_cases = [case for case in selected_cases if case.requires_vision]
    run_plan: list[tuple[list[str], list[Any]]] = []
    if text_cases:
        run_plan.append((selected_models, text_cases))
    if not vision_cases:
        return run_plan

    vision_names = set(vision_model_names(inventory_df))
    selected_vision_models = [model for model in selected_models if model in vision_names]
    if not selected_vision_models:
        st.warning(t(lang, "vision_skipped"))
    else:
        skipped_models = sorted(set(selected_models) - set(selected_vision_models))
        if skipped_models:
            st.warning(t(lang, "vision_limited", models=", ".join(selected_vision_models)))
        run_plan.append((selected_vision_models, vision_cases))
    return run_plan


def run_plan_and_store(
    *,
    base_url: str,
    run_plan: list[tuple[list[str], list[Any]]],
    settings: dict[str, Any],
    images: list[str] | None,
) -> None:
    lang = settings["lang"]
    normalized_plan = [(models, cases) for models, cases in run_plan if models and cases]
    if not normalized_plan:
        st.error(t(lang, "no_model_selected"))
        return

    repeats = settings["repeats"]
    total_steps = sum(
        len(models) * len(cases) * repeats for models, cases in normalized_plan
    )
    progress = st.progress(0, text=t(lang, "starting_runs"))
    new_results: list[dict[str, Any]] = []
    step = 0
    for models, cases in normalized_plan:
        for model in models:
            for case in cases:
                for repeat_index in range(1, repeats + 1):
                    current_step = step + 1
                    percent = int((step * 100) / total_steps)
                    progress.progress(
                        percent / 100,
                        text=t(
                            lang,
                            "running_case",
                            model=model,
                            case_title=case_title(lang, case),
                            percent=percent,
                            step=current_step,
                            total_steps=total_steps,
                        ),
                    )
                    repeat_results = run_case(
                        base_url=base_url,
                        model=model,
                        case=case,
                        options=settings["options"],
                        repeats=1,
                        timeout=settings["timeout"],
                        think=settings.get("thinking_by_model", {}).get(model),
                        images=images if case.requires_vision or case.mode == "vision_generate" else None,
                    )
                    for result in repeat_results:
                        result["run"] = repeat_index
                    new_results.extend(repeat_results)
                    step += 1
                    percent = int((step * 100) / total_steps)
                    progress.progress(
                        percent / 100,
                        text=t(
                            lang,
                            "running_case",
                            model=model,
                            case_title=case_title(lang, case),
                            percent=percent,
                            step=step,
                            total_steps=total_steps,
                        ),
                    )
    progress.empty()
    st.session_state.results.extend(new_results)
    st.success(t(lang, "run_added", count=len(new_results)))
