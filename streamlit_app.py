from __future__ import annotations

import streamlit as st

from src.config import APP_TITLE, STYLE_PATH
from src.ui import (
    init_state,
    language_selector,
    load_css,
    load_inventory,
    render_hero,
    render_sidebar,
    render_tabs,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    init_state()
    load_css(STYLE_PATH)

    lang = language_selector()
    render_hero(lang)

    settings = render_sidebar(lang)
    inventory_df = load_inventory(
        settings["base_url"],
        settings["refresh_token"],
        lang,
    )
    render_tabs(inventory_df, settings, lang)


if __name__ == "__main__":
    main()
