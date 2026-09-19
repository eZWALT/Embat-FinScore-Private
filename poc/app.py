"""POC shell. Run from the repo root: `python3 -m streamlit run poc/app.py --server.port 8601`."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poc.views import ask, overview, portfolio, watcher  # noqa: E402

st.set_page_config(page_title="Health Sentinel", layout="wide")

page = st.navigation(
    [
        st.Page(overview.render, title="Overview", default=True),
        st.Page(watcher.render, title="Watcher", url_path="watcher"),
        st.Page(ask.render, title="Ask", url_path="ask"),
        st.Page(portfolio.render, title="Portfolio", url_path="portfolio"),
    ]
)
page.run()
