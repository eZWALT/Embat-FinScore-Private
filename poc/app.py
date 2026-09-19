"""POC shell. Run from the repo root: `streamlit run poc/app.py`."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poc.views import overview, portfolio, sentinel  # noqa: E402

st.set_page_config(page_title="Embat X Ray", layout="wide")

page = st.navigation(
    [
        st.Page(overview.render, title="Overview", default=True),
        st.Page(sentinel.render, title="Sentinel", url_path="sentinel"),
        st.Page(portfolio.render, title="Portfolio", url_path="portfolio"),
    ]
)
page.run()
