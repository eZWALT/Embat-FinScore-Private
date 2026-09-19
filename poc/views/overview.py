import streamlit as st

from poc import llm
from poc.bundle import load


def render() -> None:
    st.title("Health Sentinel")
    st.write(
        "Company health from the treasury trail: a 0–100 score with its trajectory, "
        "the reasons behind each move with the € behind them, and an alert when a move is material and persistent. "
        "For the finance team of a group that already runs on Embat."
    )

    st.subheader("Three parts")
    st.write("**Portfolio** — group health map: groups, members, one company in depth.")
    st.write("**Watcher** — push: novelties on a chosen set of companies and groups, on a schedule, with a digest.")
    st.write("**Ask** — pull: questions about a company or group, answered from the bundle and the cleaned records.")

    st.subheader("Status")
    try:
        b = load()
        m = b.manifest
        st.write(
            f"Bundle: `{b.root}` · scorecard {m['scorecard_version']} · as-of {m['as_of_month']} · "
            f"{m['counts']['companies']} companies · {len(b.alerts)} alerts in the feed"
            + (" · sample" if m.get("is_sample") else "")
        )
        st.caption(m["disclaimer"])
    except FileNotFoundError as e:
        st.write(str(e))
    st.write(f"LLM: `{llm.model_name()}` via Helmcode" + ("" if llm.api_key() else " · **no key set** (HELMCODE_API_KEY)"))
