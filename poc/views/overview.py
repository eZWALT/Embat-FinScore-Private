import streamlit as st

from poc import data


def render() -> None:
    st.title("Embat X Ray")
    st.write(
        "Company health from the treasury trail: a 0–100 score with its trajectory, "
        "the reasons behind each move, and an alert when a move is material and persistent. "
        "Built for the finance team of a group that already runs on Embat."
    )

    st.subheader("What this POC is")
    st.write(
        "A place to iterate on two things before they move to `product/`: "
        "the assistant (**Sentinel**) and the group view (**Portfolio**)."
    )

    st.subheader("Status")
    if data.store_available():
        st.write("Feature store found. Portfolio shows runway and cash flows per company and group.")
    else:
        st.write("Feature store not built. Run `python -m analysis.features.build_feature_store`.")
    if data.scores_available():
        st.write("Health score 0–100: v0 dummy card loaded (`product/score/`). Train-only percentiles, not validated.")
    else:
        st.write("Health score 0–100: run `PYTHONPATH=. python -m product.score`.")
    st.write("Sentinel behaviour: not defined yet.")
