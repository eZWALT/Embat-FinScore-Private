import streamlit as st

PLACEHOLDER = "Behaviour not defined yet. Your message was recorded in this session only."


def render() -> None:
    st.title("Sentinel")
    st.caption("Assistant. No behaviour, tools or model chosen yet; this is the interface only.")

    if "sentinel_messages" not in st.session_state:
        st.session_state.sentinel_messages = []

    for m in st.session_state.sentinel_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Ask about a company or a group")
    if prompt:
        st.session_state.sentinel_messages.append({"role": "user", "content": prompt})
        st.session_state.sentinel_messages.append({"role": "assistant", "content": PLACEHOLDER})
        st.rerun()

    if st.session_state.sentinel_messages and st.button("Clear"):
        st.session_state.sentinel_messages = []
        st.rerun()
