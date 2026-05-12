import streamlit as st
from retriever import dense_search
from generator import generate_answer

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="GOVRAG Chatbot",
    layout="wide"
)

st.title("📜 GOVRAG — Tamil Nadu Government Orders Assistant")

# ===============================
# SESSION MEMORY
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# USER INPUT
# ===============================
query = st.text_input(
    "Ask about Tamil Nadu Government Orders",
    placeholder="Example: agriculture subsidy schemes after 2020"
)

# ===============================
# ASK BUTTON
# ===============================
if st.button("Ask"):

    if query.strip() == "":
        st.warning("Enter a question")
        st.stop()

    with st.spinner("Retrieving Government Orders..."):

        try:
            # ---- Retrieval ----
            retrieved_chunks = dense_search(query, top_k=5)

            # ---- Generation ----
            answer, citations = generate_answer(
                query,
                retrieved_chunks
            )

            st.session_state.history.append(
                (query, answer, citations)
            )

        except Exception as e:
            st.error(f"System Error: {e}")

# ===============================
# DISPLAY CHAT
# ===============================
for q, ans, cites in reversed(st.session_state.history):

    st.markdown("### 🧑 Question")
    st.write(q)

    st.markdown("### 📜 Answer")
    st.write(ans)

    st.markdown("#### 🔎 Retrieved Evidence")

    for c in cites:

        st.markdown(
            f"""
    ✅ **{c.get('go_number','Unknown GO')}**

    - **Department:** {c.get('department','N/A')}
    - **Year:** {c.get('year','N/A')}
    - **Clause:** {c.get('clause','-')}
    """
        )

    st.divider()