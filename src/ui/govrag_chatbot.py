import streamlit as st
import faiss
import json
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
import re


# =====================================================
# PATHS
# =====================================================

FAISS_PATH = r"E:\SRM\Project\rag_data\vectorstore\faiss_index\faiss.index"
META_PATH = r"E:\SRM\Project\rag_data\vectorstore\faiss_index\chunk_metadata.json"

MODEL_NAME = "qwen2.5:7b"


# =====================================================
# LOAD SYSTEM
# =====================================================

@st.cache_resource
def load_system():

    index = faiss.read_index(FAISS_PATH)

    metadata = json.load(
        open(META_PATH, encoding="utf-8")
    )

    encoder = SentenceTransformer(
        "intfloat/multilingual-e5-large"
    )

    return index, metadata, encoder


index, metadata, encoder = load_system()


# =====================================================
# SESSION MEMORY
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =====================================================
# DENSE RETRIEVAL
# =====================================================

def dense_search(query, top_k=5):

    emb = encoder.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )

    D, I = index.search(
        np.array(emb).astype("float32"),
        top_k
    )

    return [metadata[i] for i in I[0]]


# =====================================================
# BUILD CONTEXT
# =====================================================

def build_context(chunks):

    context = ""

    for c in chunks:
        go = c["metadata"].get("go_number", "Unknown")
        clause = c["metadata"].get("section_id", "-")

        context += f"""
GO Number: {go}
Clause: {clause}
Text:
{c['text']}
---------------------
"""

    return context


# =====================================================
# PROMPT WITH MEMORY
# =====================================================

def build_prompt(question, context):

    history_text = "\n".join([
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in st.session_state.chat_history[-3:]
    ])

    return f"""
You are GOVRAG — Tamil Nadu Government Order Assistant.

RULES:
- Answer ONLY from context.
- Respond ONLY in English or Tamil.
- Never use Chinese or other languages.
- Mention GO Number and Clause.
- If information missing say:
  "I don't have information about this Government Order."

Conversation History:
{history_text}

Context:
{context}

Question:
{question}
"""


# =====================================================
# GENERATION
# =====================================================

def generate_answer(prompt):

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    if (
        "I don't have information" in answer
        and len(answer) > 200
    ):
        answer = answer.replace(
            "I don't have information about this Government Order.",
            ""
        )

    return answer.strip()


# =====================================================
# AUTO CITATION HIGHLIGHT
# =====================================================

def highlight_citations(answer):

    answer = re.sub(
        r"(G\.O.*?No\.\s*\d+)",
        r"**📘 \1**",
        answer
    )

    answer = re.sub(
        r"(Clause\s*\d+[A-Za-z0-9().-]*)",
        r"**📌 \1**",
        answer
    )

    return answer


# =====================================================
# STREAMLIT UI
# =====================================================

st.title("🏛 GOVRAG Assistant")
st.caption(
"Dense Multilingual Retrieval | Trusted Government Order Interpretation"
)

user_query = st.text_input(
    "Ask about Tamil Nadu Government Orders"
)

if st.button("Ask") and user_query:

    with st.spinner("Retrieving Government Orders..."):

        chunks = dense_search(user_query)

        context = build_context(chunks)

        prompt = build_prompt(user_query, context)

        answer = generate_answer(prompt)

        answer = highlight_citations(answer)

        st.session_state.chat_history.append({
            "user": user_query,
            "assistant": answer
        })

    st.subheader("📜 Answer")
    st.markdown(answer)

    st.subheader("🔎 Retrieved Evidence")

    for c in chunks:

        go = c["metadata"].get("go_number", "Unknown")
        clause = c["metadata"].get("section_id", "-")

        st.markdown(f"""
---
✅ **GO:** {go}  
📌 **Clause:** {clause}

{c['text'][:400]}...
""")


# =====================================================
# CHAT HISTORY DISPLAY
# =====================================================

st.sidebar.title("Conversation Memory")

for chat in reversed(st.session_state.chat_history[-5:]):
    st.sidebar.markdown(
        f"**Q:** {chat['user']}\n\n**A:** {chat['assistant'][:120]}..."
    )