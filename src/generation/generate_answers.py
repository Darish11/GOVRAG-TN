import json
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
import os

# ================= PATHS =================
FAISS_INDEX_PATH = r"E:\SRM\Project\rag_data\vectorstore\faiss_index\faiss.index"
CHUNKS_PATH = r"E:\SRM\Project\rag_data\vectorstore\faiss_index\chunk_metadata.json"
GOLD_QA_PATH = r"E:\SRM\Project\evaluation\gold_qa\go_gold_qa.json"
OUTPUT_DIR = r"E:\SRM\Project\evaluation"

TOP_K = 20
GENERATION_TOP_K = 10
RRF_K = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD =================
print("Loading FAISS...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print("Loading Gold QAs...")
with open(GOLD_QA_PATH, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

print("Loading dense model...")
dense_model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Building BM25 corpus...")
tokenized_corpus = [
    chunk["text"].lower().split()
    for chunk in all_chunks
]
bm25 = BM25Okapi(tokenized_corpus)

# ================= RETRIEVAL FUNCTIONS =================

def dense_retrieve(query):
    q_emb = dense_model.encode(f"query: {query}", normalize_embeddings=True)
    D, I = index.search(np.array([q_emb]), TOP_K)
    return I[0]

def bm25_retrieve(query):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    return top_indices

def rrf_fusion(dense_ids, bm25_ids):
    scores = {}
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
    for rank, idx in enumerate(bm25_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:TOP_K]]

def rerank(query, candidate_ids):
    pairs = [(query, all_chunks[idx]["text"]) for idx in candidate_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:GENERATION_TOP_K]]

# ================= PROMPT =================

def build_prompt(context_chunks, question, go_number):

    context = "\n\n".join(chunk["text"] for chunk in context_chunks)

    return f"""
You are a legal assistant answering strictly from Government Order documents.

INSTRUCTIONS:
- Use ONLY the context.
- Start answer with: "According to {go_number},"
- Mention exact clause or paragraph.
- If not found, respond EXACTLY:
"I do not have information about this GO."

--------------------------
CONTEXT:
{context}
--------------------------

QUESTION:
{question}

ANSWER:
"""

# ================= GENERATION DRIVER =================

def run_method(method_name):

    print(f"\nRunning method: {method_name}")
    results = []

    for item in gold_data:

        question = item["question"]
        go_number = item["go_number"]

        # --- Retrieval ---
        dense_ids = dense_retrieve(question)
        bm25_ids = bm25_retrieve(question)

        if method_name == "Dense":
            final_ids = dense_ids[:GENERATION_TOP_K]

        elif method_name == "Dense+Rerank":
            final_ids = rerank(question, dense_ids)

        elif method_name == "BM25":
            final_ids = bm25_ids[:GENERATION_TOP_K]

        elif method_name == "Hybrid":
            hybrid_ids = rrf_fusion(dense_ids, bm25_ids)
            final_ids = hybrid_ids[:GENERATION_TOP_K]

        elif method_name == "Hybrid+Rerank":
            hybrid_ids = rrf_fusion(dense_ids, bm25_ids)
            final_ids = rerank(question, hybrid_ids)

        else:
            raise ValueError("Unknown method")

        context_chunks = [all_chunks[idx] for idx in final_ids]
        prompt = build_prompt(context_chunks, question, go_number)

        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}]
        )

        generated_answer = response["message"]["content"]

        results.append({
            "id": item["id"],
            "question": question,
            "go_number": go_number,
            "expected_answer": item.get("expected_answer", ""),
            "expected_clauses": item.get("expected_clauses", []),
            "answer_type": item.get("answer_type", "text"),
            "generated_answer": generated_answer
        })

    output_path = os.path.join(OUTPUT_DIR, f"generated_{method_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {method_name} results.")

# ================= RUN ALL =================

if __name__ == "__main__":

    methods = [
        "Dense",
        "Dense+Rerank",
        "BM25",
        "Hybrid",
        "Hybrid+Rerank"
    ]

    for m in methods:
        run_method(m)

    print("\nAll retrieval methods completed.")