import json
import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from collections import defaultdict
from math import log2

# ================= PATHS =================

BASE_PATH = r"E:\SRM\Project"
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "rag_data/vectorstore/faiss_index/faiss.index")
CHUNKS_PATH = os.path.join(BASE_PATH, "rag_data/vectorstore/faiss_index/chunk_metadata.json")
GOLD_QA_PATH = os.path.join(BASE_PATH, "evaluation/gold_qa/go_gold_qa.json")
OUTPUT_PATH = os.path.join(BASE_PATH, "evaluation/retrieval_metrics_summary.json")

TOP_K = 20
TOP_K_VALUES = [1, 5, 10, 20]
RRF_K = 60

# ================= NORMALIZATION =================

def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[().]", "", text)
    return text

# ================= LOAD =================

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print("Loading Gold QA...")
with open(GOLD_QA_PATH, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

print("Loading dense model...")
dense_model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Building BM25 corpus...")
tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# ================= RETRIEVAL FUNCTIONS =================

def dense_retrieve(query):
    emb = dense_model.encode(f"query: {query}", normalize_embeddings=True)
    D, I = index.search(np.array([emb]), TOP_K)
    return I[0]

def bm25_retrieve(query):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    return np.argsort(scores)[::-1][:TOP_K]

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
    return [idx for idx, _ in ranked]

# ================= RELEVANCE CHECK =================

def is_go_match(chunk_go, expected_go):
    return normalize(chunk_go) == normalize(expected_go)

def is_clause_match(chunk_section, expected_clauses):
    chunk_section_norm = normalize(str(chunk_section))
    return any(normalize(c) in chunk_section_norm for c in expected_clauses)

def is_relevant(chunk, expected_go, expected_clauses):
    metadata = chunk.get("metadata", {})
    chunk_go = metadata.get("go_number", "")
    chunk_section = metadata.get("section_id", "")

    go_match = is_go_match(chunk_go, expected_go)
    clause_match = is_clause_match(chunk_section, expected_clauses)

    return go_match or clause_match, go_match, clause_match

# ================= METRIC HELPERS =================

def compute_ndcg(relevances):
    dcg = sum(rel / log2(idx + 2) for idx, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / log2(idx + 2) for idx, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0

# ================= EVALUATION =================

def evaluate_method(method_name):

    metrics = defaultdict(float)
    total = len(gold_data)

    for item in gold_data:

        question = item["question"]
        expected_go = item["go_number"]
        expected_clauses = item.get("expected_clauses", [])

        dense_ids = dense_retrieve(question)
        bm25_ids = bm25_retrieve(question)

        if method_name == "Dense":
            final_ids = dense_ids

        elif method_name == "Dense+Rerank":
            final_ids = rerank(question, dense_ids)

        elif method_name == "BM25":
            final_ids = bm25_ids

        elif method_name == "Hybrid":
            final_ids = rrf_fusion(dense_ids, bm25_ids)

        elif method_name == "Hybrid+Rerank":
            hybrid_ids = rrf_fusion(dense_ids, bm25_ids)
            final_ids = rerank(question, hybrid_ids)

        else:
            raise ValueError("Unknown method")

        relevances = []
        go_hits = 0
        clause_hits = 0

        for idx in final_ids:
            chunk = all_chunks[idx]
            relevant, go_match, clause_match = is_relevant(chunk, expected_go, expected_clauses)
            relevances.append(1 if relevant else 0)

            if go_match:
                go_hits = 1
            if clause_match:
                clause_hits = 1

        # Recall & Precision
        for k in TOP_K_VALUES:
            top_k = relevances[:k]
            metrics[f"Recall@{k}"] += int(any(top_k))
            metrics[f"Precision@{k}"] += sum(top_k) / k
            metrics[f"HitRate@{k}"] += int(any(top_k))

        # MRR
        for rank, rel in enumerate(relevances):
            if rel == 1:
                metrics["MRR"] += 1 / (rank + 1)
                break

        # MAP
        hits = 0
        avg_prec = 0
        for i, rel in enumerate(relevances):
            if rel == 1:
                hits += 1
                avg_prec += hits / (i + 1)
        metrics["MAP"] += avg_prec / hits if hits > 0 else 0

        # nDCG
        metrics["nDCG@10"] += compute_ndcg(relevances[:10])

        # GO & Clause Recall
        metrics["GO_Recall"] += go_hits
        metrics["Clause_Recall"] += clause_hits

    # Normalize
    for key in metrics:
        metrics[key] = round(metrics[key] / total, 3)

    metrics["Total Questions"] = total
    return dict(metrics)

# ================= MAIN =================

def main():

    methods = [
        "Dense",
        "Dense+Rerank",
        "BM25",
        "Hybrid",
        "Hybrid+Rerank"
    ]

    results = {}

    print("\n==== RETRIEVAL EVALUATION (ALL METHODS) ====\n")

    for method in methods:
        print(f"Evaluating {method}...")
        results[method] = evaluate_method(method)

    for method, metrics in results.items():
        print(f"\n--- {method} ---")
        for k, v in metrics.items():
            print(f"{k:20}: {v}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nSaved retrieval_metrics_summary.json")

if __name__ == "__main__":
    main()