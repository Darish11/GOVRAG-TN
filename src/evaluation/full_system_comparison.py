import json
import re
import pandas as pd
from rapidfuzz import fuzz

# ============================
# 1️⃣ RETRIEVAL METRICS (ALREADY COMPUTED)
# ============================

retrieval_results = {
    "Dense": {
        "Recall@1": 0.24,
        "Recall@5": 0.30,
        "Recall@10": 0.30,
        "Recall@20": 0.34,
        "MRR": 0.2666
    },
    "Dense + Reranker": {
        "Recall@1": 0.24,
        "Recall@5": 0.32,
        "Recall@10": 0.32,
        "Recall@20": 0.34,
        "MRR": 0.2675
    },
    "BM25": {
        "Recall@1": 0.18,
        "Recall@5": 0.24,
        "Recall@10": 0.26,
        "Recall@20": 0.30,
        "MRR": 0.2055
    },
    "Hybrid (RRF)": {
        "Recall@1": 0.24,
        "Recall@5": 0.30,
        "Recall@10": 0.32,
        "Recall@20": 0.32,
        "MRR": 0.2606
    }
}

# ============================
# 2️⃣ GENERATION FILE PATHS
# ============================

generation_files = {
    "Dense": r"E:\SRM\Project\evaluation\generated_dense.json",
    "Dense + Reranker": r"E:\SRM\Project\evaluation\generated_rerank.json",
    "BM25": r"E:\SRM\Project\evaluation\generated_bm25.json",
    "Hybrid (RRF)": r"E:\SRM\Project\evaluation\generated_hybrid.json"
}

# ============================
# 3️⃣ GENERATION METRIC FUNCTION
# ============================

def normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()

def compute_generation_metrics(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    answer_correct = 0
    clause_correct = 0
    go_mentioned = 0
    hallucinations = 0
    boolean_correct = 0
    boolean_total = 0

    for item in data:

        expected = normalize(item.get("expected_answer", ""))
        generated = normalize(item.get("generated_answer", ""))
        expected_go = normalize(item.get("go_number", ""))
        expected_clauses = [normalize(c) for c in item.get("expected_clauses", [])]

        # Fuzzy Answer Accuracy
        similarity = fuzz.partial_ratio(expected, generated)
        if similarity >= 70:
            answer_correct += 1

        # Clause Citation
        if expected_clauses:
            if any(clause in generated for clause in expected_clauses):
                clause_correct += 1

        # GO Mention
        if expected_go and expected_go in generated:
            go_mentioned += 1

        # Boolean
        if item.get("answer_type") == "boolean":
            boolean_total += 1
            if ("yes" in generated and "yes" in expected) or \
               ("no" in generated and "no" in expected):
                boolean_correct += 1

        # Hallucination
        if "i do not have information" not in generated:
            if expected_go and expected_go not in generated:
                hallucinations += 1

    return {
        "Answer Accuracy": round(answer_correct / total, 4),
        "Clause Citation Accuracy": round(clause_correct / total, 4),
        "GO Mention Accuracy": round(go_mentioned / total, 4),
        "Hallucination Rate": round(hallucinations / total, 4),
        "Boolean Accuracy": round(boolean_correct / boolean_total, 4) if boolean_total > 0 else None
    }

# ============================
# 4️⃣ COMBINE EVERYTHING
# ============================

final_rows = []

for method in retrieval_results.keys():

    retrieval_metrics = retrieval_results[method]

    generation_metrics = compute_generation_metrics(
        generation_files[method]
    )

    combined = {
        "Method": method,
        **retrieval_metrics,
        **generation_metrics
    }

    final_rows.append(combined)

final_df = pd.DataFrame(final_rows)

# ============================
# 5️⃣ OUTPUT TABLE
# ============================

print("\n==== FULL RAG SYSTEM COMPARISON ====\n")
print(final_df)

# Optional export for paper
final_df.to_csv(r"E:\SRM\Project\evaluation\full_system_comparison.csv", index=False)

print("\nSaved CSV to evaluation folder.")