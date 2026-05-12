import json
import os
import re
from rapidfuzz import fuzz
from collections import Counter

# ================= PATHS =================

EVAL_DIR = r"E:\SRM\Project\evaluation"
GOLD_QA_PATH = r"E:\SRM\Project\evaluation\gold_qa\go_gold_qa.json"

generation_files = {
    "Dense": os.path.join(EVAL_DIR, "generated_Dense.json"),
    "Dense+Rerank": os.path.join(EVAL_DIR, "generated_Dense+Rerank.json"),
    "BM25": os.path.join(EVAL_DIR, "generated_BM25.json"),
    "Hybrid": os.path.join(EVAL_DIR, "generated_Hybrid.json"),
    "Hybrid+Rerank": os.path.join(EVAL_DIR, "generated_Hybrid+Rerank.json")
}

FUZZY_THRESHOLD = 70

# ================= HELPERS =================

def normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())

def token_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)

# ================= EVALUATION =================

def evaluate_method(method_name, file_path):

    if not os.path.exists(file_path):
        print(f"Skipping {method_name} (file not found)")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    answer_correct = 0
    clause_correct = 0
    go_correct = 0
    hallucinations = 0
    boolean_correct = 0
    exact_match = 0
    f1_scores = []
    similarity_scores = []

    for item in data:

        expected = normalize(item.get("expected_answer", ""))
        generated = normalize(item.get("generated_answer", ""))

        expected_clauses = item.get("expected_clauses", [])
        go_number = normalize(item.get("go_number", ""))
        answer_type = item.get("answer_type", "text")

        # ---- Similarity ----
        similarity = fuzz.partial_ratio(expected, generated)
        similarity_scores.append(similarity)

        if similarity >= FUZZY_THRESHOLD:
            answer_correct += 1

        if expected == generated:
            exact_match += 1

        f1_scores.append(token_f1(generated, expected))

        # ---- Clause Check ----
        if any(
            clause.lower() in generated
            for clause in expected_clauses
        ):
            clause_correct += 1

        # ---- GO Mention ----
        if go_number in generated:
            go_correct += 1

        # ---- Boolean Accuracy ----
        if answer_type == "boolean":
            if ("yes" in generated and "yes" in expected) or \
               ("no" in generated and "no" in expected):
                boolean_correct += 1

        # ---- Hallucination Proxy ----
        if go_number not in generated and similarity < 40:
            hallucinations += 1

    metrics = {
        "Total Questions": total,
        "Answer Accuracy": round(answer_correct / total, 3),
        "Exact Match": round(exact_match / total, 3),
        "Avg F1 Score": round(sum(f1_scores) / total, 3),
        "Avg Similarity": round(sum(similarity_scores) / total, 2),
        "Clause Citation Accuracy": round(clause_correct / total, 3),
        "GO Mention Accuracy": round(go_correct / total, 3),
        "Boolean Accuracy": round(boolean_correct / total, 3),
        "Hallucination Rate": round(hallucinations / total, 3)
    }

    return metrics

# ================= MAIN =================

def main():

    print("\n==== GENERATION EVALUATION (ALL METHODS) ====\n")

    all_results = {}

    for method, path in generation_files.items():
        metrics = evaluate_method(method, path)
        if metrics:
            all_results[method] = metrics

    # Print nicely
    for method, metrics in all_results.items():
        print(f"\n--- {method} ---")
        for k, v in metrics.items():
            print(f"{k:30}: {v}")

    # Save summary JSON
    summary_path = os.path.join(EVAL_DIR, "generation_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print("\nSaved summary to generation_metrics_summary.json")

if __name__ == "__main__":
    main()