import json
import re
from rapidfuzz import fuzz

INPUT_PATH = r"E:\SRM\Project\evaluation\generated_answers.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()

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

    # --- Answer Accuracy (Fuzzy) ---
    similarity = fuzz.partial_ratio(expected, generated)
    if similarity >= 70:
        answer_correct += 1

    # --- Clause Citation Accuracy ---
    if expected_clauses:
        if any(clause in generated for clause in expected_clauses):
            clause_correct += 1

    # --- GO Mention Accuracy ---
    if expected_go and expected_go in generated:
        go_mentioned += 1

    # --- Boolean Accuracy ---
    if item.get("answer_type") == "boolean":
        boolean_total += 1
        if ("yes" in generated and "yes" in expected) or \
           ("no" in generated and "no" in expected):
            boolean_correct += 1

    # --- Hallucination Detection ---
    if "i do not have information" not in generated:
        if expected_go and expected_go not in generated:
            hallucinations += 1

print("\n==== GENERATION METRICS ====\n")
print("Total Questions         :", total)
print("Answer Accuracy         :", round(answer_correct / total, 4))
print("Clause Citation Accuracy:", round(clause_correct / total, 4))
print("GO Mention Accuracy     :", round(go_mentioned / total, 4))
print("Hallucination Rate      :", round(hallucinations / total, 4))

if boolean_total > 0:
    print("Boolean Accuracy        :", round(boolean_correct / boolean_total, 4))