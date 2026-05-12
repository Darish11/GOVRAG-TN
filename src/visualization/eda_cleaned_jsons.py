import os
import json
import pandas as pd
import re
from tqdm import tqdm
from config import PREPROCESS_DIR

EDA_OUTPUT = r"E:\SRM\Project\rag_data\eda"
os.makedirs(EDA_OUTPUT, exist_ok=True)


def tamil_char_ratio(text):
    if not text:
        return 0
    tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    return tamil_chars / max(len(text), 1)


def clause_count(text):
    return len(re.findall(r'\n\s*\d+\.\s+', text))


def run_eda():
    records = []

    files = [f for f in os.listdir(PREPROCESS_DIR) if f.endswith("_clean.json")]
    print(f"Running EDA on {len(files)} files")

    for file in tqdm(files):
        path = os.path.join(PREPROCESS_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pages = data.get("pages", [])
        total_text = "\n".join(p["text"] for p in pages)

        records.append({
            "file": file,
            "num_pages": len(pages),
            "total_chars": len(total_text),
            "avg_chars_per_page": (
                sum(len(p["text"]) for p in pages) / max(len(pages), 1)
            ),
            "tamil_ratio": tamil_char_ratio(total_text),
            "clause_count": clause_count(total_text)
        })

    df = pd.DataFrame(records)

    df.to_csv(os.path.join(EDA_OUTPUT, "go_corpus_stats.csv"), index=False)
    print("EDA CSV saved → go_corpus_stats.csv")

    print("\n=== SUMMARY ===")
    print(df.describe(percentiles=[0.5, 0.75, 0.9, 0.95]))


if __name__ == "__main__":
    run_eda()
