import os
import json
from config import METADATA_DIR, CHUNK_DIR

CHUNK_DIR = r"E:\SRM\Project\rag_data\chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)


# ================= CLAUSE SPLIT =================

def extract_clause_chunks(pages, metadata):

    text = "\n".join(p["text"] for p in pages)

    import re
    parts = re.split(r"\n\s*(\d+\.)\s+", text)

    chunks = []

    for i in range(1, len(parts), 2):

        clause_no = parts[i].strip(".")
        clause_text = parts[i + 1].strip()

        chunks.append({
            "text": clause_text,
            "metadata": {
                **metadata,          # ⭐ CRITICAL FIX
                "chunk_type": "clause",
                "section_id": clause_no
            }
        })

    return chunks


# ================= MAIN =================

def run_chunking():

    files = [f for f in os.listdir(METADATA_DIR)
             if f.endswith("_meta.json")]

    print(f"Chunking {len(files)} documents")

    for file in files:

        with open(
            os.path.join(METADATA_DIR, file),
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        metadata = data["metadata"]
        pages = data["pages"]

        chunks = extract_clause_chunks(pages, metadata)

        out_path = os.path.join(
            CHUNK_DIR,
            file.replace("_meta.json", "_chunks.json")
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("✅ Chunking completed")


if __name__ == "__main__":
    run_chunking()