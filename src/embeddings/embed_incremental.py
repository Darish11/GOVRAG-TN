import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNK_DIR = r"E:\SRM\Project\rag_data\chunks"
EMBED_DIR = r"E:\SRM\Project\rag_data\embeddings"

os.makedirs(EMBED_DIR, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64

model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):
    return model.encode(
        [f"passage: {t}" for t in texts],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False
    )


def process_file(chunk_file):

    in_path = os.path.join(CHUNK_DIR, chunk_file)
    out_path = os.path.join(EMBED_DIR, chunk_file.replace("_chunks.json", ".npz"))

    # ---- RESUME SUPPORT ----
    if os.path.exists(out_path):
        return "skipped"

    with open(in_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        emb = embed_texts(batch)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    np.savez_compressed(out_path, embeddings=embeddings)

    return "done"


def run():

    files = [f for f in os.listdir(CHUNK_DIR) if f.endswith("_chunks.json")]

    done = 0
    skipped = 0

    for f in tqdm(files):

        result = process_file(f)

        if result == "done":
            done += 1
        else:
            skipped += 1

    print("\nEmbedding Summary")
    print("New files embedded :", done)
    print("Skipped existing   :", skipped)


if __name__ == "__main__":
    run()
