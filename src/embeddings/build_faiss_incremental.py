import os
import json
import numpy as np
import faiss
from tqdm import tqdm


# =========================
# PATHS
# =========================

EMBEDDINGS_DIR = r"E:\SRM\Project\rag_data\embeddings"
CHUNKS_DIR = r"E:\SRM\Project\rag_data\chunks"

FAISS_DIR = r"E:\SRM\Project\rag_data\vectorstore\faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss.index")
METADATA_PATH = os.path.join(FAISS_DIR, "chunk_metadata.json")


# =========================
# BUILD FAISS SAFELY
# =========================

def build_faiss():

    embedding_files = sorted([
        f for f in os.listdir(EMBEDDINGS_DIR)
        if f.endswith(".npz")
    ])

    all_vectors = []
    all_metadata = []

    print(f"\nProcessing {len(embedding_files)} embedding files...\n")

    for emb_file in tqdm(embedding_files):

        emb_path = os.path.join(EMBEDDINGS_DIR, emb_file)

        # corresponding chunk file
        chunk_file = emb_file.replace(".npz", "_chunks.json")
        chunk_path = os.path.join(CHUNKS_DIR, chunk_file)

        if not os.path.exists(chunk_path):
            continue

        # -------- embeddings ----------
        data = np.load(emb_path, allow_pickle=True)
        vectors = data["embeddings"]

        # -------- metadata ----------
        with open(chunk_path, encoding="utf-8") as f:
            chunks = json.load(f)

        # ALIGN COUNT SAFELY
        min_len = min(len(vectors), len(chunks))

        for i in range(min_len):
            all_vectors.append(vectors[i])
            all_metadata.append(chunks[i])

    embeddings = np.array(all_vectors).astype("float32")

    print("\nFinal alignment check")
    print("Vectors :", len(embeddings))
    print("Metadata:", len(all_metadata))

    assert len(embeddings) == len(all_metadata)

    # =========================
    # BUILD FAISS
    # =========================

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"\nFAISS size : {index.ntotal}")

    # =========================
    # SAVE
    # =========================

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False)

    print("\n✅ FAISS BUILD SUCCESSFUL")


if __name__ == "__main__":
    build_faiss()