import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX_PATH, METADATA_PATH

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer("intfloat/multilingual-e5-large")


def embed_query(query):
    return model.encode(
        f"query: {query}",
        normalize_embeddings=True
    )


def dense_search(query, top_k=40):

    q_emb = embed_query(query).reshape(1, -1)

    scores, indices = index.search(q_emb, top_k)

    results = []

    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return results
