import faiss
import json

index = faiss.read_index(
r"E:\SRM\Project\rag_data\vectorstore\faiss_index\faiss.index"
)

chunks = json.load(open(
r"E:\SRM\Project\rag_data\vectorstore\faiss_index\chunk_metadata.json",
encoding="utf-8"
))

print("FAISS vectors:", index.ntotal)
print("Metadata chunks:", len(chunks))