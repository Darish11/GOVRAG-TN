from pathlib import Path

# ---- PROJECT ROOT ----
PROJECT_ROOT = Path(r"E:\SRM\Project")

# ---- DATA ROOT ----
RAG_DATA = PROJECT_ROOT / "rag_data"
MASTER_GO_FOLDER = r"E:\SRM\Project\Dataset"
OCR_DIR = RAG_DATA / "ocr"
PREPROCESS_DIR = RAG_DATA / "preprocessing"
CHUNK_DIR = RAG_DATA / "chunking"
EMBEDDING_DIR = RAG_DATA / "embeddings"

LOG_DIR = RAG_DATA / "logs"
OCR_OUTPUT_DIR = r"E:\SRM\Project\rag_data\ocr"
PREPROCESS_DIR = r"E:\SRM\Project\rag_data\preprocessing"
CHUNK_DIR = r"E:\SRM\Project\rag_data\chunking"
VECTOR_DB_DIR = r"E:\SRM\Project\rag_data\vectorstore"
LOG_DIR = r"E:\SRM\Project\rag_data\logs"
METADATA_DIR = r"E:\SRM\Project\rag_data\metadata"
CHUNK_DIR = r"E:\SRM\Project\rag_data\chunks"

VECTORSTORE_DIR = r"E:\SRM\Project\rag_data\vectorstore\faiss_index"

FAISS_INDEX_PATH = f"{VECTORSTORE_DIR}\\faiss.index"
METADATA_PATH = f"{VECTORSTORE_DIR}\\chunk_metadata.json"

OLLAMA_MODEL = "qwen2.5:7b"
TOP_K = 40

# ---- OCR CONFIG ----
TESSERACT_LANG = "eng+tam"
DPI = 300

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\Library\bin"