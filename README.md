# GOVRAG-TN

### A Domain-Adaptive Cross-Lingual RAG Framework for Trustworthy Interpretation of Tamil Nadu Government Orders

---

## Overview

Government Orders (GOs) form the administrative backbone of policy implementation, financial approvals, amendments, and institutional governance in Tamil Nadu. However, these documents are typically distributed as scanned bilingual PDFs with complex legal structure, making retrieval and interpretation difficult using conventional search systems.

GOVRAG-TN is a domain-adaptive cross-lingual Retrieval-Augmented Generation (RAG) framework designed to provide trustworthy interpretation of Tamil Nadu Government Orders. The framework combines OCR-based document processing, clause-aware chunking, multilingual semantic retrieval, hybrid search, temporal policy prioritization, and citation-grounded generation within a fully offline architecture.

The system supports:
- semantic retrieval across bilingual Government Orders
- clause-level evidence retrieval
- multi-document reasoning
- grounded response generation with citations

---

# Key Features

- OCR-based processing of scanned Government Order PDFs
- Clause-aware legal document chunking
- Cross-lingual semantic embeddings
- Hybrid retrieval using Dense Retrieval + BM25 + RRF
- Multi-GO reasoning and evidence aggregation
- Temporal policy prioritization for recent amendments
- Citation-grounded response generation
- Fully offline deployment using open-source models
- Streamlit-based chatbot interface

---

# System Architecture

Place architecture image inside:

architecture/govrag_architecture.png

Then GitHub will automatically display it below.

![GOVRAG Architecture](architecture/govrag_architecture.png)

---

# Project Structure

```text
GOVRAG-TN
│
├── architecture/
├── dataset/
├── docs/
├── examples/
├── notebooks/
├── outputs/
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── metadata/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── ui/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore