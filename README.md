# 🎓 UniBristol RAG Assistant

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)
![AWS](https://img.shields.io/badge/Deployment-AWS-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An enterprise-grade **AI Academic Advisor** for the University of Bristol.

This application uses **Retrieval Augmented Generation (RAG)** to provide accurate, hallucination-free answers to student queries about courses, fees, accommodation, and campus life. It is containerized with Docker and designed for scalable cloud deployment.

---

## 🏗️ System Architecture

```mermaid
graph LR
    A[User Query] --> B(Streamlit UI);
    B --> C{Hybrid Retrieval};
    C -->|Semantic Search| D[FAISS Vector DB];
    C -->|Keyword Filter| D;
    D --> E[Top 10 Docs];
    E --> F[FlashRank Reranker];
    F -->|Top 5 Relevant| G[LLM Context Window];
    G --> H[GPT-3.5 Turbo];
    H --> I[Final Answer];