\# AdaptiveRAG – Feedback-Driven Research Paper Assistant



AdaptiveRAG is a Retrieval-Augmented Generation (RAG) research assistant designed to answer technical questions using a fixed collection of five research papers.



Unlike conventional RAG systems, AdaptiveRAG explores how historical user feedback can influence future retrieval decisions.



\## Problem Statement



Traditional RAG systems retrieve documents independently for every query. They do not inherently learn from whether previous retrieval results were useful.



AdaptiveRAG introduces a feedback-driven retrieval mechanism that uses historical feedback to improve future retrieval decisions.



\## Key Features



\- Research paper PDF ingestion

\- Structure-aware document chunking

\- Local sentence-transformer embeddings

\- Qdrant Cloud vector storage

\- Grounded answers with citations

\- Conversation memory

\- Feedback collection

\- Adaptive retrieval ranking

\- Corrective retrieval using LangGraph

\- Baseline vs Adaptive RAG evaluation



\## Technology Stack



| Component | Technology |

|---|---|

| Backend | Python, FastAPI |

| RAG | LangChain, LangGraph |

| Embeddings | Sentence Transformers, BGE |

| Vector Database | Qdrant Cloud |

| LLM | Groq |

| Database | SQLite |

| Frontend | Streamlit |

| PDF Processing | PyMuPDF, PyMuPDF4LLM |



\## Knowledge Base



The initial knowledge base consists of five genuine research papers:



1\. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

2\. Dense Passage Retrieval for Open-Domain Question Answering

3\. ColBERT: Efficient and Effective Passage Search

4\. Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)

5\. RAGAS: Automated Evaluation of Retrieval Augmented Generation



\## Main Research Objective



Can user feedback about previous retrieval results improve future retrieval decisions in a RAG system?



\## Project Status



Currently under development.



Features will be documented as implemented and tested.



\## License



MIT License



