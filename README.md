# 🏦 Enterprise Finance, Dynamic Invoice & Analytics RAG System

An end-to-end, enterprise-grade Retrieval-Augmented Generation (RAG) system designed for automated financial document intelligence, dynamic corporate PDF invoice generation, and interactive data visualization from structured and unstructured financial records.

⚡ **Developed by:** Zohaib Ahmed Khan

---

## 🏗️ Architecture & System Overview

The system processes unstructured finance documents (PDF, DOCX) and structured records (CSV, XLSX) through an integrated vector database and hybrid retrieval pipeline. It exposes a natural language conversational interface capable of multi-step retrieval, automated invoice generation without hardcoded fields, and dynamic Plotly chart generation based on user queries.

+-----------------------------------------------------------------------------------+
|                                🎨 Streamlit UI                                    |
|                 (Glassmorphic Dark Theme + Dynamic Graph Engine)                  |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
|                               🔍 Hybrid Search                                    |
|              Dense Vector (ChromaDB) + Sparse Keyword Search (BM25)               |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
|                              🤖 Groq LLM Engine                                   |
|                 (Context Synthesis + Dynamic JSON Line Item Extraction)           |
+-----------------------------------------------------------------------------------+
|                                             |
v                                             v
+-------------------------------------+       +-------------------------------------+
|      📄 FPDF Invoice Generator      |       |       📊 Plotly Graph Engine        |
|    (Client-Specific PDF Export)     |       |  (Bar, Line, Scatter, Pie, Box Plots)|
+-------------------------------------+       +-------------------------------------+

---

## ✨ Key Features

* 📄 **Multi-Format Ingestion Pipeline**
  * Supports ingestion and structural parsing of PDF, DOCX, CSV, and XLSX files.
  * Text chunks are tagged with file metadata, document type, and section numbers.

* 🔍 **Hybrid Retrieval-Augmented Generation (RRF)**
  * Combines semantic similarity (HuggingFace Embeddings + ChromaDB) with lexical keyword matching (BM25).
  * Uses Reciprocal Rank Fusion (RRF) to score and prioritize relevant context chunks.
  * Includes transparent source attribution displaying exact chunks and page locations used for every answer.

* 🧾 **Dynamic Corporate Invoice Engine**
  * Extracts line items, quantities, unit rates, tax rates, billing/shipping addresses, and payment terms directly from uploaded context via structured JSON parsing.
  * Generates production-ready PDF invoices using `FPDF` with client-specific file naming and layout formatting.
  * No hardcoded fields; strictly reliant on retrieved document context.

* 📊 **Universal Data Analytics & Graph Engine**
  * Automatically detects numeric and categorical columns from CSV and Excel files.
  * Generates interactive Plotly charts (Bar Charts, Line Charts, Scatter Plots, Pie Charts, Box Plots) based on natural language intent.
  * Filters records by entity or metric dynamically.

* 🛡️ **Enterprise-Grade UI & Security Control**
  * Futuristic Glassmorphism Dark Theme with animated mesh gradients and interactive cards.
  * Fine-grained temperature slider for controlling LLM precision versus creativity.
  * Complete knowledge base reset utility and financial risk acknowledgment disclaimer.

---

## 🛠️ Tech Stack

* 🖥️ **Frontend & Dashboard:** Streamlit
* 🔗 **LLM Orchestration:** LangChain
* 🧠 **Language Model:** Groq API (`openai/gpt-oss-20b`)
* 🔤 **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* 🗄️ **Vector Database:** ChromaDB
* 🔎 **Lexical Search:** BM25Okapi (`rank_bm25`)
* 📁 **Document Loaders:** PyPDF, Docx2txt, CSVLoader, UnstructuredExcelLoader
* 📑 **PDF Generation:** FPDF2
* 📈 **Data Analytics & Visualization:** Pandas, Plotly Express

---

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.11 or higher
* Git

### 2. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/Enterprise-Finance-RAG-System.git
cd Enterprise-Finance-RAG-System
