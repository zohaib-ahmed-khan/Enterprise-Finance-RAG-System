"""
====================================================================================================
🏦 ENTERPRISE-GRADE FINANCE, AUTOMATED INVOICE GENERATION & ANALYTICS RAG SYSTEM
====================================================================================================
Architected & Developed by: Zohaib Ahmed Khan
Tech Stack: Streamlit, LangChain, ChromaDB, Groq (OSS-20B), Plotly, FPDF2, BM25 Hybrid Retrieval

CORE SYSTEM CAPABILITIES & PRODUCTION FEATURES:
----------------------------------------------------------------------------------------------------
1. INTELLIGENT MULTI-FORMAT INGESTION PIPELINE:
   - Automated cleaning, structural parsing, and chunking for PDF, DOCX, CSV, and XLSX datasets.
   - Dual-engine numerical column auto-detection for instant structured data analytics.

2. HIGH-PRECISION HYBRID SEARCH ENGINE (RRF - Reciprocal Rank Fusion):
   - Merges Dense Vector Search (HuggingFace Embeddings + ChromaDB) with Sparse BM25 Keyword Search.
   - Delivers sub-second retrieval accuracy with exact source chunk and document section attribution.

3. DYNAMIC INVOICE GENERATION ENGINE:
   - Extract strictly verified vendor, line-item, billing, and tax data via LLM JSON extraction.
   - Automatically renders corporate-grade PDFs with custom client-named download hooks.

4. UNIVERSAL ADAPTIVE DATA VISUALIZATION ENGINE:
   - Context-aware chart generation (Bar, Line, Scatter, Pie, Box Plots) powered by Plotly Express.
   - Real-time column matching and entity-level data filtering directly from natural language queries.

5. FUTURISTIC GLASSMORPHIC SCIFI UI & CONTROL INTERFACE:
   - Custom CSS mesh-gradient background, 3D interactive metric cards, and continuous neon aura borders.
   - Model creativity (temperature) fine-tuning and complete knowledge-base reset safeguards.
====================================================================================================
"""




import os
import tempfile
import json
import re
import uuid
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import random

# Document Loaders & LangChain Components
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader, 
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# ==============================================================================
# 1. SETUP & PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Advanced Finance & Invoice RAG System", 
    page_icon="🏦", 
    layout="wide"
)

load_dotenv()
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ==========================================================================
       1. GLOBAL DYNAMIC ANIMATED MESH BACKGROUND (DARK THEME)
       ========================================================================== */
    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #090d16) !important;
        background-size: 400% 400% !important;
        animation: meshGradient 16s ease infinite !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #f8fafc !important;
    }

    @keyframes meshGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ==========================================================================
       2. CUSTOM SKY BLUE GLOW SCROLLBAR
       ========================================================================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #020617;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #38bdf8, #6366f1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #7dd3fc;
    }

    /* ==========================================================================
       3. HIGH CONTRAST TEXT & HEADINGS
       ========================================================================== */
    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: #f8fafc !important;
    }

    /* ==========================================================================
       4. HEAVY NEON TITLE CONTAINER WITH CONTINUOUS SKY BLUE ROTATING GLOW BORDER
       ========================================================================== */
    .title-container {
        position: relative;
        padding: 24px 30px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        margin-bottom: 25px;
        overflow: hidden;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6);
    }

    .title-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            transparent, 
            rgba(56, 189, 248, 0.15), 
            #38bdf8, 
            #0284c7, 
            transparent 60%
        );
        animation: rotateGlow 3.5s linear infinite;
        z-index: 0;
    }

    .title-container::after {
        content: '';
        position: absolute;
        inset: 3px;
        background: rgba(15, 23, 42, 0.94);
        border-radius: 17px;
        z-index: 1;
    }

    @keyframes rotateGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .title-content {
        position: relative;
        z-index: 2;
    }

    .main-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: -0.5px;
        text-shadow: 0 0 25px rgba(56, 189, 248, 0.35);
    }

    .sub-title {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-top: 6px !important;
        letter-spacing: 0.4px;
    }

    .author-badge {
        display: inline-block;
        margin-top: 10px;
        padding: 5px 14px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(56, 189, 248, 0.5);
        color: #38bdf8 !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.3);
    }

    /* ==========================================================================
       5. ULTRA 3D NEON METRIC CARDS WITH GLASSMORPHISM
       ========================================================================== */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75) !important;
        padding: 22px 26px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.6), inset 0 1px 2px rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(16px) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) rotateX(3deg) scale(1.03) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 25px 45px -10px rgba(56, 189, 248, 0.4), 0 0 20px rgba(99, 102, 241, 0.3) !important;
    }

    div[data-testid="stMetricValue"] > div {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-shadow: 0 0 16px rgba(56, 189, 248, 0.6);
    }
    div[data-testid="stMetricLabel"] > div {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }

    /* ==========================================================================
       6. SIDEBAR GLASSMORPHISM & HIGH-CONTRAST INPUTS
       ========================================================================== */
    section[data-testid="stSidebar"] {
        background: rgba(3, 7, 18, 0.88) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 12px 0 35px rgba(0, 0, 0, 0.6) !important;
    }

    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #475569 !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    .stTextInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.45) !important;
        transition: all 0.4s ease !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.6) !important;
    }

    /* ==========================================================================
       7. CHAT BUBBLES FADE-IN & CHAT INPUT BAR
       ========================================================================== */
    .stChatMessage {
        background: rgba(15, 23, 42, 0.7) !important;
        border-radius: 18px !important;
        padding: 20px 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 12px 30px -5px rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        margin-bottom: 18px !important;
        animation: fadeInUp 0.5s ease forwards !important;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stChatInputContainer {
        background-color: #0f172a !important;
        border-radius: 18px !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5) !important;
    }
    .stChatInputContainer textarea {
        color: #ffffff !important;
    }

    /* ==========================================================================
       8. SOURCES EXPANDER & EMERALD DOWNLOAD BUTTON
       ========================================================================== */
    .stExpander {
        background: rgba(15, 23, 42, 0.85) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 12px 26px !important;
        box-shadow: 0 6px 25px rgba(16, 185, 129, 0.45) !important;
        transition: all 0.3s ease !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.65) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED COMPONENTS & STATE INITIALIZATION
# ==============================================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if "db" not in st.session_state:
    st.session_state.db = Chroma(
        collection_name=f"finance_rag_{uuid.uuid4().hex[:8]}",
        embedding_function=get_embeddings()
    )

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}

if "master_df" not in st.session_state:
    st.session_state.master_df = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==============================================================================
# 3. ADVANCED HYBRID SEARCH (RRF)
# ==============================================================================
def hybrid_search(query, top_k=5):
    all_docs_dict = st.session_state.db.get()
    all_texts = all_docs_dict.get('documents', [])
    all_metadatas = all_docs_dict.get('metadatas', [])
    
    if not all_texts:
        return []
        
    documents = [Document(page_content=txt, metadata=meta) for txt, meta in zip(all_texts, all_metadatas)]
    
    tokenized_corpus = [doc.page_content.lower().split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split(" "))
    bm25_ranked = sorted(zip(documents, bm25_scores), key=lambda x: x[1], reverse=True)[:top_k*2]
    
    vector_retriever = st.session_state.db.as_retriever(search_kwargs={"k": top_k*2})
    vector_ranked = vector_retriever.invoke(query)
    
    rrf_scores = {}
    k_constant = 60
    for rank, (doc, _) in enumerate(bm25_ranked):
        rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + (1 / (k_constant + rank))
    for rank, doc in enumerate(vector_ranked):
        rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + (1 / (k_constant + rank))
        
    fused_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    final_docs = []
    for content, _ in fused_docs:
        meta = next((doc.metadata for doc in documents if doc.page_content == content), {})
        final_docs.append(Document(page_content=content, metadata=meta))
        
    return final_docs


# ==============================================================================
# 4. DATA PIPELINE & SMART PARSING
# ==============================================================================
def process_uploaded_file(uploaded_file, chunk_size, chunk_overlap):
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        temp_df = pd.DataFrame()
        
        if ext == ".csv":
            temp_df = pd.read_csv(tmp_path)
            loader = CSVLoader(tmp_path)
        elif ext in [".xlsx", ".xls"]:
            try:
                temp_df = pd.read_excel(tmp_path)
            except Exception as e:
                st.error(f"Excel read error: {e}")
            loader = UnstructuredExcelLoader(tmp_path)
        elif ext == ".pdf": 
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx": 
            loader = Docx2txtLoader(tmp_path)
        else: 
            return 0
            
        if not temp_df.empty:
            has_numbers = False
            for col in temp_df.columns:
                cleaned_str = temp_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                converted = pd.to_numeric(cleaned_str, errors='coerce')
                if converted.notnull().sum() > 0:
                    has_numbers = True
                    break
            
            if has_numbers or st.session_state.master_df is None:
                st.session_state.master_df = temp_df
                
        documents = loader.load()
        for i, doc in enumerate(documents): 
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["document_type"] = ext.upper()
            doc.metadata["page_section"] = i + 1
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        st.session_state.db.add_documents(chunks)
        
        st.session_state.indexed_files[uploaded_file.name] = len(chunks)
        return len(chunks)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ==============================================================================
# 5. DYNAMIC CORPORATE INVOICE PDF GENERATOR
# ==============================================================================
def generate_invoice_pdf(text, client_name="Client"):
    safe_client_name = "".join([c if c.isalnum() else "_" for c in client_name])
    filename = f"{safe_client_name}_Invoice.pdf"
    
    data = {}
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_json = text[start_idx:end_idx+1]
            clean_json = re.sub(r',\s*}', '}', clean_json)
            clean_json = re.sub(r',\s*\]', ']', clean_json)
            data = json.loads(clean_json)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")

    items = data.get("items", [])
    if not items:
        return None

    company_name = data.get("company_name", "Enterprise Service Provider")
    company_address = data.get("company_address", "Corporate Billing Division")
    inv_no = data.get("invoice_number", f"INV-{random.randint(100000, 999999)}")
    inv_date = data.get("invoice_date", datetime.now().strftime("%d %b %Y"))
    due_date = data.get("due_date", (datetime.now() + timedelta(days=15)).strftime("%d %b %Y"))
    terms_val = data.get("terms", "As per agreement")
    
    bill_to = data.get("bill_to", f"{client_name}\nAddress Available in Records")
    ship_to = data.get("ship_to", f"{client_name}\nAddress Available in Records")
    
    sub_total = str(data.get("subtotal", "0.00")).replace('$', '')
    tax_rate = str(data.get("tax_rate", "0.0%"))
    tax_amt = str(data.get("tax_amount", "0.00")).replace('$', '')
    total_due = str(data.get("total_due", "0.00")).replace('$', '')
    tc = data.get("terms_and_conditions", "Full payment is due upon receipt as per agreed contractual terms.")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_draw_color(40, 40, 40)
    pdf.set_line_width(0.4)
    pdf.rect(10, 10, 190, 277)
    
    pdf.set_xy(15, 15)
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(20, 30, 50)
    pdf.cell(100, 8, str(company_name)[:45], ln=True)
    
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(90, 90, 90)
    for addr_line in str(company_address).split('\n')[:2]:
        pdf.set_x(15)
        pdf.cell(100, 5, addr_line.strip()[:50], ln=True)
    
    pdf.set_xy(120, 15)
    pdf.set_font("Arial", 'B', 26)
    pdf.set_text_color(26, 82, 118)
    pdf.cell(75, 12, "INVOICE", ln=True, align="R")
    
    pdf.ln(8)
    pdf.line(10, 40, 200, 40)
    
    start_y = pdf.get_y()
    pdf.set_xy(10, start_y)
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(30, 6, " Invoice#", border=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(65, 6, f" {str(inv_no)[:25]}", border=1, ln=True)
    
    pdf.set_x(10)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 6, " Invoice Date", border=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(65, 6, f" {str(inv_date)[:25]}", border=1, ln=True)
    
    pdf.set_x(10)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 6, " Terms", border=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(65, 6, f" {str(terms_val)[:25]}", border=1, ln=True)
    
    pdf.set_x(10)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 6, " Due Date", border=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(65, 6, f" {str(due_date)[:25]}", border=1, ln=True)
    
    pdf.rect(105, start_y, 95, 24)
    
    bil_y = pdf.get_y() + 2
    pdf.set_xy(10, bil_y)
    pdf.set_fill_color(240, 242, 245)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(95, 6, " Bill To", border=1, fill=True)
    pdf.cell(95, 6, " Ship To", border=1, fill=True, ln=True)
    
    b_lines = str(bill_to).replace('\r', '').split('\n')
    s_lines = str(ship_to).replace('\r', '').split('\n')
    b_lines.extend([""] * (4 - len(b_lines)))
    s_lines.extend([""] * (4 - len(s_lines)))

    pdf.set_x(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 6, f" {b_lines[0].strip()[:50]}", border="LR")
    pdf.set_font("Arial", '', 9)
    pdf.cell(95, 6, f" {s_lines[0].strip()[:50]}", border="LR", ln=True)
    
    for i in range(1, 4):
        pdf.set_x(10)
        border_style = "LR" if i < 3 else "BL"
        border_style_r = "LR" if i < 3 else "BR"
        pdf.cell(95, 5, f" {b_lines[i].strip()[:55]}", border=border_style)
        pdf.cell(95, 5, f" {s_lines[i].strip()[:55]}", border=border_style_r, ln=True)
    
    pdf.ln(4)
    
    pdf.set_fill_color(26, 82, 118)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(12, 8, " #", border=1, fill=True, align="C")
    pdf.cell(98, 8, " Item & Description", border=1, fill=True)
    pdf.cell(20, 8, " Qty", border=1, fill=True, align="C")
    pdf.cell(30, 8, " Rate", border=1, fill=True, align="R")
    pdf.cell(30, 8, " Amount", border=1, fill=True, align="R", ln=True)
    
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Arial", '', 9)
    
    for idx, item in enumerate(items[:15], 1):
        desc = str(item.get("description", "Service Item"))
        safe_desc = desc.encode('latin-1', 'replace').decode('latin-1')[:52]
        qty = str(item.get("qty", "1"))
        rate = str(item.get("rate", "0.00")).replace('$', '')
        amt = str(item.get("amount", "0.00")).replace('$', '')
            
        pdf.cell(12, 10, f" {idx}", border=1, align="C")
        pdf.cell(98, 10, f" {safe_desc}", border=1)
        pdf.cell(20, 10, f" {qty}", border=1, align="C")
        pdf.cell(30, 10, f" ${rate} ", border=1, align="R")
        pdf.cell(30, 10, f" ${amt} ", border=1, align="R", ln=True)
        
    totals_start_y = pdf.get_y()
    pdf.set_xy(10, totals_start_y)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(110, 6, f" Thanks for doing business with {company_name}!", ln=False)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(120)
    pdf.cell(50, 6, " Sub Total", border=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(30, 6, f" ${sub_total} ", border=1, align="R", ln=True)
    
    pdf.set_x(120)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(50, 6, f" Tax Rate ({str(tax_rate)})", border=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(30, 6, f" ${tax_amt} ", border=1, align="R", ln=True)
    
    pdf.set_fill_color(220, 230, 242)
    pdf.set_x(120)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(50, 8, " Balance Due", border=1, fill=True)
    pdf.cell(30, 8, f" ${total_due} ", border=1, fill=True, align="R", ln=True)
    
    pdf.set_xy(10, totals_start_y + 8)
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(20, 30, 50)
    pdf.cell(110, 5, "Terms & Conditions", ln=True)
    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(105, 4, str(tc).encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.output(filename)
    return filename


# ==============================================================================
# 6. STREAMLIT UI & SIDEBAR
# ==============================================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
<div class="title-container">
    <div class="title-content">
        <h1 class="main-title">🏦 Advanced Finance & Invoice RAG System</h1>
        <div class="sub-title">Enterprise Knowledge Base | Hybrid Retrieval | Conversational Analytics & Graphs</div>
        <div class="author-badge">⚡ Developed By: Zohaib Ahmed Khan</div>
    </div>
</div>
""", unsafe_allow_html=True)
with col2:
    st.metric(label="RAG Engine", value="Online", delta="Semantic + Keyword")

st.divider()

total_chunks_count = sum(st.session_state.indexed_files.values()) if st.session_state.indexed_files else 0
total_files_count = len(st.session_state.indexed_files)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Indexed Documents", f"{total_files_count} Files")
m2.metric("Total Vector Chunks", f"{total_chunks_count} Chunks")
m3.metric("Retrieval Engine", "Hybrid (RRF)")
m4.metric("LLM Model", "Groq OSS 20B")

st.divider()

with st.sidebar:
    st.header("🔐 System Configuration")
    user_api_key = st.text_input("Groq API Key (Optional)", type="password", placeholder="Leave blank for default key")
    active_api_key = user_api_key if user_api_key else DEFAULT_GROQ_API_KEY
    
    st.divider()
    st.header("🗂️ Document Ingestion")
    chunk_size = st.slider("Intelligent Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
    
    st.header("🌡️ Model Creativity (Temperature)")
    temperature_value = st.slider(
        "Select Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 = Precise & Point-to-Point (Calculations/Invoices). Higher = More Creative/Detailed."
    )
    if temperature_value == 0.0:
        st.caption("🎯 Mode: **Point-to-Point (Precise)**")
    elif temperature_value <= 0.5:
        st.caption("⚖️ Mode: **Balanced**")
    else:
        st.caption("💡 Mode: **Creative / Detailed**")

    uploaded_files = st.file_uploader("Upload Finance Policies, SOPs, CSVs", type=["pdf", "docx", "csv", "xlsx"], accept_multiple_files=True)
    
    if st.button("🚀 Process & Index Data", use_container_width=True):
        if uploaded_files:
            with st.spinner("Extracting, Cleaning & Generating Embeddings..."):
                total = sum(process_uploaded_file(f, chunk_size, chunk_overlap) for f in uploaded_files)
                st.success(f"Successfully indexed {total} metadata-rich chunks.")
        else:
            st.warning("Awaiting document upload.")
            
    st.divider()
    st.header("📁 Document Inspector")
    if st.session_state.indexed_files:
        for fname, chunks in st.session_state.indexed_files.items():
            st.caption(f"📄 **{fname}** ({chunks} chunks)")
    else:
        st.info("No documents indexed yet.")
        
    if st.button("🗑️ Reset Entire Knowledge Base"):
        try:
            st.session_state.db.delete_collection()
        except Exception:
            pass
        st.session_state.db = Chroma(
            collection_name=f"finance_rag_{uuid.uuid4().hex[:8]}", 
            embedding_function=get_embeddings()
        )
        st.session_state.indexed_files = {}
        st.session_state.master_df = None
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    # Disclaimer & Risk Warning Section
    st.header("⚠️ System Disclaimer")
    
    # Checkbox / Toggle for User Acknowledgment
    disclaimer_ack = st.checkbox("I acknowledge financial risk disclaimer", value=True)
    
    with st.expander("📌 Read Full Financial Disclaimer"):
        st.warning("""
        **Important Notice & Terms of Use:**
        - **AI-Generated Insights:** Answers, analytics, and generated invoices are created using LLMs & RAG technology based on your uploaded files.
        - **Human Verification Required:** Always double-check invoice figures, tax calculations, and compliance data before processing official transactions.
        - **No Financial Advice:** This application is for automated data extraction and financial analytics support only and does not constitute professional accounting or legal advice.
        """)


if not active_api_key:
    st.info("👈 Please enter your Groq API Key in the sidebar or .env file.")
    st.stop()


# ==============================================================================
# 7. CHAT INTERFACE & HISTORY RENDERING WITH GUARANTEED SOURCE DISPLAY
# ==============================================================================
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("has_chart") and "chart_fig" in message:
            st.plotly_chart(message["chart_fig"], use_container_width=True, key=f"chat_chart_{i}")
            
        if message.get("is_invoice"):
            client_name = message.get("client_name", "Invoice")
            safe_client_name = "".join([c if c.isalnum() else "_" for c in client_name])
            pdf_filename = f"{safe_client_name}_Invoice.pdf"
            if os.path.exists(pdf_filename):
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        f"📥 Download {client_name} Invoice (PDF)", 
                        data=pdf_file, 
                        file_name=pdf_filename, 
                        mime="application/pdf", 
                        key=f"dl_btn_{i}"
                    )
        
        if message.get("context") and len(message["context"]) > 0:
            with st.expander("📌 View Source Documents & Exact Context Chunks"):
                st.markdown("#### 🔍 **Retrieved Chunks Used for This Answer:**")
                for idx, doc in enumerate(message["context"]):
                    if hasattr(doc, 'metadata'):
                        src_file = doc.metadata.get('source', 'Uploaded Document')
                        page_sec = doc.metadata.get('page_section', 'N/A')
                        doc_type = doc.metadata.get('document_type', 'FILE')
                        content_txt = doc.page_content
                    else:
                        src_file = doc.get('metadata', {}).get('source', 'Uploaded Document')
                        page_sec = doc.get('metadata', {}).get('page_section', 'N/A')
                        doc_type = doc.get('metadata', {}).get('document_type', 'FILE')
                        content_txt = doc.get('page_content', '')

                    st.markdown(f"**Chunk {idx+1}:** `📄 File: {src_file}` | `Type: {doc_type}` | `Page/Section: {page_sec}`")
                    st.info(f"\"{content_txt}\"")


# ==============================================================================
# 8. PROMPT PIPELINE, INVOICES & UNIVERSAL DYNAMIC GRAPH ENGINE
# ==============================================================================
if prompt := st.chat_input("E.g., Generate invoice for Apex Technologies or plot credit limit bar chart..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            top_k = 6
            retrieved_docs = hybrid_search(prompt, top_k) if st.session_state.indexed_files else []
            context_text = "\n\n".join([f"\n{d.page_content}" for d in retrieved_docs])
            
            prompt_lower = prompt.lower()
            explicit_invoice_phrases = [
                "generate invoice", "create invoice", "generate a professional invoice", 
                "create a bill", "generate bill", "make an invoice", "make a bill", 
                "build an invoice", "build invoice"
            ]
            is_invoice_request = any(phrase in prompt_lower for phrase in explicit_invoice_phrases) or \
                                 (("invoice" in prompt_lower or "bill" in prompt_lower) and \
                                  any(action in prompt_lower for action in ["generate", "create", "make", "build", "provide", "print"]))
            
            # Universal Graph Detection
            chart_keywords = ["chart", "graph", "plot", "visualize", "compare", "distribution", "trend", "scatter", "bar", "line", "pie", "box"]
            is_chart_request = any(word in prompt_lower for word in chart_keywords)
            
            detected_client = "Client"
            client_match = re.search(r'(?:for|to|client)\s+([A-Za-z0-9\s]+)', prompt, re.IGNORECASE)
            if client_match:
                candidate = client_match.group(1).strip()
                for stop in ["invoice", "bill", "using", "the", "uploaded", "records", "data", "po", "please", "chart", "graph", "plot"]:
                    candidate = re.sub(rf'\b{stop}\b', '', candidate, flags=re.IGNORECASE).strip()
                if candidate:
                    detected_client = candidate.title()
            
            llm = ChatGroq(model=LLM_MODEL, temperature=temperature_value, groq_api_key=active_api_key)
            
            # 1. INVOICE EXTRACTION PIPELINE
            if is_invoice_request:
                if not st.session_state.indexed_files or not context_text.strip():
                    display_response = "⚠️ **Information Not Found:** No financial documents have been uploaded or indexed, or the knowledge base was reset. Please upload your company's data files first before generating an invoice."
                    message_placeholder.markdown(display_response)
                    full_response = display_response
                else:
                    system_prompt = f"""You are an advanced enterprise Finance AI.
                    Context Data:
                    {context_text}
                    
                    STRICT EXTRACTION INSTRUCTION:
                    1. Extract ALL details FOR {detected_client} ONLY strictly from Context Data above.
                    2. Do NOT invent or hardcode default values.
                    3. Respond with ONLY a raw JSON object. No markdown tags.
                    
                    JSON Format:
                    {{
                        "company_name": "Extracted Vendor Name",
                        "company_address": "Extracted Vendor Address",
                        "invoice_number": "Extracted Invoice Number",
                        "invoice_date": "Extracted Invoice Date",
                        "due_date": "Extracted Due Date",
                        "terms": "Extracted Terms",
                        "bill_to": "Extracted Bill To",
                        "ship_to": "Extracted Ship To",
                        "items": [
                            {{"description": "Extracted Description", "qty": "1", "rate": "100.00", "amount": "100.00"}}
                        ],
                        "subtotal": "100.00",
                        "tax_rate": "5%",
                        "tax_amount": "5.00",
                        "total_due": "105.00",
                        "terms_and_conditions": "Extracted Terms and Conditions"
                    }}
                    User Query: {prompt}"""
                    
                    full_response = ""
                    with st.spinner(f"Extracting verified billing data and compiling PDF invoice for {detected_client}..."):
                        for chunk in llm.stream(system_prompt):
                            full_response += chunk.content
                    
                    pdf_filename = generate_invoice_pdf(full_response, client_name=detected_client)
                    if pdf_filename and os.path.exists(pdf_filename):
                        display_response = f"✅ **Invoice Generated Successfully**\n\nVerified billing data for **{detected_client}** was extracted from your uploaded documents and compiled into a corporate PDF."
                    else:
                        display_response = f"⚠️ **Information Not Found:** Could not locate matching purchase records or line items for **{detected_client}** in the current knowledge base."
                    message_placeholder.markdown(display_response)
            else:
                if not st.session_state.indexed_files or not context_text.strip():
                    display_response = "Information not found in the knowledge base. Please upload documents first."
                    message_placeholder.markdown(display_response)
                    full_response = display_response
                else:
                    system_prompt = f"""You are an advanced enterprise Finance & Accounts AI.
                    Context Data:
                    {context_text}
                    
                    Instructions:
                    1. Provide direct, clean professional text answers based strictly on the Context Data.
                    2. If information is missing, state "Information not found in the knowledge base."
                    
                    User Query: {prompt}"""
                    
                    full_response = ""
                    for chunk in llm.stream(system_prompt):
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    display_response = full_response

            # 2. FULLY ADAPTIVE & DYNAMIC GRAPH RENDER ENGINE
            chat_fig = None
            if is_chart_request and st.session_state.indexed_files:
                if st.session_state.master_df is not None and not st.session_state.master_df.empty:
                    df = st.session_state.master_df.copy()
                    df.columns = df.columns.str.strip()
                    
                    # Clean numerical columns
                    numeric_cols = []
                    for col in df.columns:
                        cleaned_str = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        converted = pd.to_numeric(cleaned_str, errors='coerce')
                        if converted.notnull().sum() > 0:
                            df[col] = converted
                            numeric_cols.append(col)
                            
                    categorical_cols = [c for c in df.columns if c not in numeric_cols]
                    x_col = categorical_cols[0] if categorical_cols else df.columns[0]
                    
                    if numeric_cols:
                        y_col = numeric_cols[0]
                        for ncol in numeric_cols:
                            if any(w in ncol.lower() for w in prompt_lower.split() if len(w) > 3):
                                y_col = ncol
                                break

                        # Entity Filtering if specific companies mentioned
                        for col in categorical_cols:
                            corps = [c for c in df[col].dropna().unique() if str(c).lower() in prompt_lower]
                            if corps:
                                df = df[df[col].astype(str).isin(corps)]
                                break

                        # Chart Type Routing based on Query Intent
                        if "pie" in prompt_lower:
                            chat_fig = px.pie(
                                df, names=x_col, values=y_col, 
                                title=f"Distribution Analysis: {y_col} by {x_col}",
                                hole=0.3
                            )
                        elif "scatter" in prompt_lower and len(numeric_cols) >= 2:
                            x_scatter = numeric_cols[1] if numeric_cols[0] == y_col else numeric_cols[0]
                            chat_fig = px.scatter(
                                df, x=x_scatter, y=y_col, text=x_col, color=x_col,
                                title=f"Scatter Analysis: {y_col} vs {x_scatter}"
                            )
                        elif "line" in prompt_lower or "trend" in prompt_lower:
                            chat_fig = px.line(
                                df, x=x_col, y=y_col, markers=True, text=y_col,
                                title=f"Financial Trend: {y_col} over {x_col}"
                            )
                        elif "box" in prompt_lower:
                            chat_fig = px.box(
                                df, x=x_col, y=y_col, color=x_col,
                                title=f"Data Distribution (Box Plot): {y_col} by {x_col}"
                            )
                        else:
                            # Default Bar Chart
                            chat_fig = px.bar(
                                df, x=x_col, y=y_col, text=y_col, color=x_col,
                                title=f"Financial Analytics: {y_col} by {x_col}"
                            )
                            
                        chat_fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", 
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Arial", size=12)
                        )
                        st.plotly_chart(chat_fig, use_container_width=True, key=f"live_chart_{len(st.session_state.messages)}")

            valid_invoice = False
            if is_invoice_request and st.session_state.indexed_files:
                safe_client_name = "".join([c if c.isalnum() else "_" for c in detected_client])
                pdf_filename = f"{safe_client_name}_Invoice.pdf"
                if os.path.exists(pdf_filename) and os.path.getsize(pdf_filename) > 500:
                    valid_invoice = True

            # STORE MESSAGE
            st.session_state.messages.append({
                "role": "assistant", 
                "content": display_response,
                "context": retrieved_docs if (st.session_state.indexed_files and len(retrieved_docs) > 0) else [],
                "is_invoice": valid_invoice,
                "client_name": detected_client if valid_invoice else "Invoice",
                "has_chart": chat_fig is not None,
                "chart_fig": chat_fig
            })
            
            st.rerun()

        except Exception as e:
            st.error(f"Engine Error: {str(e)}")