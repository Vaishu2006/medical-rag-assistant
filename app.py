"""
MedQuery AI — Streamlit Cloud Version
=======================================
Key differences from local version:
  - API key read from st.secrets (Streamlit Cloud secrets panel)
  - ChromaDB rebuilt on startup from pubmed_raw.json (committed to repo)
  - chroma_db/ folder lives in /tmp (writable on cloud)

Deploy steps:
  1. Push this file + requirements.txt + pubmed_raw.json to GitHub
  2. Go to share.streamlit.io → New app → connect repo
  3. Add GROQ_API_KEY in App Settings → Secrets
"""

import os
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# ============================================================
# CONFIG
# ============================================================
COLLECTION_NAME  = "medquery_pubmed"
EMBEDDING_MODEL  = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
LLM_MODEL        = "llama-3.3-70b-versatile"
TOP_K_RESULTS    = 5
CHROMA_DB_PATH   = "/tmp/chroma_db"  # writable on Streamlit Cloud

SYSTEM_PROMPT = """You are MedQuery AI, a clinical research assistant that answers
medical questions based strictly on peer-reviewed PubMed literature.

STRICT RULES:
1. Answer ONLY using the provided context from PubMed abstracts
2. If context is insufficient say: "The provided literature does not sufficiently address this."
3. Always cite sources using [PMID: XXXXXXX] inline
4. Include a confidence level: HIGH / MEDIUM / LOW

FORMAT:
**Answer:** [evidence-based answer with inline citations]

**Key Evidence:**
- [bullet points of key findings]

**Confidence:** [HIGH/MEDIUM/LOW] — [reason]

**Sources:**
- [PMID: XXXXX] Title | Authors (Year)

⚠️ Disclaimer: For informational purposes only. Not medical advice."""

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="MedQuery AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a5276,#2e86c1);padding:1.5rem 2rem;border-radius:12px;color:white;margin-bottom:1.5rem;}
.confidence-high{background:#d5f5e3;color:#1e8449;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;}
.confidence-medium{background:#fef9e7;color:#b7950b;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;}
.confidence-low{background:#fadbd8;color:#922b21;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;}
.source-card{background:#f8f9fa;border-left:4px solid #2e86c1;padding:10px 14px;border-radius:6px;margin:6px 0;font-size:13px;}
.disclaimer{background:#fef9e7;border:1px solid #f9e79f;padding:10px 14px;border-radius:8px;font-size:12px;color:#7d6608;margin-top:10px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# GET GROQ KEY — works both locally and on Streamlit Cloud
# ============================================================
def get_groq_key():
    # Streamlit Cloud: reads from secrets panel
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    # Local fallback: reads from environment / .env
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        st.error("❌ GROQ_API_KEY not found. Add it in App Settings → Secrets on Streamlit Cloud.")
        st.stop()
    return key

# ============================================================
# BUILD RAG PIPELINE — cached, runs once per session
# ============================================================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    groq_key = get_groq_key()

    # 1. Load BioBERT (only for query embedding — NOT for ingestion)
    with st.spinner("🧬 Loading BioBERT model..."):
        model = SentenceTransformer(EMBEDDING_MODEL)

    # 2. Load ChromaDB — fast path if cache exists, slow path if not
    with st.spinner("🗄️ Loading vector store..."):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        import os, numpy as np

        if os.path.exists("embeddings_cache.npz"):
            # ⚡ FAST PATH: load pre-computed embeddings (~15 sec)
            cache      = np.load("embeddings_cache.npz", allow_pickle=True)
            documents  = cache["documents"].tolist()
            embeddings = cache["embeddings"]
            metadatas  = [
                {
                    "pubmed_id":  str(cache["pubmed_ids"][i]),
                    "title":      str(cache["titles"][i]),
                    "authors":    str(cache["authors"][i]),
                    "year":       str(cache["years"][i]),
                    "source_url": str(cache["urls"][i]),
                }
                for i in range(len(documents))
            ]
            ids = [f"chunk_{i}" for i in range(len(documents))]

            # Add in batches
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                collection.add(
                    documents  = documents[i:i+batch_size],
                    embeddings = embeddings[i:i+batch_size].tolist(),
                    metadatas  = metadatas[i:i+batch_size],
                    ids        = ids[i:i+batch_size]
                )
        else:
            # 🐢 SLOW PATH: recompute from pubmed_raw.json (fallback)
            with open("pubmed_raw.json", "r") as f:
                articles = json.load(f)

            def chunk_text(text, size=400, overlap=50):
                words = text.split()
                chunks, step = [], size - overlap
                for i in range(0, len(words), step):
                    chunk = " ".join(words[i:i + size])
                    if len(chunk.split()) >= 50:
                        chunks.append(chunk)
                return chunks

            documents, metadatas, ids = [], [], []
            for article in articles:
                full_text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
                for idx, chunk in enumerate(chunk_text(full_text)):
                    documents.append(chunk)
                    metadatas.append({
                        "pubmed_id":  article["pubmed_id"],
                        "title":      article["title"],
                        "authors":    article["authors"],
                        "year":       article["year"],
                        "source_url": article["source_url"],
                    })
                    ids.append(f"pmid_{article['pubmed_id']}_chunk_{idx}")

            for i in range(0, len(documents), 64):
                batch      = documents[i:i+64]
                embeddings = model.encode(batch, normalize_embeddings=True).tolist()
                collection.add(
                    documents  = batch,
                    embeddings = embeddings,
                    metadatas  = metadatas[i:i+64],
                    ids        = ids[i:i+64]
                )

    # 3. Groq LLM
    llm = Groq(api_key=groq_key)
    return model, collection, llm


def run_query(question, model, collection, llm):
    emb = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=emb,
        n_results=TOP_K_RESULTS,
        include=["documents", "metadatas", "distances"]
    )
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, m in enumerate(metas):
        m["relevancy_score"] = round((1 - dists[i]) * 100, 1)

    context = "\n\n---\n\n".join(
        f"[Source {i+1}] PMID:{m['pubmed_id']} | {m['title']} | "
        f"{m['authors']} ({m['year']}) | {m['relevancy_score']}%\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas))
    )

    avg  = sum(m["relevancy_score"] for m in metas) / len(metas)
    dist = len(set(m["pubmed_id"] for m in metas))
    conf = "HIGH" if avg >= 70 and dist >= 3 else ("MEDIUM" if avg >= 50 else "LOW")

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"QUESTION: {question}\n\nCONTEXT:\n{context}"}
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return {
        "answer":            resp.choices[0].message.content,
        "sources":           metas,
        "confidence":        conf,
        "avg_relevancy":     round(avg, 1),
        "chunks_retrieved":  len(docs),
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 MedQuery AI")
    st.markdown("*Clinical Research Assistant*")
    st.divider()
    st.markdown("""Answers medical questions from **10,000+ PubMed abstracts** using:
- 🧬 **BioBERT** embeddings
- 🗄️ **ChromaDB** vector store
- 🤖 **LLaMA-3** via Groq""")
    st.divider()
    st.markdown("### Sample Questions")
    for q in [
        "First-line treatment for type 2 diabetes?",
        "Cardiovascular risks of SGLT2 inhibitors?",
        "Metformin vs GLP-1 agonists comparison?",
        "HbA1c targets in elderly patients?",
    ]:
        if st.button(q, use_container_width=True):
            st.session_state.prefill = q
    st.divider()
    st.caption("Built by [Your Name] · AIML Student Project")
    st.caption("Data: PubMed via NCBI Entrez API")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
  <h2 style="margin:0">🩺 MedQuery AI</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:14px">
    Evidence-based clinical Q&A · PubMed · BioBERT · LLaMA-3
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHAT HISTORY
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.results  = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            res = st.session_state.results[i // 2]
            c   = res["confidence"]
            c1, c2, c3 = st.columns([2, 2, 3])
            c1.markdown(f'<span class="confidence-{c.lower()}">● {c}</span>', unsafe_allow_html=True)
            c2.markdown(f'<small>📊 {res["avg_relevancy"]}% relevancy</small>', unsafe_allow_html=True)
            c3.markdown(f'<small>📄 {res["chunks_retrieved"]} chunks</small>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            with st.expander(f"📚 {len(res['sources'])} sources"):
                for s in res["sources"]:
                    st.markdown(
                        f'<div class="source-card"><b>PMID {s["pubmed_id"]}</b> | '
                        f'{s["year"]} | <b>{s["relevancy_score"]}%</b><br>'
                        f'<i>{s["title"][:90]}...</i><br>'
                        f'<small>{s["authors"]}</small><br>'
                        f'<a href="{s["source_url"]}" target="_blank">🔗 PubMed</a></div>',
                        unsafe_allow_html=True
                    )
            st.markdown('<div class="disclaimer">⚠️ For informational purposes only. Not medical advice.</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# ============================================================
# CHAT INPUT
# ============================================================
prefill  = st.session_state.pop("prefill", None)
question = st.chat_input("Ask a clinical question...")
if prefill:
    question = prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        model, collection, llm = load_pipeline()
        with st.spinner("🔍 Searching PubMed literature..."):
            res = run_query(question, model, collection, llm)

        c = res["confidence"]
        c1, c2, c3 = st.columns([2, 2, 3])
        c1.markdown(f'<span class="confidence-{c.lower()}">● {c}</span>', unsafe_allow_html=True)
        c2.markdown(f'<small>📊 {res["avg_relevancy"]}% relevancy</small>', unsafe_allow_html=True)
        c3.markdown(f'<small>📄 {res["chunks_retrieved"]} chunks</small>', unsafe_allow_html=True)
        st.markdown(res["answer"])
        with st.expander(f"📚 {len(res['sources'])} sources"):
            for s in res["sources"]:
                st.markdown(
                    f'<div class="source-card"><b>PMID {s["pubmed_id"]}</b> | '
                    f'{s["year"]} | <b>{s["relevancy_score"]}%</b><br>'
                    f'<i>{s["title"][:90]}...</i><br>'
                    f'<small>{s["authors"]}</small><br>'
                    f'<a href="{s["source_url"]}" target="_blank">🔗 PubMed</a></div>',
                    unsafe_allow_html=True
                )
        st.markdown('<div class="disclaimer">⚠️ For informational purposes only. Not medical advice.</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    st.session_state.results.append(res)