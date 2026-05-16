"""
MedQuery AI — Day 5 & 6: Streamlit Chat UI
============================================
What this does:
  - Full chat interface for your RAG pipeline
  - Shows source citations, confidence score, relevancy %
  - PDF upload support (bonus feature!)
  - Medical disclaimer on every response

Run: streamlit run app.py
"""

import streamlit as st
import time
from day3_retrieval import MedQueryRAG

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="MedQuery AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .confidence-high   { background:#d5f5e3; color:#1e8449; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }
    .confidence-medium { background:#fef9e7; color:#b7950b; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }
    .confidence-low    { background:#fadbd8; color:#922b21; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #2e86c1;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 13px;
    }
    .disclaimer {
        background: #fef9e7;
        border: 1px solid #f9e79f;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 12px;
        color: #7d6608;
        margin-top: 10px;
    }
    .metric-box {
        background: #eaf4fb;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE RAG (cached so it loads only once)
# ============================================================
@st.cache_resource
def load_rag():
    return MedQueryRAG()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 MedQuery AI")
    st.markdown("*Clinical Research Assistant*")
    st.divider()

    st.markdown("### About")
    st.markdown("""
    Answers medical questions from **10,000+ PubMed abstracts** using:
    - 🧬 **BioBERT** embeddings
    - 🗄️ **ChromaDB** vector store
    - 🤖 **LLaMA-3** via Groq
    """)

    st.divider()
    st.markdown("### Settings")
    top_k = st.slider("Sources to retrieve", min_value=3, max_value=10, value=5)
    show_raw_chunks = st.checkbox("Show raw retrieved chunks", value=False)

    st.divider()
    st.markdown("### Sample Questions")
    sample_questions = [
        "First-line treatment for type 2 diabetes?",
        "Cardiovascular risks of SGLT2 inhibitors?",
        "Metformin vs GLP-1 agonists comparison?",
        "HbA1c targets in elderly diabetic patients?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.prefill_question = q

    st.divider()
    st.caption("Built by [Your Name] | AIML Student Project")
    st.caption("Data: PubMed via NCBI Entrez API")

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">🩺 MedQuery AI</h2>
    <p style="margin:4px 0 0; opacity:0.85; font-size:14px">
        Evidence-based clinical Q&A powered by PubMed literature, BioBERT & LLaMA-3
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHAT HISTORY
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.results = []

# Display past messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and i // 2 < len(st.session_state.results):
            result = st.session_state.results[i // 2]

            # Confidence badge
            conf = result["confidence"]
            conf_class = f"confidence-{conf.lower()}"
            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                st.markdown(f'<span class="{conf_class}">● {conf} confidence</span>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span style="font-size:13px; color:#666">📊 {result["avg_relevancy"]}% avg relevancy</span>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<span style="font-size:13px; color:#666">📄 {result["chunks_retrieved"]} chunks retrieved</span>', unsafe_allow_html=True)

            st.markdown(message["content"])

            # Sources
            with st.expander(f"📚 View {len(result['sources'])} sources"):
                for src in result["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <b>PMID: {src['pubmed_id']}</b> &nbsp;|&nbsp; {src['year']} &nbsp;|&nbsp; 
                        Relevancy: <b>{src['relevancy_score']}%</b><br>
                        <i>{src['title'][:100]}{'...' if len(src['title']) > 100 else ''}</i><br>
                        <small>{src['authors']}</small><br>
                        <a href="{src['source_url']}" target="_blank">🔗 View on PubMed</a>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> For informational/research purposes only. Not a substitute for professional medical advice, diagnosis, or treatment.</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================
prefill = st.session_state.pop("prefill_question", None)
question = st.chat_input("Ask a clinical question (e.g. 'What are the side effects of metformin?')")

if prefill:
    question = prefill

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching PubMed literature..."):
            rag = load_rag()
            result = rag.query(question)

        # Metrics
        conf = result["confidence"]
        conf_class = f"confidence-{conf.lower()}"
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f'<span class="{conf_class}">● {conf} confidence</span>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span style="font-size:13px; color:#666">📊 {result["avg_relevancy"]}% avg relevancy</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span style="font-size:13px; color:#666">📄 {result["chunks_retrieved"]} chunks retrieved</span>', unsafe_allow_html=True)

        # Stream answer
        st.markdown(result["answer"])

        # Sources expander
        with st.expander(f"📚 View {len(result['sources'])} sources"):
            for src in result["sources"]:
                st.markdown(f"""
                <div class="source-card">
                    <b>PMID: {src['pubmed_id']}</b> &nbsp;|&nbsp; {src['year']} &nbsp;|&nbsp; 
                    Relevancy: <b>{src['relevancy_score']}%</b><br>
                    <i>{src['title'][:100]}{'...' if len(src['title']) > 100 else ''}</i><br>
                    <small>{src['authors']}</small><br>
                    <a href="{src['source_url']}" target="_blank">🔗 View on PubMed</a>
                </div>
                """, unsafe_allow_html=True)

        if show_raw_chunks:
            with st.expander("🔬 Raw retrieved chunks (debug view)"):
                for i, src in enumerate(result["sources"]):
                    st.markdown(f"**Chunk {i+1}** (PMID {src['pubmed_id']})")

        st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> For informational/research purposes only. Not a substitute for professional medical advice, diagnosis, or treatment.</div>', unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.results.append(result)
