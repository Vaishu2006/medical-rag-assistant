"""
MedQuery AI — FIX SCRIPT
=========================
Run this FIRST to diagnose your ChromaDB and fix the dimension mismatch.

The error "expecting 768, got 384" means:
  - Your ChromaDB collection was created with a 384-dim model (all-MiniLM)
  - But day3_retrieval.py tries to query with a 768-dim model (BioBERT)
  - Fix: delete the old collection and re-ingest with BioBERT consistently

Run: python fix_and_reingest.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "medquery_pubmed"

# ============================================================
# STEP 1: DIAGNOSE — check what model/dimension is in your DB
# ============================================================
print("=" * 55)
print("  MedQuery AI — Diagnosis & Fix")
print("=" * 55)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

print("\n[1/4] Checking existing ChromaDB collections...")
existing = client.list_collections()
print(f"      Collections found: {[c.name for c in existing]}")

if not existing:
    print("      ⚠  No collections found — ChromaDB is empty.")
    print("      → Run day1_ingest.py first, then come back.")
    exit()

try:
    col = client.get_collection(COLLECTION_NAME)
    count = col.count()
    print(f"      Collection '{COLLECTION_NAME}' has {count} chunks stored")

    # Peek at one record to check its embedding dimension
    sample = col.peek(limit=1)
    if sample["embeddings"] and len(sample["embeddings"]) > 0:
        dim = len(sample["embeddings"][0])
        print(f"      Embedding dimension in DB: {dim}")
        if dim == 384:
            print("      ⚠  Detected 384-dim → was created with all-MiniLM (NOT BioBERT)")
            print("      → Will delete and re-ingest with BioBERT (768-dim)")
        elif dim == 768:
            print("      ✓ Detected 768-dim → BioBERT. Retrieval should work fine.")
            print("      → If you still see errors, check GROQ_API_KEY in day3_retrieval.py")
            exit()
    else:
        print("      Could not read embeddings from peek — will reset anyway")
except Exception as e:
    print(f"      Could not read collection: {e}")

# ============================================================
# STEP 2: DELETE the mismatched collection
# ============================================================
print("\n[2/4] Deleting old mismatched collection...")
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"      ✓ Deleted '{COLLECTION_NAME}'")
except Exception as e:
    print(f"      Could not delete: {e}")

# ============================================================
# STEP 3: RE-INGEST using the correct BioBERT model
# ============================================================
print("\n[3/4] Re-ingesting with BioBERT (768-dim)...")

# Check if raw articles were saved from day1
if not os.path.exists("pubmed_raw.json"):
    print("      ⚠  pubmed_raw.json not found!")
    print("      → Run day1_ingest.py again first (it re-downloads PubMed data)")
    print("         Tip: it saves pubmed_raw.json so you don't re-download next time")
    exit()

print("      Loading pubmed_raw.json (no re-download needed)...")
with open("pubmed_raw.json", "r") as f:
    articles = json.load(f)
print(f"      ✓ Loaded {len(articles)} articles from cache")

# Load BioBERT (768-dim) — the correct model
BIOBERT_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
print(f"\n      Loading BioBERT model: {BIOBERT_MODEL}")
print("      (Downloads ~400MB on first run, instant after that)")
model = SentenceTransformer(BIOBERT_MODEL)

# Verify it really is 768-dim
test_emb = model.encode(["test"])
actual_dim = len(test_emb[0])
print(f"      ✓ Model loaded — embedding dimension: {actual_dim}")
assert actual_dim == 768, f"Expected 768, got {actual_dim}. Wrong model?"

# Create new collection
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"      ✓ Fresh collection '{COLLECTION_NAME}' created")

# Chunk articles
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks, step = [], chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 50:
            chunks.append(chunk)
    return chunks

documents, metadatas, ids = [], [], []
for article in articles:
    full_text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
    chunks = chunk_text(full_text)
    for idx, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "pubmed_id": article["pubmed_id"],
            "title": article["title"],
            "authors": article["authors"],
            "year": article["year"],
            "source_url": article["source_url"],
            "chunk_index": idx,
        })
        ids.append(f"pmid_{article['pubmed_id']}_chunk_{idx}")

print(f"      Prepared {len(documents)} chunks for embedding...")

# Embed and store in batches
batch_size = 64
for i in range(0, len(documents), batch_size):
    batch_docs  = documents[i:i + batch_size]
    batch_meta  = metadatas[i:i + batch_size]
    batch_ids   = ids[i:i + batch_size]

    # Embed using BioBERT
    embeddings = model.encode(
        batch_docs,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection.add(
        documents=batch_docs,
        embeddings=embeddings,
        metadatas=batch_meta,
        ids=batch_ids
    )
    done = min(i + batch_size, len(documents))
    print(f"      Stored {done}/{len(documents)} chunks...", end="\r")

print(f"\n      ✓ All {len(documents)} chunks stored with BioBERT (768-dim)")

# ============================================================
# STEP 4: VERIFY — do a test query to confirm it works
# ============================================================
print("\n[4/4] Verifying retrieval works...")

test_query = "treatment for type 2 diabetes"
query_embedding = model.encode([test_query], normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print(f"\n      Test query: '{test_query}'")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    relevancy = round((1 - dist) * 100, 1)
    print(f"\n      Result {i+1} ({relevancy}% relevancy):")
    print(f"        Title:  {meta['title'][:70]}...")
    print(f"        PMID:   {meta['pubmed_id']}  ({meta['year']})")

print("\n" + "=" * 55)
print("  ✓ Fix complete! ChromaDB is ready with BioBERT.")
print("\n  NEXT STEPS:")
print("  1. Make sure GROQ_API_KEY is set:")
print("     Windows: set GROQ_API_KEY=gsk_xxxx")
print("     Or edit line 26 in day3_retrieval.py directly")
print("\n  2. Test retrieval:")
print("     python day3_retrieval.py")
print("\n  3. Launch the app:")
print("     streamlit run app.py")
print("=" * 55)