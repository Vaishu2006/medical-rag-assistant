"""
MedQuery AI — Pre-compute Embeddings
======================================
Run this ONCE locally. It saves embeddings to embeddings_cache.npz
Commit that file to GitHub so Streamlit Cloud just loads it instantly
instead of recomputing BioBERT embeddings on every cold start.

Startup time:  WITH this file  →  ~15 seconds
               WITHOUT         →  ~3-5 minutes

Run: python precompute_embeddings.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
OUTPUT_FILE     = "embeddings_cache.npz"

def chunk_text(text, size=400, overlap=50):
    words = text.split()
    chunks, step = [], size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if len(chunk.split()) >= 50:
            chunks.append(chunk)
    return chunks

print("Loading pubmed_raw.json...")
with open("pubmed_raw.json", "r") as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles. Chunking...")

documents, pubmed_ids, titles, authors, years, urls = [], [], [], [], [], []

for article in articles:
    full_text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
    for chunk in chunk_text(full_text):
        documents.append(chunk)
        pubmed_ids.append(article["pubmed_id"])
        titles.append(article["title"])
        authors.append(article["authors"])
        years.append(article["year"])
        urls.append(article["source_url"])

print(f"Created {len(documents)} chunks. Computing BioBERT embeddings...")
print("This takes ~5-10 min locally but saves 3-5 min on every cloud startup.\n")

model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(
    documents,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(f"\nSaving to {OUTPUT_FILE}...")
np.savez_compressed(
    OUTPUT_FILE,
    embeddings = embeddings,
    documents  = np.array(documents,  dtype=object),
    pubmed_ids = np.array(pubmed_ids, dtype=object),
    titles     = np.array(titles,     dtype=object),
    authors    = np.array(authors,    dtype=object),
    years      = np.array(years,      dtype=object),
    urls       = np.array(urls,       dtype=object),
)

size_mb = __import__("os").path.getsize(OUTPUT_FILE) / 1024 / 1024
print(f"✓ Saved {OUTPUT_FILE} ({size_mb:.1f} MB)")
print(f"  {len(documents)} chunks, {embeddings.shape[1]}-dim BioBERT embeddings")
print(f"\nNext steps:")
print(f"  1. git add embeddings_cache.npz")
print(f"  2. git commit -m 'add precomputed embeddings cache'")
print(f"  3. git push origin main")
print(f"  4. Reboot app on Streamlit Cloud")
print(f"  → Cold start will now take ~15 seconds instead of 3-5 minutes")
