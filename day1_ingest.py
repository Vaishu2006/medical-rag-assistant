
import os
import time
import json
from typing import List, Dict

# --- Biopython for PubMed API ---
from Bio import Entrez

# --- ChromaDB: local vector store ---
import chromadb
from chromadb.config import Settings

# --- BioBERT embeddings via sentence-transformers ---
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION — edit these
# ============================================================
ENTREZ_EMAIL = "anjana.devel@gmail.com"   # Required by NCBI (free)
SEARCH_QUERY = "diabetes treatment guidelines"  # Change to your focus area
MAX_ARTICLES = 50                        # Start with 200, scale to 1000+
CHUNK_SIZE = 400                            # Words per chunk
CHUNK_OVERLAP = 50                          # Words of overlap between chunks
CHROMA_DB_PATH = "./chroma_db"             # Local folder for ChromaDB
COLLECTION_NAME = "medquery_pubmed"

# BioBERT model — trained on PubMed + PMC (your secret weapon!)
# This understands medical language far better than generic models
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
# Alternatives:
# "ncats/MedCPT-Query-Encoder"  — even better for medical retrieval
# "all-MiniLM-L6-v2"           — generic, weaker for medical text

# ============================================================
# STEP 1: Fetch PubMed abstracts
# ============================================================

def fetch_pubmed_abstracts(query: str, max_results: int) -> List[Dict]:
    """
    Fetch abstracts from PubMed using the free Entrez API.
    Returns a list of dicts with title, abstract, authors, year, pubmed_id.
    """
    Entrez.email = ENTREZ_EMAIL
    print(f"\n[1/4] Searching PubMed for: '{query}'")
    print(f"      Fetching up to {max_results} articles...\n")

    # Search PubMed
    search_handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )
    search_results = Entrez.read(search_handle)
    search_handle.close()

    ids = search_results["IdList"]
    print(f"      Found {len(ids)} article IDs. Fetching full records...")

    # Fetch full records in batches of 50 (NCBI rate limit)
    articles = []
    batch_size = 50

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch_ids),
            rettype="xml",
            retmode="xml"
        )
        records = Entrez.read(fetch_handle)
        fetch_handle.close()

        for record in records["PubmedArticle"]:
            try:
                article = record["MedlineCitation"]["Article"]

                # Extract title
                title = str(article.get("ArticleTitle", ""))

                # Extract abstract (some papers have no abstract)
                abstract_obj = article.get("Abstract", {})
                abstract_text = abstract_obj.get("AbstractText", [""])
                if isinstance(abstract_text, list):
                    abstract = " ".join(str(t) for t in abstract_text)
                else:
                    abstract = str(abstract_text)

                # Skip papers with no abstract
                if not abstract.strip() or len(abstract) < 100:
                    continue

                # Extract metadata
                pubmed_id = str(record["MedlineCitation"]["PMID"])
                year = str(
                    article.get("Journal", {})
                    .get("JournalIssue", {})
                    .get("PubDate", {})
                    .get("Year", "Unknown")
                )

                # Extract authors
                authors_list = article.get("AuthorList", [])
                authors = []
                for author in authors_list[:3]:  # First 3 authors
                    if "LastName" in author:
                        authors.append(author["LastName"])
                author_str = ", ".join(authors) + (" et al." if len(authors_list) > 3 else "")

                articles.append({
                    "pubmed_id": pubmed_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": author_str,
                    "year": year,
                    "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
                })

            except Exception as e:
                continue  # Skip malformed records

        # Be polite to NCBI servers
        time.sleep(0.5)
        print(f"      Fetched {min(i + batch_size, len(ids))}/{len(ids)} articles...")

    print(f"\n      ✓ Successfully fetched {len(articles)} articles with abstracts")
    return articles


# ============================================================
# STEP 2: Chunk documents
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Why overlap? So that a sentence split across two chunks
    is still fully retrievable from either chunk.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 50:  # Skip tiny trailing chunks
            chunks.append(chunk)

    return chunks


def prepare_documents(articles: List[Dict]) -> tuple:
    """
    Convert articles into chunks suitable for embedding.
    Returns (documents, metadatas, ids)
    """
    print("\n[2/4] Chunking documents...")
    documents = []
    metadatas = []
    ids = []
    chunk_count = 0

    for article in articles:
        # Combine title + abstract for richer context
        full_text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
        chunks = chunk_text(full_text)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"pmid_{article['pubmed_id']}_chunk_{idx}"
            documents.append(chunk)
            metadatas.append({
                "pubmed_id": article["pubmed_id"],
                "title": article["title"],
                "authors": article["authors"],
                "year": article["year"],
                "source_url": article["source_url"],
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })
            ids.append(chunk_id)
            chunk_count += 1

    print(f"      ✓ Created {chunk_count} chunks from {len(articles)} articles")
    return documents, metadatas, ids


# ============================================================
# STEP 3: Create BioBERT embeddings + store in ChromaDB
# ============================================================
# ============================================================
# STEP 3: Create BioBERT embeddings + store in ChromaDB
# ============================================================

class BioBERTEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using BioBERT.

    WHY BioBERT over generic embeddings?
    - Trained on 29M PubMed abstracts + full PMC texts
    - Understands: "MI" = "myocardial infarction" = "heart attack"
    - Handles clinical abbreviations, drug names, ICD codes natively
    - 15-20% better retrieval accuracy on medical queries vs generic models
    """

    def __init__(self, model_name: str):
        print(f"\n[3/4] Loading BioBERT model: {model_name}")
        print("      (First run downloads ~400MB — subsequent runs are instant)")
        self.model = SentenceTransformer(model_name)
        print("      ✓ BioBERT model loaded")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Used by ChromaDB when adding documents.
        """
        embeddings = self.model.encode(
            input,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """
        Required by newer versions of ChromaDB.
        """
        return self.__call__(input)

    def embed_query(self, input: str) -> List[float]:
        """
        Required by newer versions of ChromaDB when querying.
        """
        embedding = self.model.encode(
            input,
            normalize_embeddings=True
        )
        return embedding.tolist()


def setup_chromadb(embedding_fn) -> chromadb.Collection:
    """
    Initialize ChromaDB with a persistent local store.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"      Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    return collection



def ingest_to_chromadb(collection, documents, metadatas, ids, batch_size=100):
    """Add documents to ChromaDB in batches."""
    print(f"\n[4/4] Embedding & storing {len(documents)} chunks in ChromaDB...")
    print("      This takes 5-15 minutes depending on your hardware.")
    print("      BioBERT is creating 768-dimensional medical embeddings...\n")

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )

        progress = min(i + batch_size, len(documents))
        print(f"      Stored {progress}/{len(documents)} chunks...")

    print(f"\n      ✓ All chunks embedded and stored in ChromaDB at '{CHROMA_DB_PATH}'")


# ============================================================
# STEP 4: Test retrieval
# ============================================================

def test_retrieval(collection, embedding_fn, query: str, n_results: int = 3):
    """
    Quick test to verify your pipeline works end-to-end.
    This is exactly how the RAG query step will work later.
    """
    print(f"\n{'='*60}")
    print(f"TEST QUERY: '{query}'")
    print(f"{'='*60}")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        relevancy = round((1 - dist) * 100, 1)  # Convert cosine distance to %
        print(f"\n--- Result {i+1} (Relevancy: {relevancy}%) ---")
        print(f"Title:   {meta['title'][:80]}...")
        print(f"Authors: {meta['authors']} ({meta['year']})")
        print(f"Source:  {meta['source_url']}")
        print(f"Chunk:   {doc[:200]}...")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MedQuery AI — Multi-Topic Ingestion Pipeline")
    print("  PubMed → BioBERT → ChromaDB")
    print("=" * 60)

    # Load BioBERT + setup ChromaDB once
    embedding_fn = BioBERTEmbeddingFunction(EMBEDDING_MODEL)
    collection = setup_chromadb(embedding_fn)

    # Broader topic coverage for stronger medical knowledge base
    SEARCH_QUERIES = [
    # Common diseases
    "diabetes treatment guidelines",
    "hypertension treatment guidelines",
    "asthma treatment guidelines",
    "pneumonia treatment guidelines",
    "thyroid disease management",
    "chronic kidney disease treatment",
    "depression treatment guidelines",
    "anxiety disorder treatment",

    # First aid
    "burn first aid",
    "fracture first aid",
    "bleeding control first aid",
    "CPR guidelines",
    "snake bite first aid",

    # Medicines
    "paracetamol uses side effects",
    "ibuprofen uses side effects",
    "amoxicillin uses side effects",
    "metformin therapy",
    "omeprazole uses side effects",

    # Infectious diseases
    "dengue fever management",
    "COVID-19 treatment guidelines",
]
    all_articles = []

    # Fetch, chunk, and ingest each topic
    for idx, query in enumerate(SEARCH_QUERIES, start=1):
        print(f"\n{'=' * 60}")
        print(f"TOPIC {idx}/{len(SEARCH_QUERIES)}: {query}")
        print(f"{'=' * 60}")

        # Step 1: Fetch articles for this topic
        articles = fetch_pubmed_abstracts(query, 250)
        all_articles.extend(articles)

        # Step 2: Chunk documents
        documents, metadatas, ids = prepare_documents(articles)

        # Step 3: Ingest into ChromaDB
        ingest_to_chromadb(collection, documents, metadatas, ids)

    # Save all fetched articles as backup
    with open("pubmed_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2)

    print(f"\n      Raw articles saved to pubmed_raw.json")
    print(f"      Total articles fetched: {len(all_articles)}")
    print(f"      Total chunks stored: {collection.count()}")

    # Optional retrieval tests
    test_retrieval(
        collection,
        embedding_fn,
        "What are the recommended first-line treatments for type 2 diabetes?"
    )

    test_retrieval(
        collection,
        embedding_fn,
        "What are the cardiovascular benefits of SGLT2 inhibitors?"
    )

    print("\n" + "=" * 60)
    print("  ✓ Multi-topic ingestion complete! ChromaDB is ready.")
    print("  Next: Run day3_retrieval.py")
    print("=" * 60)