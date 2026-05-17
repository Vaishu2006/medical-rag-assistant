"""
MedQuery AI — Day 3 & 4: RAG Query Chain (FIXED VERSION)
==========================================================
KEY FIX: We now pass query_embeddings= explicitly to ChromaDB
instead of query_texts=. This bypasses ChromaDB's internal
embedding function entirely → no more dimension mismatch errors.

Run: python day3_retrieval.py
"""

import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# ============================================================
# CONFIGURATION
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "medquery_pubmed"
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"  # 768-dim
TOP_K_RESULTS = 5
LLM_MODEL = "llama-3.3-70b-versatile"

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

⚠️ Disclaimer: For informational purposes only. Not medical advice.
"""

class MedQueryRAG:

    def __init__(self):
        print("Initializing MedQuery AI...")

        # Load embedding model
        print(f"  Loading model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        test = self.embedding_model.encode(["test"])
        self.emb_dim = len(test[0])
        print(f"  Model embedding dimension: {self.emb_dim}")

        # Connect to ChromaDB WITHOUT embedding function
        print("  Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        try:
            self.collection = client.get_collection(COLLECTION_NAME)
            sample = self.collection.peek(limit=1)

            embs = sample["embeddings"]
            if embs is not None and len(embs) > 0:
                db_dim = len(embs[0])
                print(f"  ChromaDB stored dimension: {db_dim}")

                if db_dim != self.emb_dim:
                    print(f"\n  ⚠  Dimension mismatch: DB={db_dim}, model={self.emb_dim}")
                    print(f"  AUTO-FIX: switching model to match DB...")
                    if db_dim == 384:
                        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                        self.emb_dim = 384
                        print("  ✓ Switched to all-MiniLM-L6-v2 (384-dim)")
                    elif db_dim == 768:
                        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                        self.emb_dim = 768
                        print("  ✓ Switched to BioBERT (768-dim)")

            print(f"  ✓ ChromaDB ready — {self.collection.count()} chunks")

        except Exception as e:
            print(f"  ❌ ChromaDB error: {e}")
            print("  → Run fix_and_reingest.py first")
            raise

        if GROQ_API_KEY == "YOUR_GROQ_KEY_HERE":
            print("\n  ❌ GROQ_API_KEY not set!")
            print("  → Get free key: https://console.groq.com")
            print("  → Windows CMD: set GROQ_API_KEY=gsk_xxxx")
            print("  → Or hardcode it on line 17 of this file")
            raise ValueError("Set your GROQ_API_KEY first")

        self.llm = Groq(api_key=GROQ_API_KEY)
        print("  ✓ Groq connected. MedQuery AI ready!\n")

    def retrieve(self, query: str, n_results: int = TOP_K_RESULTS):
        # THE FIX: embed query ourselves, pass raw vector to ChromaDB
        # Never use query_texts= — that triggers ChromaDB's internal embedder
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,  # ← KEY FIX
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, meta in enumerate(metadatas):
            meta["relevancy_score"] = round((1 - distances[i]) * 100, 1)

        return documents, metadatas

    def build_context(self, documents, metadatas):
        parts = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            parts.append(
                f"[Source {i+1}] PMID: {meta['pubmed_id']} | "
                f"{meta['title']} | {meta['authors']} ({meta['year']}) "
                f"| Relevancy: {meta['relevancy_score']}%\n{doc}"
            )
        return "\n\n---\n\n".join(parts)

    def compute_confidence(self, metadatas):
        avg = sum(m["relevancy_score"] for m in metadatas) / len(metadatas)
        distinct = len(set(m["pubmed_id"] for m in metadatas))
        if avg >= 70 and distinct >= 3:   return "HIGH",   avg
        elif avg >= 50 or distinct >= 2:  return "MEDIUM", avg
        else:                             return "LOW",     avg

    def query(self, question: str) -> Dict:
        print(f"Query: {question}")
        documents, metadatas = self.retrieve(question)
        context = self.build_context(documents, metadatas)
        confidence_level, avg_relevancy = self.compute_confidence(metadatas)

        user_prompt = f"""Based on the following PubMed abstracts, answer this question:

QUESTION: {question}

CONTEXT:
{context}

Provide a comprehensive, evidence-based answer with citations."""

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        return {
            "question":        question,
            "answer":          response.choices[0].message.content,
            "sources":         metadatas,
            "confidence":      confidence_level,
            "avg_relevancy":   avg_relevancy,
            "chunks_retrieved": len(documents),
        }


if __name__ == "__main__":
    print("=" * 55)
    print("  MedQuery AI — Verification Test")
    print("=" * 55)
    rag = MedQueryRAG()
    result = rag.query("What are first-line treatments for type 2 diabetes?")
    print(f"\nCONFIDENCE: {result['confidence']} ({result['avg_relevancy']:.1f}%)")
    print(f"ANSWER:\n{result['answer']}")
    print("\n✓ All good! Now run: streamlit run app.py")
