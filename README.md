# VectorMatch

VectorMatch is a small, dependency-light implementation of semantic
(embedding-based) retrieval: given a free-text query, return the catalog items whose meaning
is closest to it, ranked by vector similarity. This is the retrieval
pattern behind recommendations, semantic search, and "more like this"
features on any content platform.

## Why this project

Modern recommendation and search systems separate the problem into two
stages:

1. **Retrieval** - quickly narrow millions of candidate items down to a
   few hundred that are plausibly relevant, using approximate nearest
   neighbor search over embeddings. Speed and recall matter most here.
2. **Ranking** - a heavier model re-scores those few hundred candidates
   with much richer features (user history, freshness, business rules) to
   produce the final ordered list.

VectorMatch implements the retrieval stage end to end: embedding text
into vectors, indexing those vectors, and searching them by similarity.

## Architecture

```
                     ┌─────────────────────┐
   catalog.py        │  21 sample titles     │  (content catalog analog)
                     │  with text descriptions│
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   embedder.py        │  TF-IDF + Truncated SVD (LSA)
                     │   fit() / encode()   │  (neural embedding model analog)
                     └──────────┬──────────┘
                                │ vectors
                                ▼
                     ┌─────────────────────┐
                     │  vector_index.py     │  Brute-force cosine similarity
                     │  build() / search()  │  (FAISS/pgvector analog)
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ search_service.py    │  Flask API
                     │ GET /search?q=...    │  (production retrieval endpoint)
                     └─────────────────────┘
```

## Production mapping

| VectorMatch                         | Production equivalent                              |
|--------------------------------------|-----------------------------------------------------|
| `embedder.py` (TF-IDF + SVD)         | sentence-transformers, OpenAI/Cohere embeddings API, or a two-tower model trained on interaction data |
| `vector_index.py` (brute-force numpy)| FAISS (IVF/HNSW), ScaNN, Pinecone, or pgvector       |
| `catalog.py`                         | Content metadata table / feature store               |
| `search_service.py`                  | Retrieval microservice sitting in front of a ranking model |

The most important thing VectorMatch demonstrates isn't the embedding
quality (TF-IDF+SVD is intentionally simple and runs with no external
downloads) - it's the **interface boundary**: everything downstream of
`encode()` only depends on getting back a fixed-size vector, so swapping in
a much stronger neural embedding model later doesn't touch the index or
the serving code at all.

## Running it

```bash
pip install flask scikit-learn numpy
python demo.py
```

This will:
1. Fit the embedder on a 21-item sample content catalog spanning 6 genres
2. Embed every item and build the vector index
3. Run 6 free-text queries and print the top-3 retrieved items for each
4. Confirm every query retrieves the correct genre as its top result

To try the search API directly:
```bash
python src/serving/search_service.py
curl "http://localhost:8001/search?q=space+adventure+with+aliens"
```

Run the correctness test:
```bash
python tests/test_retrieval.py
```

## What I'd build next with more time

- Swap `embedder.py` for a real neural embedding model
  (`sentence-transformers`, e.g. `all-MiniLM-L6-v2`) to get true semantic
  generalization - matching queries and items that share no vocabulary at
  all, not just co-occurring terms.
- Swap `vector_index.py`'s brute-force search for FAISS with an HNSW or
  IVF index once the catalog is large enough that exact search becomes a
  latency problem, and measure the recall/latency trade-off directly.
- Add a second-stage ranking model on top of retrieval, using features
  like a user's watch history, to demonstrate the full
  retrieval-then-ranking pipeline.
- Add embedding refresh/versioning: re-embed the catalog when the model
  changes, without breaking indexes built on the old embedding space.

## Background

VectorMatch was built as a portfolio project to demonstrate the
embedding-based retrieval pattern used in modern recommendation and
search systems - the same architectural shape used by production systems
built on FAISS, pgvector, or managed vector databases.
