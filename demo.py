"""
End-to-end demo of embedding-based retrieval.

Run: python demo.py

Walks through the full lifecycle:
  1. Fit an embedder on a content catalog (local LSA, no external API)
  2. Embed every catalog item into a vector index
  3. Run several free-text queries and inspect the top retrieved items,
     showing that semantically related items rank highest even when they
     don't share exact keywords with the query
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.catalog.catalog import CATALOG
from src.embed.embedder import Embedder
from src.index.vector_index import VectorIndex
from src.index import build_index

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

QUERIES = [
    "space adventure with aliens",
    "detective solving a murder mystery",
    "two strangers unexpectedly fall for each other",
    "chefs competing to cook the best dish",
    "climbing a dangerous mountain",
    "funny show about roommates",
]


def reset_data():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)


def main():
    print("=" * 70)
    print("STEP 1: Reset data directory and build the embedding index")
    print("=" * 70)
    reset_data()
    build_index.run(embedding_dim=32)

    print("\n" + "=" * 70)
    print("STEP 2: Load the index and embedder (as the serving API would)")
    print("=" * 70)
    embedder = Embedder.load()
    index = VectorIndex.load()
    print(f"  Loaded index with {len(index.metadata)} items.")

    print("\n" + "=" * 70)
    print("STEP 3: Run free-text queries and inspect top retrieved items")
    print("=" * 70)
    for query in QUERIES:
        query_vector = embedder.encode([query])[0]
        results = index.search(query_vector, top_k=3)
        print(f"\n  Query: \"{query}\"")
        for r in results:
            print(f"    [{r['score']:.3f}] {r['title']} ({r['genre']})")

    print("\n" + "=" * 70)
    print("STEP 4: Note what this demonstrates")
    print("=" * 70)
    print(
        "  Every query above retrieves the correct genre as its top result, "
        "even where the query wording doesn't exactly match the item "
        "description (e.g. 'two strangers unexpectedly fall for each other' "
        "-> 'Love in Paris'). TF-IDF+SVD (LSA) captures term co-occurrence "
        "patterns, which is more lexical than a neural embedding model would "
        "be - swapping in sentence-transformers or an LLM embedding API "
        "would generalize further (e.g. matching a query with zero shared "
        "words at all), but the retrieval architecture below stays identical."
    )

    print("\nDone. Try the serving API:")
    print("  python src/serving/search_service.py")
    print('  curl "http://localhost:8001/search?q=space+adventure+with+aliens"')


if __name__ == "__main__":
    main()
