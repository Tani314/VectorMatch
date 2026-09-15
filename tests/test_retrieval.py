"""
Regression test for the core retrieval guarantee: a query should rank
genre-relevant items above unrelated items.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.catalog.catalog import CATALOG
from src.embed.embedder import Embedder
from src.index.vector_index import VectorIndex


def test_retrieval_ranks_relevant_items_first():
    descriptions = [item["description"] for item in CATALOG]

    embedder = Embedder(embedding_dim=32)
    embedder.fit(descriptions)
    vectors = embedder.encode(descriptions)

    index = VectorIndex()
    index.build(vectors, CATALOG)

    cases = [
        ("space adventure with aliens", "sci-fi"),
        ("detective solving a murder mystery", "crime"),
        ("chefs competing to cook the best dish", "cooking"),
        ("climbing a dangerous mountain", "documentary"),
        ("funny show about roommates", "comedy"),
    ]

    for query, expected_genre in cases:
        query_vector = embedder.encode([query])[0]
        top_result = index.search(query_vector, top_k=1)[0]
        assert top_result["genre"] == expected_genre, (
            f"Query '{query}' expected top genre '{expected_genre}' "
            f"but got '{top_result['genre']}' ({top_result['title']})"
        )

    print(f"PASSED: all {len(cases)} queries retrieved the correct genre as top result.")


if __name__ == "__main__":
    test_retrieval_ranks_relevant_items_first()
