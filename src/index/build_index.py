"""
Build Index
============
Offline job: fit the embedder on the catalog, embed every item, and persist
the resulting index to disk. Analogous to a nightly/periodic batch job that
re-embeds a content catalog as new items are added.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.catalog.catalog import CATALOG
from src.embed.embedder import Embedder
from src.index.vector_index import VectorIndex


def run(embedding_dim: int = 32) -> None:
    descriptions = [item["description"] for item in CATALOG]

    embedder = Embedder(embedding_dim=embedding_dim)
    embedder.fit(descriptions)
    vectors = embedder.encode(descriptions)

    index = VectorIndex()
    index.build(vectors, CATALOG)

    embedder.save()
    index.save()

    print(f"Embedded {len(CATALOG)} catalog items into {vectors.shape[1]}-dim vectors.")
    print("Embedder saved to data/embedder.pkl")
    print("Index saved to data/vectors.npy + data/metadata.json")


if __name__ == "__main__":
    run()
