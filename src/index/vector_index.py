"""
Vector Index
=============
Stores embeddings alongside their item metadata and answers "which items
are closest to this query vector" - the core operation behind
embedding-based retrieval (recommendations, semantic search, "more like
this").

This implementation is exact brute-force search (cosine similarity against
every stored vector via a single matrix multiply), which is entirely
adequate up to roughly hundreds of thousands of vectors on one machine.

Substitution note: at production scale (millions-billions of vectors,
sub-10ms latency requirements) you would swap this for an approximate
nearest neighbor (ANN) index - FAISS (IVF/HNSW), ScaNN, or a managed vector
database like Pinecone or pgvector - behind the same `add()` / `search()`
interface. The accuracy/latency trade-off ANN indexes make (approximate
instead of exact nearest neighbors, in exchange for sub-linear search time)
is the central design decision in any real retrieval system like VectorMatch.
"""

import json
import os

import numpy as np

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
VECTORS_PATH = os.path.join(INDEX_DIR, "vectors.npy")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")


class VectorIndex:
    def __init__(self):
        self.vectors: np.ndarray | None = None   # shape: (n_items, dim)
        self.metadata: list[dict] = []             # parallel list of item metadata

    def build(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        assert vectors.shape[0] == len(metadata), "vectors and metadata must be the same length"
        self.vectors = vectors
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the top_k closest items to query_vector by cosine similarity.

        Since both query and stored vectors are L2-normalized, cosine
        similarity is just the dot product - this is the exact operation a
        real ANN index approximates at scale.
        """
        if self.vectors is None:
            raise RuntimeError("Index has not been built yet.")
        scores = self.vectors @ query_vector  # (n_items,)
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {**self.metadata[i], "score": round(float(scores[i]), 4)}
            for i in top_idx
        ]

    def save(self, vectors_path: str = VECTORS_PATH, metadata_path: str = METADATA_PATH) -> None:
        os.makedirs(os.path.dirname(vectors_path), exist_ok=True)
        np.save(vectors_path, self.vectors)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @staticmethod
    def load(vectors_path: str = VECTORS_PATH, metadata_path: str = METADATA_PATH) -> "VectorIndex":
        idx = VectorIndex()
        idx.vectors = np.load(vectors_path)
        with open(metadata_path) as f:
            idx.metadata = json.load(f)
        return idx


if __name__ == "__main__":
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    idx = VectorIndex()
    idx.build(vectors, metadata)
    print(idx.search(np.array([1.0, 0.0]), top_k=2))
