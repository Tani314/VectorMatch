"""
Embedder
=========
Turns text into a fixed-size dense vector ("embedding") such that
semantically similar text ends up close together in vector space.

This uses TF-IDF + Truncated SVD (i.e. classic Latent Semantic Analysis) -
a real, well-established embedding technique that trains entirely locally
on your own corpus, with no external API calls or downloaded model weights.

Substitution note: in production you would swap this for a neural embedding
model - sentence-transformers, OpenAI/Cohere embeddings, or a two-tower
model trained on your own interaction data - behind the exact same
`fit(texts)` / `encode(texts)` interface. Everything downstream (the index,
the search service) only depends on that interface, not on how the vectors
are produced, so swapping in a stronger embedding model later is a one-file
change.
"""

import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "embedder.pkl")


class Embedder:
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        """Learn the vocabulary and semantic dimensions from a corpus.

        Analogous to loading a pretrained embedding model, except here the
        'model' is trained on your own catalog text instead of a giant
        external corpus.
        """
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.embedding_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_components != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed one or more texts into normalized dense vectors.

        Returns an (n_texts, embedding_dim) array. Vectors are L2-normalized
        so that cosine similarity reduces to a simple dot product - the same
        convention most production embedding APIs use.
        """
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before encode().")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return normalize(dense, norm="l2")

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "Embedder":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    corpus = [
        "A crew of explorers travels through a wormhole to find a new home for humanity.",
        "A detective reopens a decades-old murder investigation in the city.",
        "Two strangers meet by chance in Paris and fall in love.",
    ]
    emb = Embedder(embedding_dim=2)
    emb.fit(corpus)
    vectors = emb.encode(corpus)
    print("Embeddings shape:", vectors.shape)
    print(vectors)
