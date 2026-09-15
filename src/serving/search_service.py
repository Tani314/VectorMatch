"""
Search Serving API
====================
Exposes semantic search over the catalog: given a free-text query, embed it
with the same embedder used at index-build time, then retrieve the closest
items from the vector index.

This is the retrieval half of a recommendation/search system - what would
sit behind a "search" box or a "more like this" feature, typically followed
by a separate ranking model that re-scores the retrieved candidates.

Run with: python src/serving/search_service.py
Then:     curl "http://localhost:8001/search?q=space+adventure+with+aliens"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from flask import Flask, jsonify, request
from src.embed.embedder import Embedder
from src.index.vector_index import VectorIndex

app = Flask(__name__)

_embedder = None
_index = None


def _load():
    global _embedder, _index
    if _embedder is None:
        _embedder = Embedder.load()
    if _index is None:
        _index = VectorIndex.load()


@app.route("/search", methods=["GET"])
def search():
    _load()
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 5))
    if not query:
        return jsonify({"error": "Missing required query param 'q'"}), 400

    query_vector = _embedder.encode([query])[0]
    results = _index.search(query_vector, top_k=top_k)
    return jsonify({"query": query, "results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    _load()
    app.run(host="0.0.0.0", port=8001)
