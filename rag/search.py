"""
Loads the FAISS index built by build_index.py and exposes a search function
that agents can call to answer menu questions semantically instead of relying
on the full menu text being stuffed into every prompt.
"""

import logging
import pickle
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.search")

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "menu.index"
STORE_PATH = ROOT / "data" / "menu_store.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_store: dict | None = None
_load_lock = Lock()


def _ensure_loaded() -> None:
    """Lazy-load the model/index/store once, thread-safe."""
    global _model, _index, _store

    if _model is not None and _index is not None and _store is not None:
        return

    with _load_lock:
        if _model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)

        if _index is None:
            if not INDEX_PATH.exists():
                raise FileNotFoundError(
                    f"Menu index not found at {INDEX_PATH}. "
                    "Run `uv run python -m rag.build_index` first."
                )
            logger.info(f"Loading FAISS index from {INDEX_PATH}")
            _index = faiss.read_index(str(INDEX_PATH))

        if _store is None:
            if not STORE_PATH.exists():
                raise FileNotFoundError(
                    f"Menu store not found at {STORE_PATH}. "
                    "Run `uv run python -m rag.build_index` first."
                )
            with open(STORE_PATH, "rb") as f:
                _store = pickle.load(f)


def search_menu(query: str, k: int = 3) -> list[dict]:
    """
    Semantic search over the menu.

    Args:
        query: natural-language question, e.g. "something vegetarian under $6"
        k: number of results to return

    Returns:
        List of menu item dicts (name, category, price, description, tags),
        ordered by relevance.
    """
    _ensure_loaded()
    assert _model is not None and _index is not None and _store is not None

    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    k = min(k, len(_store["items"]))
    if k == 0:
        return []

    scores, indices = _index.search(query_vec, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        item = dict(_store["items"][idx])
        item["relevance_score"] = float(score)
        results.append(item)

    return results