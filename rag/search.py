"""
Loads the FAISS index built by build_index.py and exposes search functions
that agents can call to answer questions about the menu, FAQ, policies,
and allergens semantically instead of relying on text being stuffed into
every prompt.
"""

import json
import logging
import pickle
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.search")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Paths for each index
INDEX_PATHS = {
    "menu": DATA_DIR / "menu.index",
    "faq": DATA_DIR / "faq.index",
    "policies": DATA_DIR / "policies.index",
}
STORE_PATHS = {
    "menu": DATA_DIR / "menu_store.pkl",
    "faq": DATA_DIR / "faq_store.pkl",
    "policies": DATA_DIR / "policies_store.pkl",
}

# Static data files
ALLERGENS_PATH = DATA_DIR / "allergens.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_indexes: dict[str, faiss.Index] = {}
_stores: dict[str, dict] = {}
_allergens: dict[str, list[str]] = {}
_load_lock = Lock()


def _ensure_loaded() -> None:
    """Lazy-load the model/indexes/stores once, thread-safe."""
    global _model, _indexes, _stores, _allergens

    if _model is not None and _indexes and _stores and _allergens:
        return

    with _load_lock:
        if _model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)

        # Load each index + store
        for name in ("menu", "faq", "policies"):
            if name not in _indexes:
                index_path = INDEX_PATHS[name]
                store_path = STORE_PATHS[name]

                if not index_path.exists():
                    logger.warning(f"Index not found: {index_path}. Skipping '{name}' search.")
                    continue
                if not store_path.exists():
                    logger.warning(f"Store not found: {store_path}. Skipping '{name}' search.")
                    continue

                logger.info(f"Loading FAISS index: {index_path}")
                _indexes[name] = faiss.read_index(str(index_path))

                with open(store_path, "rb") as f:
                    _stores[name] = pickle.load(f)

        # Load allergen data (static JSON, no index needed)
        if not _allergens:
            if ALLERGENS_PATH.exists():
                with open(ALLERGENS_PATH, "r") as f:
                    _allergens = json.load(f)
                logger.info(f"Loaded allergen data: {len(_allergens)} items")
            else:
                logger.warning(f"Allergens file not found: {ALLERGENS_PATH}")
                _allergens = {}


def _search_index(index_name: str, query: str, k: int = 3) -> list[dict]:
    """Search a single FAISS index and return results."""
    _ensure_loaded()

    if index_name not in _indexes or index_name not in _stores:
        return []

    index = _indexes[index_name]
    store = _stores[index_name]
    assert _model is not None

    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    k = min(k, len(store["items"]))
    if k == 0:
        return []

    scores, indices = index.search(query_vec, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        item = dict(store["items"][idx])
        item["relevance_score"] = float(score)
        results.append(item)

    return results


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
    return _search_index("menu", query, k)


def search_faq(query: str, k: int = 3) -> list[dict]:
    """
    Semantic search over FAQ entries.

    Returns:
        List of FAQ dicts (question, answer), ordered by relevance.
    """
    return _search_index("faq", query, k)


def search_policies(query: str, k: int = 3) -> list[dict]:
    """
    Semantic search over restaurant policies.

    Returns:
        List of policy dicts (topic, content), ordered by relevance.
    """
    return _search_index("policies", query, k)


def search_all(query: str, k: int = 3) -> list[dict]:
    """
    Search across all knowledge sources (menu, FAQ, policies).
    Merges and sorts results by relevance score.

    Returns:
        List of dicts with 'source', 'content', and 'relevance_score'.
    """
    results = []

    # Search menu
    menu_results = search_menu(query, k)
    for r in menu_results:
        results.append({
            "source": "menu",
            "content": f"{r['name']} (${r['price']:.2f}) — {r['description']}",
            "relevance_score": r.get("relevance_score", 0),
        })

    # Search FAQ
    faq_results = search_faq(query, k)
    for r in faq_results:
        results.append({
            "source": "faq",
            "content": f"Q: {r.get('question', '')}\nA: {r.get('answer', r.get('content', ''))}",
            "relevance_score": r.get("relevance_score", 0),
        })

    # Search policies
    policy_results = search_policies(query, k)
    for r in policy_results:
        results.append({
            "source": "policies",
            "content": f"{r.get('topic', '')}: {r.get('content', '')}",
            "relevance_score": r.get("relevance_score", 0),
        })

    # Sort by relevance score (descending)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return results[:k * 2]  # return top results


def search_allergens(item_name: str) -> str:
    """
    Look up allergen information for a specific menu item.

    Args:
        item_name: name of the menu item, e.g. "Pizza"

    Returns:
        Human-readable allergen info string.
    """
    _ensure_loaded()

    # Try exact match first
    if item_name in _allergens:
        allergens = _allergens[item_name]
        if not allergens or allergens == ["none"]:
            return f"{item_name} has no known allergens."
        return f"{item_name} contains: {', '.join(allergens)}."

    # Try case-insensitive match
    for key, allergens in _allergens.items():
        if key.lower() == item_name.lower():
            if not allergens or allergens == ["none"]:
                return f"{key} has no known allergens."
            return f"{key} contains: {', '.join(allergens)}."

    return f"Sorry, I don't have allergen information for '{item_name}'."