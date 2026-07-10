from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.search")

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Globals (lazy-loaded)
_embedder: SentenceTransformer | None = None
_indexes: dict[str, faiss.Index] = {}
_metadata: dict[str, list[dict]] = {}
_allergens: dict[str, dict] = {}


def _ensure_loaded() -> None:
    """Load the embedding model, FAISS indexes, metadata, and allergen data
    on first use. Safe against missing or corrupted files."""
    global _embedder, _indexes, _metadata, _allergens

    if _embedder is not None:
        return

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    _embedder = SentenceTransformer(EMBED_MODEL)

    # ── Load FAISS indexes + metadata from .pkl files ──
    for name in ["menu", "faq", "policies"]:
        index_path = DATA_DIR / f"{name}.index"
        store_path = DATA_DIR / f"{name}_store.pkl"

        if not index_path.exists():
            logger.warning(
                f"Index not found for '{name}', skipping. "
                f"Run `uv run python -m rag.build_index` to build indexes."
            )
            continue

        if not store_path.exists():
            logger.warning(
                f"Store not found for '{name}', skipping. "
                f"Run `uv run python -m rag.build_index` to build indexes."
            )
            continue

        # Load FAISS index
        logger.info(f"Loading FAISS index: {index_path}")
        _indexes[name] = faiss.read_index(str(index_path))

        # Load metadata from .pkl
        logger.info(f"Loading metadata from: {store_path}")
        try:
            with open(store_path, "rb") as f:
                store = pickle.load(f)

            # The .pkl structure is: {"items": [...], "texts": [...]}
            items = store.get("items", [])
            texts = store.get("texts", [])

            # Build metadata list with a "content" field for each entry
            metadata_list = []
            for i, item in enumerate(items):
                entry = dict(item)  # copy the original item

                # Use the pre-built text if available, otherwise build content
                if i < len(texts):
                    entry["content"] = texts[i]
                elif "content" in entry:
                    pass  # already has content
                elif "answer" in entry and "question" in entry:
                    entry["content"] = f"{entry['question']} {entry['answer']}"
                elif "answer" in entry:
                    entry["content"] = entry["answer"]
                elif "description" in entry:
                    entry["content"] = entry["description"]
                else:
                    entry["content"] = str(entry)

                metadata_list.append(entry)

            _metadata[name] = metadata_list
            logger.info(f"Loaded {len(_metadata[name])} entries for '{name}'")

        except Exception as e:
            logger.warning(f"Failed to load store for '{name}': {e}. Skipping.")
            _metadata[name] = []

    # ── Safe allergen loading ──
    allergen_path = DATA_DIR / "allergens.json"
    if allergen_path.exists():
        try:
            with open(allergen_path, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                if content:
                    _allergens = json.loads(content)
                    logger.info(f"Loaded allergen data: {len(_allergens)} items")
                else:
                    logger.warning("allergens.json is empty, using empty dict.")
                    _allergens = {}
        except json.JSONDecodeError as e:
            logger.warning(f"allergens.json has invalid JSON: {e}. Using empty dict.")
            _allergens = {}
    else:
        logger.warning("allergens.json not found, using empty dict.")
        _allergens = {}


def _search_index(index_name: str, query: str, k: int = 3) -> list[dict]:
    """Search a specific FAISS index and return matching results.

    Uses inner-product (cosine similarity) — HIGHER score = BETTER match.

    Returns a list of dicts, each with:
        - source: index name
        - content: the matched text
        - score: similarity score (higher = better)
        - rank: result rank
        - metadata: the full original entry
    """
    _ensure_loaded()

    if index_name not in _indexes:
        logger.warning(f"Index '{index_name}' not loaded.")
        return []

    if index_name not in _metadata:
        logger.warning(f"Metadata for '{index_name}' not loaded.")
        return []

    index = _indexes[index_name]
    metadata = _metadata[index_name]

    if index.ntotal == 0:
        logger.warning(f"Index '{index_name}' is empty.")
        return []

    if not metadata:
        logger.warning(f"Metadata for '{index_name}' is empty.")
        return []

    # Limit k to available entries
    k = min(k, index.ntotal, len(metadata))

    # Embed the query
    query_vec = _embedder.encode([query], convert_to_numpy=True)
    query_vec = query_vec.astype("float32")
    faiss.normalize_L2(query_vec)

    # Search (inner product = cosine similarity since vectors are normalized)
    scores, indices = index.search(query_vec, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue

        if idx >= len(metadata):
            logger.warning(
                f"Index {idx} out of range for metadata "
                f"(len={len(metadata)}), skipping."
            )
            continue

        entry = metadata[idx]
        results.append(
            {
                "source": index_name,
                "content": entry.get("content", ""),
                "score": float(score),
                "rank": rank,
                "metadata": entry,
            }
        )

    return results


def search_menu(query: str, k: int = 3) -> list[dict]:
    """Search the menu index for items matching the query."""
    return _search_index("menu", query, k)


def search_faq(query: str, k: int = 3) -> list[dict]:
    """Search the FAQ index for answers matching the query."""
    return _search_index("faq", query, k)


def search_policies(query: str, k: int = 3) -> list[dict]:
    """Search the policies index for policy information."""
    return _search_index("policies", query, k)


def search_allergens(item_name: str) -> str:
    """Look up allergen information for a menu item by name.

    Returns a human-readable string describing allergens.
    """
    _ensure_loaded()

    # Normalize the query
    key = item_name.lower().strip()

    # Try exact match first
    if key in _allergens:
        info = _allergens[key]
        return _format_allergen_info(item_name, info)

    # Try partial match
    for allergen_key, info in _allergens.items():
        if allergen_key in key or key in allergen_key:
            return _format_allergen_info(allergen_key, info)

    return f"No allergen information found for '{item_name}'."


def _format_allergen_info(name: str, info: dict) -> str:
    """Format allergen info dict into a readable string."""
    contains = info.get("contains", [])
    may_contain = info.get("may_contain", [])

    parts = [f"{name.title()}"]

    if contains:
        parts.append(f"contains: {', '.join(contains)}")
    else:
        parts.append("contains: no known allergens")

    if may_contain:
        parts.append(f"may contain: {', '.join(may_contain)}")

    return " | ".join(parts)


def search_all(query: str, k: int = 3) -> list[dict]:
    """Search across all indexes (menu, FAQ, policies) in one call.

    Returns combined results sorted by score (highest = best match).
    """
    _ensure_loaded()

    all_results = []

    for index_name in ["menu", "faq", "policies"]:
        if index_name in _indexes:
            results = _search_index(index_name, query, k)
            all_results.extend(results)

    # Sort by score (HIGHER = better, since we use cosine similarity)
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Return top k across all sources
    return all_results[:k]


def get_all_menu_items() -> list[dict]:
    """Return all menu items from the metadata.

    Useful for displaying the full menu to the user.
    """
    _ensure_loaded()

    if "menu" not in _metadata:
        return []

    return _metadata["menu"]