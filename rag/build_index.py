"""
Builds a FAISS index over the menu items so agents can semantically search
for dishes (e.g. "something vegetarian under $6" or "what desserts do you have").

Run this once whenever data/menu.json changes:
    uv run python -m rag.build_index
"""

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.build_index")
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
MENU_PATH = ROOT / "data" / "menu.json"
INDEX_PATH = ROOT / "data" / "menu.index"
STORE_PATH = ROOT / "data" / "menu_store.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _item_to_text(item: dict) -> str:
    """Flatten a menu item into a single string for embedding."""
    tags = ", ".join(item.get("tags", [])) or "none"
    return (
        f"{item['name']} ({item['category']}, ${item['price']:.2f}): "
        f"{item['description']} Dietary tags: {tags}."
    )


def build_index() -> None:
    if not MENU_PATH.exists():
        raise FileNotFoundError(f"Menu data not found at {MENU_PATH}")

    with open(MENU_PATH, "r", encoding="utf-8-sig") as f:
        menu_items = json.load(f)

    logger.info(f"Loaded {len(menu_items)} menu items from {MENU_PATH}")

    texts = [_item_to_text(item) for item in menu_items]

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Encoding menu items...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(STORE_PATH, "wb") as f:
        pickle.dump({"items": menu_items, "texts": texts}, f)

    logger.info(f"✅ Index built: {INDEX_PATH}")
    logger.info(f"✅ Store saved: {STORE_PATH}")


if __name__ == "__main__":
    build_index()