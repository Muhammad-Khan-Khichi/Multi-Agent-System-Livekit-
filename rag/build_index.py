"""
Builds FAISS indexes for menu, FAQ, and policies so agents can semantically
search across all restaurant knowledge.

Run this whenever data files change:
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
DATA_DIR = ROOT / "data"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─── Menu ───────────────────────────────────────────────────────────────

def _item_to_text(item: dict) -> str:
    """Flatten a menu item into a single string for embedding."""
    tags = ", ".join(item.get("tags", [])) or "none"
    return (
        f"{item['name']} ({item['category']}, ${item['price']:.2f}): "
        f"{item['description']} Dietary tags: {tags}."
    )


def build_menu_index(model: SentenceTransformer) -> None:
    """Build FAISS index from menu.json."""
    menu_path = DATA_DIR / "menu.json"
    index_path = DATA_DIR / "menu.index"
    store_path = DATA_DIR / "menu_store.pkl"

    if not menu_path.exists():
        raise FileNotFoundError(f"Menu data not found at {menu_path}")

    with open(menu_path, "r", encoding="utf-8-sig") as f:
        menu_items = json.load(f)

    logger.info(f"Loaded {len(menu_items)} menu items from {menu_path}")

    texts = [_item_to_text(item) for item in menu_items]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(store_path, "wb") as f:
        pickle.dump({"items": menu_items, "texts": texts}, f)

    logger.info(f"✅ Menu index built: {index_path} ({len(menu_items)} items)")


# ─── FAQ ────────────────────────────────────────────────────────────────

def build_faq_index(model: SentenceTransformer) -> None:
    """Build FAISS index from faq.md."""
    faq_path = DATA_DIR / "faq.md"
    index_path = DATA_DIR / "faq.index"
    store_path = DATA_DIR / "faq_store.pkl"

    if not faq_path.exists():
        logger.warning(f"⚠️  FAQ file not found: {faq_path}. Skipping FAQ index.")
        return

    content = faq_path.read_text(encoding="utf-8")
    sections = content.split("## ")[1:]  # Split by ## headers

    items = []
    texts = []
    for section in sections:
        lines = section.strip().split("\n", 1)
        if len(lines) < 2:
            continue
        question = lines[0].strip()
        answer = lines[1].strip()
        items.append({"question": question, "answer": answer})
        texts.append(f"{question} {answer}")

    if not items:
        logger.warning("⚠️  No FAQ entries found. Skipping FAQ index.")
        return

    logger.info(f"Loaded {len(items)} FAQ entries from {faq_path}")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(store_path, "wb") as f:
        pickle.dump({"items": items, "texts": texts}, f)

    logger.info(f"✅ FAQ index built: {index_path} ({len(items)} entries)")


# ─── Policies ───────────────────────────────────────────────────────────

def build_policies_index(model: SentenceTransformer) -> None:
    """Build FAISS index from policies.md."""
    policies_path = DATA_DIR / "policies.md"
    index_path = DATA_DIR / "policies.index"
    store_path = DATA_DIR / "policies_store.pkl"

    if not policies_path.exists():
        logger.warning(f"⚠️  Policies file not found: {policies_path}. Skipping policies index.")
        return

    content = policies_path.read_text(encoding="utf-8")
    sections = content.split("## ")[1:]

    items = []
    texts = []
    for section in sections:
        lines = section.strip().split("\n", 1)
        if len(lines) < 2:
            continue
        topic = lines[0].strip()
        policy_text = lines[1].strip()
        items.append({"topic": topic, "content": policy_text})
        texts.append(f"{topic} {policy_text}")

    if not items:
        logger.warning("⚠️  No policy entries found. Skipping policies index.")
        return

    logger.info(f"Loaded {len(items)} policy entries from {policies_path}")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(store_path, "wb") as f:
        pickle.dump({"items": items, "texts": texts}, f)

    logger.info(f"✅ Policies index built: {index_path} ({len(items)} entries)")


# ─── Main ───────────────────────────────────────────────────────────────

def build_all() -> None:
    """Build all indexes."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    build_menu_index(model)
    build_faq_index(model)
    build_policies_index(model)

    logger.info("🎉 All indexes built successfully!")


if __name__ == "__main__":
    build_all()