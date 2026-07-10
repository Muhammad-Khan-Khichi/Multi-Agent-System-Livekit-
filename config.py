import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MENU_PATH = ROOT / "data" / "menu.json"

# Load menu from JSON file
with open(MENU_PATH, "r", encoding="utf-8-sig") as f:
    MENU_ITEMS = json.load(f)

# Keep a simple string version for backward compatibility (agents that expect a string)
MENU = ", ".join(f"{item['name']}: ${item['price']:.2f}" for item in MENU_ITEMS)

VOICES = {
    "greeter": "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    "reservation": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    "takeaway": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
    "checkout": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
    "faq": "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
}