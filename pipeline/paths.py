"""Filesystem locations shared by the offline build steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
BUILD_DIR = DATA_DIR / "build"

SOURCE_PDF = SOURCE_DIR / "ibn_kathir" / "Tafsir Ibn Kathir all 10 volumes.pdf"

# Step 1 output: one JSON record per PDF page.
PAGES_JSONL = BUILD_DIR / "pages.jsonl"
# Step 2 output: one JSON record per passage (a verse section or one topic part of it).
PASSAGES_JSONL = BUILD_DIR / "passages.jsonl"
