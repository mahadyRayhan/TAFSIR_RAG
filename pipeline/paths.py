"""Filesystem locations shared by the offline build steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
BUILD_DIR = DATA_DIR / "build"

SOURCE_PDF = SOURCE_DIR / "ibn_kathir" / "Tafsir Ibn Kathir all 10 volumes.pdf"

# Step 5: reviewed glossary (tracked in git, unlike data/).
GLOSSARY_CSV = ROOT / "glossary" / "glossary.csv"
SURAH_NAMES_CSV = ROOT / "glossary" / "surah_names.csv"

# Step 1 output: one JSON record per PDF page.
PAGES_JSONL = BUILD_DIR / "pages.jsonl"
# Step 2 output: one JSON record per passage (a verse section or one topic part of it).
PASSAGES_JSONL = BUILD_DIR / "passages.jsonl"
# Step 3 output: one JSON record per Qur'an reference found in a passage.
CITATIONS_JSONL = BUILD_DIR / "citations.jsonl"
# Step 4 output: SQLite database with the passages, FTS5 keyword index and citation graph.
DB_PATH = BUILD_DIR / "tafsir.db"
