"""Step 4: build the SQLite search database (keyword index + citation graph).

One file, data/build/tafsir.db, with:
  passages      every passage from step 2 (display text, verse range, pages, headings)
  passages_fts  FTS5 keyword index over title + text, BM25-ranked
  citations     one row per (reference, target section) from step 3

The index uses the Porter stemmer with diacritics removed, so "prostration" matches
"prostrate" and "Taqwā" matches "taqwa". Ayn and apostrophes are removed before indexing
("Mu`adh" -> "Muadh", "Qur'an" -> "Quran"); hyphenated names split into words
("Al-Baqarah" matches "Baqarah"). Building it costs nothing and takes seconds.

Usage:
    python -m pipeline.index [--passages PATH] [--citations PATH] [--out PATH]
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

from pipeline import paths
from pipeline.citations import load_passages

TOKENIZER = "porter unicode61 remove_diacritics 2"
TITLE_WEIGHT = 4.0  # bm25 column weight of headings relative to body text

_INWORD_MARKS_RE = re.compile(r"(?<=\w)[`'‘’]+(?=\w)|(?<=\s)[`]+(?=\w)|^[`]+(?=\w)")

SCHEMA = f"""
CREATE TABLE passages (
    rowid INTEGER PRIMARY KEY,
    id TEXT UNIQUE NOT NULL,
    section_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    surah INTEGER NOT NULL,
    ayah_start INTEGER NOT NULL,
    ayah_end INTEGER NOT NULL,
    part INTEGER NOT NULL,
    parts INTEGER NOT NULL,
    title TEXT,
    headings TEXT NOT NULL,       -- JSON list
    marked_verses TEXT NOT NULL,  -- JSON list
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX passages_section ON passages(section_id);
CREATE INDEX passages_verse ON passages(surah, ayah_start, ayah_end);

CREATE VIRTUAL TABLE passages_fts USING fts5(title, text, tokenize = '{TOKENIZER}');

CREATE TABLE citations (
    passage_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    surah INTEGER NOT NULL,
    ayah_start INTEGER NOT NULL,
    ayah_end INTEGER NOT NULL,
    raw TEXT NOT NULL,
    target_section_id TEXT NOT NULL,
    is_self INTEGER NOT NULL
);
CREATE INDEX citations_source ON citations(section_id);
CREATE INDEX citations_target ON citations(target_section_id);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def search_text(text: str) -> str:
    """Text as indexed: ayn marks and apostrophes inside words removed."""
    return _INWORD_MARKS_RE.sub("", text or "")


def build(passages: List[dict], citation_rows: List[dict], out: Path) -> None:
    tmp = out.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SCHEMA)
        for rowid, p in enumerate(passages, 1):
            conn.execute(
                "INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rowid, p["id"], p["section_id"], p["kind"], p["surah"], p["ayah_start"],
                 p["ayah_end"], p["part"], p["parts"], p["title"], json.dumps(p["headings"]),
                 json.dumps(p["marked_verses"]), p["page_start"], p["page_end"], p["text"]))
            conn.execute("INSERT INTO passages_fts(rowid, title, text) VALUES (?,?,?)",
                         (rowid, search_text(" ".join(p["headings"])), search_text(p["text"])))
        conn.executemany(
            "INSERT INTO citations VALUES (?,?,?,?,?,?,?,?)",
            [(r["passage_id"], r["section_id"], r["surah"], r["ayah_start"], r["ayah_end"],
              r["raw"], t, int(t == r["section_id"])) for r in citation_rows for t in r["targets"]])
        conn.executemany("INSERT INTO meta VALUES (?,?)", [
            ("tokenizer", TOKENIZER),
            ("passages", str(len(passages))),
            ("citations", str(len(citation_rows))),
        ])
        conn.execute("INSERT INTO passages_fts(passages_fts) VALUES ('optimize')")
        conn.commit()
    finally:
        conn.close()
    tmp.replace(out)


def fts_query(text: str) -> str:
    """Turn free text into an FTS5 query that ORs its words (each word quoted, so
    punctuation and FTS operators in user input are harmless)."""
    words = re.findall(r"\w+", search_text(text))
    return " OR ".join(f'"{w}"' for w in words)


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> List[sqlite3.Row]:
    """BM25 keyword search over passages; `query` is an FTS5 expression (see fts_query)."""
    conn.row_factory = sqlite3.Row
    return conn.execute(
        f"""SELECT p.*, bm25(passages_fts, {TITLE_WEIGHT}, 1.0) AS score
            FROM passages_fts JOIN passages p ON p.rowid = passages_fts.rowid
            WHERE passages_fts MATCH ? ORDER BY score LIMIT ?""",
        (query, limit)).fetchall()


def connect(path: Path = paths.DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--passages", type=Path, default=paths.PASSAGES_JSONL)
    parser.add_argument("--citations", type=Path, default=paths.CITATIONS_JSONL)
    parser.add_argument("--out", type=Path, default=paths.DB_PATH)
    args = parser.parse_args(argv)

    passages = load_passages(args.passages)
    citation_rows = load_passages(args.citations)
    build(passages, citation_rows, args.out)

    conn = connect(args.out)
    n_terms = conn.execute("SELECT count(*) FROM passages").fetchone()[0]
    n_cit = conn.execute("SELECT count(*) FROM citations").fetchone()[0]
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB): "
          f"{n_terms} passages, {n_cit} citation edges")


if __name__ == "__main__":
    main()
