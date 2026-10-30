import json
import sqlite3

import pytest

from pipeline import paths
from pipeline.extract_pages import page_text
from pipeline.index import build, connect, fts_query, search, search_text


@pytest.mark.parametrize("raw, expected", [
    ("Mu`adh said", "Muadh said"),
    ("Ibn `Abbas and the Qur'an", "Ibn Abbas and the Quran"),
    ("Surat Al-Ma'idah", "Surat Al-Maidah"),
    ("`prayer' here", "prayer' here"),
])
def test_search_text(raw, expected):
    assert search_text(raw) == expected


def test_fts_query_quotes_words_and_neutralises_operators():
    assert fts_query('Iblis AND NEAR( "Adam"') == '"Iblis" OR "AND" OR "NEAR" OR "Adam"'
    assert fts_query("Mu`adh") == '"Muadh"'


def passage(pid, text, headings=(), surah=2, a=1, b=1):
    return {"id": pid, "section_id": pid.split("#")[0], "kind": "verses", "surah": surah,
            "ayah_start": a, "ayah_end": b, "part": 1, "parts": 1,
            "title": headings[0] if headings else None, "headings": list(headings),
            "marked_verses": [a], "page_start": 1, "page_end": 1, "text": text}


@pytest.fixture
def mini_db(tmp_path):
    ps = [
        passage("2:34", "The angels prostrated before Adam, except Iblis.", ["Honoring Adam"], a=34, b=34),
        passage("2:35", "Adam and his wife dwelt in Paradise. Adam.", a=35, b=35),
        passage("7:11", "Prostration of the angels; Shaytan's arrogance.", ["Prostration of the Angels"], 7, 11, 11),
        passage("2:2", "This is the Book wherein is no doubt, guidance for those who have Taqwā.", a=2, b=2),
    ]
    cites = [{"passage_id": "2:34", "section_id": "2:34", "surah": 18, "ayah_start": 50, "ayah_end": 50,
              "raw": "18:50", "targets": ["18:50"], "self": False},
             {"passage_id": "2:35", "section_id": "2:35", "surah": 2, "ayah_start": 35, "ayah_end": 35,
              "raw": "2:35", "targets": ["2:35"], "self": True}]
    out = tmp_path / "t.db"
    build(ps, cites, out)
    return connect(out)


def ids(rows):
    return [r["id"] for r in rows]


def test_stemming_and_diacritics(mini_db):
    assert set(ids(search(mini_db, fts_query("prostrate")))) == {"2:34", "7:11"}
    assert ids(search(mini_db, fts_query("taqwa"))) == ["2:2"]


def test_headings_outrank_body_mentions(mini_db):
    # "Adam" appears twice in 2:35's body but in 2:34's heading
    assert ids(search(mini_db, fts_query("Adam")))[0] == "2:34"


def test_boolean_queries(mini_db):
    assert ids(search(mini_db, '"Adam" AND "Iblis"')) == ["2:34"]


def test_citations_table(mini_db):
    rows = mini_db.execute("SELECT section_id, target_section_id, is_self FROM citations ORDER BY 1").fetchall()
    assert [tuple(r) for r in rows] == [("2:34", "18:50", 0), ("2:35", "2:35", 1)]


def test_database_is_read_only(mini_db):
    with pytest.raises(sqlite3.OperationalError):
        mini_db.execute("DELETE FROM passages")


# --- full build output (run `python -m pipeline.index` first) ---------------------------------

@pytest.fixture(scope="module")
def db():
    if not paths.DB_PATH.exists():
        pytest.skip("data/build/tafsir.db not built")
    return connect()


@pytest.mark.data
def test_counts_match_build_outputs(db):
    with paths.PASSAGES_JSONL.open() as f:
        n = sum(1 for _ in f)
    assert db.execute("SELECT count(*) FROM passages").fetchone()[0] == n
    assert db.execute("SELECT count(*) FROM passages_fts").fetchone()[0] == n
    assert db.execute("SELECT count(*) FROM citations").fetchone()[0] > 4500


@pytest.mark.data
def test_every_page_mentioning_iblis_is_found(db):
    with paths.PAGES_JSONL.open() as f:
        pages = {r["page"] for r in map(json.loads, f) if "Iblis" in page_text(r)}
    covered = set()
    for r in search(db, fts_query("Iblis"), limit=10_000):
        covered.update(range(r["page_start"], r["page_end"] + 1))
    assert len(pages) > 60 and pages <= covered


@pytest.mark.data
def test_adam_and_iblis_found_across_surahs(db):
    top = search(db, '"Adam" AND "Iblis"', limit=15)
    surahs = {r["surah"] for r in top}
    assert {2, 7, 15, 17, 18, 20, 38} <= surahs
