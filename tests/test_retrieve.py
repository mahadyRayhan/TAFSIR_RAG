import time

import pytest

from pipeline import paths
from pipeline.index import build, connect
from pipeline.surahs import surah_key
from rag.query import parse_query
from rag.retrieve import (
    citation_list,
    concept_share,
    coverage_list,
    fuse,
    neighbour_list,
    retrieve,
    surah_list,
    verse_list,
)

NAMES = {surah_key("Al-Baqarah"): 2, surah_key("Al-A`raf"): 7, surah_key("Yusuf"): 12, surah_key("Al-Kahf"): 18}
GLOSSARY = [{"iblis", "shaytan", "satan"}]


def p(pid, surah, a, b, text, title=None, kind="verses", part=1, parts=1):
    return {"id": pid, "section_id": pid.split("#")[0], "kind": kind, "surah": surah, "ayah_start": a,
            "ayah_end": b, "part": part, "parts": parts, "title": title,
            "headings": [title] if title else [], "marked_verses": [a] if a else [],
            "page_start": 100 + surah, "page_end": 100 + surah, "text": f"{text} Allah."}


PASSAGES = [
    p("2:0", 2, 0, 0, "Virtues of Surat Al-Baqarah.", "Surat Al-Baqarah", kind="intro"),
    p("2:30", 2, 30, 30, "(30. Your Lord said to the angels.)\n\nAdam was made a Khalifah on earth.",
      "Adam inhabited the Earth"),
    p("2:31-33", 2, 31, 33, "(31. He taught Adam all the names.)\n\nAdam's knowledge; Iblis saw it."),
    p("2:34#1", 2, 34, 34, "(34. The angels prostrated before Adam, except Iblis.)\n\nHonor for Adam.",
      "Honoring Adam", part=1, parts=2),
    p("2:34#2", 2, 34, 34, "The obedience was to Allah; Iblis refused to prostrate to Adam.", part=2, parts=2),
    p("2:35-36", 2, 35, 36, "(35. Dwell in Paradise.)\n\nAdam and his wife; Shaytan made them slip."),
    p("2:37", 2, 37, 37, "(37. Adam received Words.)\n\nAdam repents."),
    p("2:38-39", 2, 38, 39, "(38. Get down.)\n\nGuidance will come."),
    p("7:11", 7, 11, 11, "(11. We created you.)\n\nProstration of the angels to Adam and Shaytan's arrogance."),
    p("12:4", 12, 4, 4, "(4. Yusuf said to his father.)\n\nThe dream of Yusuf."),
    p("12:5", 12, 5, 5, "(5. Do not tell your brothers.)\n\nThe brothers of Yusuf."),
    p("18:50", 18, 50, 50, "(50. Except Iblis.)\n\nIblis was one of the Jinn and disobeyed; Adam honored."),
    p("18:51", 18, 51, 51, "(51. I did not make them witnesses.)\n\nNo partners."),
]
CITATIONS = [
    {"passage_id": "2:34#2", "section_id": "2:34", "surah": 18, "ayah_start": 50, "ayah_end": 50,
     "raw": "18:50", "targets": ["18:50"], "self": False},
    {"passage_id": "2:34#2", "section_id": "2:34", "surah": 18, "ayah_start": 51, "ayah_end": 51,
     "raw": "18:51", "targets": ["18:51"], "self": False},
]


@pytest.fixture
def conn(tmp_path):
    out = tmp_path / "t.db"
    build(PASSAGES, CITATIONS, out)
    return connect(out)


def parse(text):
    return parse_query(text, glossary=GLOSSARY, surah_names=NAMES)


def ids(res):
    return [r.id for r in res.passages]


# --- building blocks ----------------------------------------------------------------------------

def test_coverage_list_requires_two_concepts_when_several():
    assert coverage_list([["a", "b", "c"], ["b", "d"]]) == ["b"]
    assert coverage_list([["a", "b"]]) == ["a", "b"]
    assert coverage_list([["x", "a", "b"], ["b", "a"], ["a"]]) == ["a", "b"]


def test_fuse_weights_and_verse_focus():
    lists = {"verse": ["v"], "concept:adam": ["c", "v"]}
    fused = fuse(lists)
    assert fused["v"][0] > fused["c"][0] and fused["v"][1] == ["verse#1", "concept:adam#2"]
    focused = fuse(lists, verse_focus=True)
    assert focused["c"][0] == pytest.approx(fused["c"][0] / 4)


def test_verse_list_finds_overlapping_sections(conn):
    assert verse_list(conn, [(2, 34, 34)]) == ["2:34#1", "2:34#2"]
    assert verse_list(conn, [(2, 32, 34)]) == ["2:31-33", "2:34#1", "2:34#2"]


def test_surah_list_puts_verse_sections_before_extra_parts(conn):
    assert surah_list(conn, [2])[:4] == ["2:0", "2:30", "2:31-33", "2:34#1"]
    assert surah_list(conn, [2]).index("2:34#2") > surah_list(conn, [2]).index("2:38-39")


def test_concept_share_measures_the_book(conn):
    q = parse("What does Allah say about Adam?")
    shares = {c.typed: concept_share(conn, c) for c in q.concepts}
    assert shares["allah"] == 1.0 and 0 < shares["adam"] < 1.0


def test_citation_list_follows_references_but_needs_a_concept(conn):
    adam = parse("Adam").concepts
    assert citation_list(conn, ["2:34"], adam) == ["18:50"]        # 18:51 never mentions Adam


def test_neighbour_list_adds_adjacent_sections_of_a_story(conn):
    adam = parse("Adam").concepts
    assert set(neighbour_list(conn, ["2:31-33", "2:35-36"], adam)) == {"2:30", "2:34#1", "2:34#2", "2:37"}
    assert neighbour_list(conn, ["2:31-33"], adam) == []            # one seed is not a story


# --- end to end -------------------------------------------------------------------------------------

def test_adam_and_iblis(conn):
    res = retrieve(conn, parse("From Ibn Kathir's explanation of the story of Adam and Iblis (Satan)"))
    got = ids(res)
    assert {"2:34#1", "2:34#2", "18:50", "7:11"} <= set(got)        # 7:11 via the alias "Shaytan"
    assert "12:4" not in got and "18:51" not in got
    assert [(r.surah, r.ayah_start) for r in res.passages] == sorted((r.surah, r.ayah_start) for r in res.passages)
    top = max(res.passages, key=lambda r: r.score)
    assert top.section_id in {"2:34", "18:50", "2:31-33"} and top.reasons
    r34 = next(r for r in res.passages if r.id == "2:34#2")
    assert r34.verse_text.startswith("(34. The angels prostrated") and r34.citation == "2:34, p. 102"


def test_named_verse_dominates(conn):
    res = retrieve(conn, parse("What does Surah Al-Baqarah verse 34 say about Adam?"))
    assert ids(res) == ["2:34#1", "2:34#2"]


def test_surah_scope_limits_keyword_search(conn):
    res = retrieve(conn, parse("Adam in Al-Baqarah"))
    assert res.passages and all(r.surah == 2 for r in res.passages)


def test_unknown_word_and_budget(conn):
    res = retrieve(conn, parse("zebra"))
    assert res.passages == [] and "'zebra' does not occur in the book" in res.notes
    assert len(retrieve(conn, parse("Adam and Iblis"), max_passages=2).passages) == 2


def test_general_question_is_flagged(conn):
    res = retrieve(conn, parse("What does Allah say?"))
    assert any("no specific subject" in n for n in res.notes)


# --- against the real index ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    if not paths.DB_PATH.exists():
        pytest.skip("data/build/tafsir.db not built")
    return connect()


@pytest.mark.data
def test_real_adam_and_iblis_spans_the_book(real):
    start = time.time()
    res = retrieve(real, parse_query("From Ibn Kathir's explanation of the story of Adam and Iblis (Satan)"))
    assert time.time() - start < 1.0
    assert {2, 7, 15, 17, 18, 20, 38} <= {r.surah for r in res.passages}


@pytest.mark.data
def test_real_named_verse_and_named_passage(real):
    res = retrieve(real, parse_query("What does Surah Al-Baqarah verse 34 say about the angels?"))
    assert {r.section_id for r in res.passages} == {"2:34"}
    res = retrieve(real, parse_query("What is the virtue of Ayat al-Kursi?"))
    assert "2:255" in {r.section_id for r in res.passages}


def test_coverage_weights_rare_concepts_over_common_pairs():
    # "Musa" (common) + "story" (common) must not outrank a passage with only "Khidr" (rare)
    story, musa, khidr = ["s1", "both"], ["s1", "both", "k1"], ["k1", "both"]
    got = coverage_list([story, musa, khidr], weights=[1.0, 1.0, 3.0])
    assert got == ["both", "k1"]                                     # "s1" (story+musa = 2.0 < 3.0) is out


def test_comparison_keeps_passages_about_one_side(conn):
    # (in this 13-passage fixture "Iblis" is too common to be a subject; "Khalifah" is rare)
    res = retrieve(conn, parse("Contrast Yusuf with the Khalifah"))
    assert res.query.compare
    got = {r.section_id for r in res.passages}
    assert {"12:4", "12:5"} <= got and "2:30" in got
