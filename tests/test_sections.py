import json
from collections import defaultdict

import pytest

from pipeline import paths
from pipeline.quran import SURAH_COUNT, VERSE_COUNTS, is_valid_verse, verse_count
from pipeline.sections import (
    MAX_PASSAGE_CHARS,
    chapter_anchor,
    headings_of,
    passages,
    section_id,
    segment,
    split_passages,
    verse_markers,
)


def body(text):
    return {"kind": "body", "text": text}


def heading(text):
    return {"kind": "heading", "text": text, "size": 13.5}


ARABIC = {"kind": "arabic", "script": "verse"}


# --- reference data -------------------------------------------------------------------------

def test_verse_counts_table():
    assert SURAH_COUNT == 114 and sum(VERSE_COUNTS) == 6236
    assert verse_count(1) == 7 and verse_count(2) == 286 and verse_count(114) == 6
    assert is_valid_verse(2, 286) and not is_valid_verse(2, 287) and not is_valid_verse(115, 1)


# --- block recognisers ----------------------------------------------------------------------

def test_chapter_anchor():
    assert chapter_anchor(heading("(Chapter - 107)")) == 107
    assert chapter_anchor(heading("(Chapter 3)")) == 3
    assert chapter_anchor(body("The Tafsir of Surat Al-Fatihah (Chapter 1)")) == 1
    assert chapter_anchor(body("This story is similar to the story in Surat Al-A`raf (Chapter 7)")) is None
    assert chapter_anchor(heading("(Chapter - 200)")) is None


@pytest.mark.parametrize("text, expected", [
    ("(30. And (remember) when your Lord said to the angels.) (31. And He taught Adam.)", [30, 31]),
    ("(1.Ta Sin. These are the Ayat of the Qur'an)", [1]),
    ("(And (remember) when We said to the angels.) (2:34)", []),
    ("(1.5 is not a verse)", []),
    ("Allah said (5. not at the start)", []),
])
def test_verse_markers(text, expected):
    assert verse_markers(text) == expected


def test_headings_of_joins_wrapped_headings_only():
    blocks = [(1, heading("This Surah has been separated from the one that preceded it in the")),
              (1, heading("primary Mushaf.")),
              (1, body("text")),
              (1, heading("Which was revealed in Makkah")),
              (1, heading("The Meaning of Al-Fatihah"))]
    assert headings_of(blocks) == [
        "This Surah has been separated from the one that preceded it in the primary Mushaf.",
        "Which was revealed in Makkah",
        "The Meaning of Al-Fatihah",
    ]


# --- segmentation ---------------------------------------------------------------------------

def book():
    """A miniature book: front matter, surah 1 (anchored), surah 2 (no anchor, found by the
    numbering restart), surah 3 anchored twice (the second copy is a duplicate)."""
    return [
        (1, body("Biography of Ibn Kathir.")),
        (2, heading("The Tafsir of Surat Al-Fatihah")),
        (2, heading("(Chapter 1)")),
        (2, body("Virtues of Al-Fatihah.")),
        (3, ARABIC),
        (3, body("(1. In the Name of Allah.) (2. All praise is due to Allah.)")),
        (3, ARABIC),
        (3, body("(3. The Most Gracious.)")),           # same header group -> 1:1-3
        (3, body("Commentary on the first verses.")),
        (4, body("(5. You we worship.)")),             # verse 4 has no printed marker
        (4, body("Commentary on 5.")),
        (4, body("(4. quoted out of order.)")),        # rejected: stays in 1:5-7
        (5, heading("Surat Al-Baqarah")),              # earliest title -> start of 2's intro
        (5, body("The individual letters Alif Lam Mim.")),
        (5, heading("Surat Al-Baqarah was revealed in Al-Madinah")),
        (5, body("More about the surah.")),
        (6, body("(2. This is the Book.)")),           # restart -> surah 2, intro takes 2:1
        (6, body("Commentary on 2:2.")),
        (7, heading("The Tafsir of Surah Al `Imran")),
        (7, heading("(Chapter 3)")),
        (7, body("(1. Alif Lam Mim.)")),
        (7, body("Commentary on 3:1.")),
        (8, heading("The Tafsir of Surah Al `Imran")),  # duplicate copy's title, dropped
        (8, heading("(Chapter 3)")),                   # duplicate copy, skipped
        (8, body("(1. Alif Lam Mim.)")),
        (8, body("Duplicate commentary.")),
        (9, heading("The Tafsir of Surat An-Nisa")),   # seen while skipping; belongs to 4
        (9, heading("Chapter - 4)")),                  # anchor printed without "("
        (9, body("(1. O mankind!)")),
    ]


def test_segment_structure():
    sections, report = segment(book())
    ids = [section_id(s) for s in sections]
    assert ids == ["0:0", "1:0", "1:1-4", "1:5-7", "2:1", "2:2-286", "3:0", "3:1-200", "4:0", "4:1-176"]
    by_id = dict(zip(ids, sections))
    texts = {sid: [b["text"] for _, b in s["blocks"] if "text" in b] for sid, s in by_id.items()}
    assert by_id["1:1-4"]["marked"] == [1, 2, 3]
    assert by_id["2:1"]["kind"] == "intro"            # carved from the end of surah 1
    assert texts["1:5-7"][-1] == "(4. quoted out of order.)"
    assert texts["2:1"][:2] == ["Surat Al-Baqarah", "The individual letters Alif Lam Mim."]
    assert "Surat Al-Baqarah was revealed in Al-Madinah" in texts["2:1"]
    assert texts["3:0"][0] == "The Tafsir of Surah Al `Imran"
    assert texts["3:1-200"] == ["(1. Alif Lam Mim.)", "Commentary on 3:1."]
    assert texts["4:0"] == ["The Tafsir of Surat An-Nisa", "Chapter - 4)"]
    assert report["resets"] == [(6, 2)]
    assert report["duplicates"] == [(8, 3)]
    assert report["rejected_headers"] == [(4, 1, [4])]
    assert not any("Duplicate" in t for ts in texts.values() for t in ts)


def test_passages_records():
    sections, _ = segment(book())
    recs = passages(sections)
    assert len({r["id"] for r in recs}) == len(recs)
    r = next(r for r in recs if r["id"] == "1:1-4")
    assert (r["surah"], r["ayah_start"], r["ayah_end"], r["page_start"], r["page_end"]) == (1, 1, 4, 3, 3)
    assert r["text"].startswith("(1. In the Name of Allah.)")


# --- passage splitting ------------------------------------------------------------------------

def test_split_passages_at_headings_and_merges_small_parts():
    para = "x" * 1000
    blocks = [(1, body("(1. verse)")), (1, body(para)),
              (1, heading("Topic A")), (1, body(para)),
              (1, heading("Tiny")), (1, body("y" * 50)),
              (2, heading("Topic B")), (2, body(para))]
    parts = split_passages(blocks)
    titles = [headings_of(p) for p in parts]
    assert titles == [[], ["Topic A", "Tiny"], ["Topic B"]]


def test_split_passages_splits_long_parts_at_paragraphs():
    blocks = [(1, body("p" * 2000)) for _ in range(7)]
    parts = split_passages(blocks)
    assert len(parts) == 3
    assert all(sum(len(b["text"]) for _, b in p) <= MAX_PASSAGE_CHARS for p in parts)
    assert sum(len(p) for p in parts) == 7


# --- full build output (run `python -m pipeline.sections` first) ----------------------------------

@pytest.fixture(scope="module")
def recs():
    if not paths.PASSAGES_JSONL.exists():
        pytest.skip("data/build/passages.jsonl not built")
    with paths.PASSAGES_JSONL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.mark.data
def test_every_verse_in_exactly_one_section(recs):
    ranges = defaultdict(set)
    for r in recs:
        if r["ayah_start"] > 0:
            ranges[r["surah"]].add((r["ayah_start"], r["ayah_end"]))
    assert sorted(ranges) == list(range(1, 115))
    for surah, rs in ranges.items():
        verses = [v for a, b in sorted(rs) for v in range(a, b + 1)]
        assert verses == list(range(1, verse_count(surah) + 1)), f"surah {surah}"


@pytest.mark.data
def test_most_verses_have_printed_markers(recs):
    marked = {(r["surah"], v) for r in recs for v in r["marked_verses"]}
    assert len(marked) / 6236 > 0.94
    assert all(is_valid_verse(s, v) for s, v in marked)


@pytest.mark.data
def test_passage_records_are_well_formed(recs):
    assert len({r["id"] for r in recs}) == len(recs)
    assert all(r["text"].strip() for r in recs)
    assert max(len(r["text"]) for r in recs) <= 2 * MAX_PASSAGE_CHARS
    starts = [r["page_start"] for r in recs]
    assert starts == sorted(starts)
    assert recs[0]["kind"] == "front" and recs[-1]["surah"] == 114


@pytest.mark.data
def test_adam_and_iblis_sections(recs):
    by_title = {r["title"]: r for r in recs if r["surah"] == 2}
    r = by_title["The Prostration was before Adam but the Obedience was to Allah"]
    assert r["section_id"] == "2:34" and r["page_start"] == 161
    assert by_title["Adam repents and supplicates to Allah"]["section_id"] == "2:37"
    assert any(r["section_id"] == "2:1" and "Individual Letters" in (r["title"] or "") for r in recs)
