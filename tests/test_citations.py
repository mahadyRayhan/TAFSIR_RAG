import json

import pytest

from pipeline import paths
from pipeline.citations import citations, parse_refs, section_edges, verse_index


def refs_of(text):
    return [r for r, _ in parse_refs(text)[0]]


@pytest.mark.parametrize("text, expected", [
    ("(And they fell down prostrate.) (12:100)", [(12, 100, 100)]),
    ("as Allah said (7:199-200), and", [(7, 199, 200)]),
    ("united them) 8:62,63, until", [(8, 62, 62), (8, 63, 63)]),
    ("see (2:6-7, 9)", [(2, 6, 7), (2, 9, 9)]),
    ("(2:30) and (41:34-36)", [(2, 30, 30), (41, 34, 36)]),
    ("Ibn Kathir died in 774 H.", []),
])
def test_parse_refs(text, expected):
    assert refs_of(text) == expected


def test_parse_refs_skips_invalid_and_hadith_sources():
    refs, skipped = parse_refs("Muslim recorded similar wording (1:554) and (2:287)")
    assert refs == [] and skipped["invalid"] == 2
    refs, skipped = parse_refs("collected by An-Nasa'i (Al-Kubra 5:12) and")
    assert refs == [] and skipped["source"] == 1


def passage(pid, surah, a, b, text, sid=None):
    return {"id": pid, "section_id": sid or (f"{surah}:{a}-{b}" if a else f"{surah}:0"),
            "surah": surah, "ayah_start": a, "ayah_end": b, "text": text}


def test_citations_resolve_targets_and_flag_self_refs():
    ps = [
        passage("2:1-6", 2, 1, 6, "Commentary citing itself (2:3) and Al-Kahf (18:50)."),
        passage("2:7-10", 2, 7, 10, "Spans two sections (2:5-8)."),
        passage("18:50-51", 18, 50, 51, "Except Iblis."),
        passage("18:0", 18, 0, 0, "Intro without a range."),
    ]
    assert verse_index(ps)[(2, 7)] == "2:7-10" and (18, 0) not in verse_index(ps)
    rows, stats = citations(ps)
    by_raw = {r["raw"]: r for r in rows}
    assert by_raw["2:3"]["self"] and by_raw["2:3"]["targets"] == ["2:1-6"]
    assert by_raw["18:50"]["targets"] == ["18:50-51"] and not by_raw["18:50"]["self"]
    assert by_raw["2:5-8"]["targets"] == ["2:1-6", "2:7-10"]
    assert stats["refs"] == stats["resolved"] == 3 and stats["self"] == 1
    assert section_edges(rows) == {("2:1-6", "18:50-51"): 1, ("2:7-10", "2:1-6"): 1}


# --- full build output (run `python -m pipeline.citations` first) ------------------------------

@pytest.fixture(scope="module")
def rows():
    if not paths.CITATIONS_JSONL.exists():
        pytest.skip("data/build/citations.jsonl not built")
    with paths.CITATIONS_JSONL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.mark.data
def test_references_resolve(rows):
    assert len(rows) > 4500
    assert all(r["targets"] for r in rows)


@pytest.mark.data
def test_graph_links_adam_story_to_al_kahf(rows):
    edges = section_edges(rows)
    assert ("2:34", "18:50") in edges
