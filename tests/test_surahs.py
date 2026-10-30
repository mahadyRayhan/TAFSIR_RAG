import pytest

from pipeline import paths
from pipeline.surahs import build_rows, load_surah_names, merge_with_existing, mine_names, read_rows, surah_key


@pytest.mark.parametrize("a, b", [
    ("Al-Fatihah", "fatiha"), ("Surat Al-Baqarah", "Al-Baqara"), ("Al `Imran", "Al-Imran"),
    ("An-Nisa", "Nisa'"), ("`Abasa", "Abasa"), ("Ya Sin", "Yasin"), ("Quraysh", "Quraish"),
])
def test_surah_key_matches_spelling_variants(a, b):
    assert surah_key(a) == surah_key(b)


def test_surah_key_keeps_different_surahs_apart():
    assert surah_key("An-Nas") != surah_key("An-Nisa")
    assert surah_key("Al-Falaq") != surah_key("Al-Fil")


def passage(pid, surah, kind, headings=(), text=""):
    return {"id": pid, "surah": surah, "kind": kind, "headings": list(headings), "text": text}


def test_mine_names_from_opening_closing_and_heading_lines():
    ps = [
        passage("2:1", 2, "intro", ["Surat Al-Baqarah", "Which was revealed in Al-Madinah"]),
        passage("49:0", 49, "intro", ["(Chapter - 49)"]),
        passage("49:14-18", 49, "verses", text="This is the end of the Tafsir of Surat Al-Hujurat. Verily"),
        passage("94:0", 94, "intro", ["The Tafsir of Surah Alam Nashrah (Surat Ash-Sharh)", "(Chapter - 94)"]),
        passage("80:0", 80, "intro", text="The Tafsir of Surah `Abasa\n\n(Chapter - 80)"),
        passage("60:7", 60, "verses", text="as explained in the Tafsir of Surat Al-Hujurat. We"),  # not 60's name
    ]
    names = mine_names(ps)
    assert list(names[2]) == ["Al-Baqarah"]
    assert list(names[49]) == ["Al-Hujurat"]
    assert list(names[94]) == ["Alam Nashrah", "Ash-Sharh"]
    assert list(names[80]) == ["`Abasa"]
    assert 60 not in names


def test_build_rows_fills_mu_awwidhatayn_and_reports_missing():
    rows = {int(r["surah"]): r for r in build_rows([])}
    assert len(rows) == 114
    assert rows[113]["names"] == "Al-Falaq" and "Mu`awwidhatayn" in rows[113]["evidence"]
    assert rows[1]["names"] == "" and rows[1]["evidence"] == "not found in book"


def test_reviewer_names_survive_regeneration():
    new = [{"surah": "2", "names": "Al-Baqarah", "evidence": "x"}]
    old = [{"surah": "2", "names": "Al-Baqarah | The Cow", "evidence": "y"}]
    assert merge_with_existing(new, old)[0]["names"] == "Al-Baqarah | The Cow"


@pytest.mark.data
def test_reviewed_file_names_every_surah_once():
    if not paths.SURAH_NAMES_CSV.exists():
        pytest.skip("glossary/surah_names.csv not generated")
    rows = read_rows(paths.SURAH_NAMES_CSV)
    assert [int(r["surah"]) for r in rows] == list(range(1, 115))
    assert all(r["names"].strip() for r in rows)
    table = load_surah_names()
    assert len(set(table.values())) == 114
    assert table[surah_key("Al-Kahf")] == 18 and table[surah_key("An-Nas")] == 114
