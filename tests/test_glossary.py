import csv

import pytest

from pipeline import paths
from pipeline.glossary import (
    candidates,
    clean_gloss,
    is_english,
    load_glossary,
    merge_with_existing,
    normalize_term,
    orthographic_key,
    read_rows,
    term_key,
    trim_term,
    write_rows,
)

# A tiny stand-in for /usr/share/dict/words so the tests do not depend on the OS.
EN = {"prayer", "piety", "righteousness", "satan", "devil", "except", "and", "the", "said",
      "waiting", "period", "may", "mercy", "on", "him", "he", "was", "a", "noble", "book"}


def test_is_english_handles_inflections():
    assert is_english("Prayers", EN) and is_english("devils", EN)
    assert not is_english("Taqwa", EN) and not is_english("Shaytan", EN)


def test_keys():
    assert term_key("As-Salah") == term_key("Salah") == "salah"
    assert term_key("Fir`awn") == "firawn" and term_key("Isma'il") == "ismail"
    assert orthographic_key("Quraysh") == orthographic_key("Quraish")
    assert orthographic_key("Hellfire") == orthographic_key("Hell-fire")
    assert orthographic_key("Salah") != orthographic_key("Salih")
    assert orthographic_key("Umar") != orthographic_key("Amr")


@pytest.mark.parametrize("gloss, expected", [
    ("piety, righteousness", ["piety", "righteousness"]),
    ("the prayer", ["prayer"]),
    ("Satan", ["Satan"]),
    ("Shaytan", ["Shaytan"]),                         # a single transliterated term
    ("bin Al-Khattab", []),                           # continues a name
    ("may Allah have mercy on him", []),              # honorific
    ("he", []),
    ("he was a noble book", []),                      # too long
])
def test_clean_gloss(gloss, expected):
    assert clean_gloss(gloss, EN) == expected


def test_trim_term():
    assert trim_term("Except Iblis", EN) == "Iblis"
    assert trim_term("Abu Hurayrah", EN) == "Abu Hurayrah"


def passage(pid, text):
    return {"id": pid, "text": text}


def test_candidates_from_book_text():
    ps = [passage("2:3", "They perform As-Salah (the prayer). Salah (prayer) is due. Iblis (Satan) refused."),
          passage("7:11", "Except Iblis (Satan). Iblis (Shaytan) and Iblis (Shaytan). The Quraysh and Quraysh."),
          passage("9:1", "Quraish came. Quraish left. Umar (bin Al-Khattab) and Umar (bin Al-Khattab).")]
    rows = {r["id"]: r for r in candidates(ps, EN)}
    assert rows["gloss:salah"]["terms"] == "As-Salah | Salah | prayer"
    assert rows["gloss:iblis"]["terms"] == "Iblis | Satan | Shaytan"
    assert rows["spelling:quraysh"]["terms"] in ("quraysh | quraish", "quraish | quraysh")
    assert "gloss:umar" not in rows
    assert all(r["keep"] == "" for r in rows.values())


def test_merge_keeps_review_and_hand_added_rows():
    existing = [{"id": "gloss:salah", "keep": "y", "terms": "Salah | prayer", "evidence": "old", "example": ""},
                {"id": "custom:adam", "keep": "y", "terms": "Adam | first man", "evidence": "", "example": ""}]
    new = [{"id": "gloss:salah", "keep": "", "terms": "As-Salah | Salah | prayer | blessings",
            "evidence": "new", "example": ""}]
    merged = merge_with_existing(new, existing)
    assert merged[0]["keep"] == "y" and merged[0]["terms"] == "Salah | prayer"
    assert merged[0]["evidence"] == "new"
    assert merged[1]["id"] == "custom:adam"


def test_load_glossary_uses_only_approved_rows_and_merges_overlaps(tmp_path):
    path = tmp_path / "g.csv"
    write_rows([
        {"id": "gloss:iblis", "keep": "y", "terms": "Iblis | Shaytan | Satan", "evidence": "", "example": ""},
        {"id": "gloss:shaytan", "keep": "Y", "terms": "Shaytan | devil", "evidence": "", "example": ""},
        {"id": "gloss:salah", "keep": "", "terms": "Salah | prayer", "evidence": "", "example": ""},
        {"id": "gloss:taqwa", "keep": "n", "terms": "At-Taqwa | piety", "evidence": "", "example": ""},
    ], path)
    assert read_rows(path)[0]["id"] == "gloss:iblis"
    assert load_glossary(path) == [{"iblis", "shaytan", "satan", "devil"}]


def test_normalize_term():
    assert normalize_term("At-Taqwa") == "taqwa"
    assert normalize_term("Waiting Period") == "waiting period"
    assert normalize_term("Mu`adh") == "muadh"


@pytest.mark.data
def test_generated_glossary_has_core_terms():
    if not paths.GLOSSARY_CSV.exists():
        pytest.skip("glossary/glossary.csv not generated")
    with paths.GLOSSARY_CSV.open(newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert "Satan" in rows["gloss:iblis"]["terms"]
    assert "piety" in rows["gloss:taqwa"]["terms"]
    assert "devils" in rows["gloss:shayatin"]["terms"]
