import pytest

from pipeline import paths
from pipeline.surahs import surah_key
from rag.query import fts_terms, parse_query

NAMES = {surah_key(n): s for n, s in [("Al-Fatihah", 1), ("Al-Baqarah", 2), ("Al `Imran", 3),
                                     ("Yusuf", 12), ("Al-Kahf", 18), ("Ya Sin", 36)]}
GLOSSARY = [{"iblis", "shaytan", "satan", "devil"}, {"taqwa", "piety", "righteousness"},
            {"muttaqin", "pious"}, {"iddah", "waiting period"}]


def parse(text):
    return parse_query(text, glossary=GLOSSARY, surah_names=NAMES)


def typed(q):
    return [c.typed for c in q.concepts]


# --- verse and surah references ---------------------------------------------------------------

@pytest.mark.parametrize("text, verses, surahs", [
    ("What is said in 2:255 and (3:7)?", [(2, 255, 255), (3, 7, 7)], []),
    ("Explain 2:30-39", [(2, 30, 39)], []),
    ("What does Surah Al-Baqarah verse 34 say?", [(2, 34, 34)], []),
    ("Surat Al-Baqara, ayah 30-33", [(2, 30, 33)], []),
    ("chapter 18 verse 50", [(18, 50, 50)], []),
    ("Explain verses 30 to 39 of Al-Baqarah", [(2, 30, 39)], []),
    ("ayah 7 of Surah Al `Imran", [(3, 7, 7)], []),
    ("Al-Imran verses 1-4", [(3, 1, 4)], []),
    ("Surah Ya Sin", [], [36]),
    ("Explain the teaching of Surah Al-Fatiha", [], [1]),
    ("the hypocrites in Al-Baqarah", [], [2]),
    ("the people of the Al-Kahf", [], [18]),
])
def test_references(text, verses, surahs):
    q = parse(text)
    assert q.verses == verses and q.surahs == surahs


def test_out_of_range_verse_is_reported_not_guessed():
    q = parse("Al-Baqarah verse 300")
    assert q.verses == [] and "286 verses" in q.notes[0]


def test_prophet_name_without_surah_word_stays_a_concept():
    q = parse("Story of Yusuf and his brothers")
    assert q.surahs == [] and typed(q) == ["story", "yusuf", "brothers"]
    assert parse("Surah Yusuf").surahs == [12]


def test_references_are_removed_from_concepts():
    q = parse("What does Surah Al-Baqarah verse 34 say about the angels?")
    assert typed(q) == ["angels"]


# --- concepts --------------------------------------------------------------------------------

def test_adam_and_iblis_question():
    q = parse("From Ibn Kathir's explanation of the story of Adam and Iblis (Satan)")
    assert typed(q) == ["story", "adam", "iblis"]          # "Satan" folds into Iblis's group
    iblis = q.concepts[2]
    assert iblis.aliases == ["devil", "satan", "shaytan"] and iblis.proper
    assert q.concepts[1].proper and not q.concepts[0].proper


def test_scaffolding_and_ibn_kathir_are_dropped():
    q = parse("Explain the concept of taqwa according to Ibn Kathir's Tafsir")
    assert typed(q) == ["taqwa"] and q.concepts[0].aliases == ["piety", "righteousness"]


def test_lowercase_article_and_multiword_glossary_terms():
    q = parse("the believers (al-muttaqin) and the waiting period of a woman")
    assert typed(q) == ["believers", "muttaqin", "waiting period", "woman"]
    assert q.concepts[2].aliases == ["iddah"]


def test_name_phrase_kept_whole_with_article():
    assert typed(parse("What is the virtue of Ayat al-Kursi?")) == ["virtue", "ayat al-kursi"]


def test_fts_terms():
    assert fts_terms(["iblis", "waiting period", 'x"y']) == '("iblis" OR "waiting period" OR "x y")'


# --- with the real reviewed files ------------------------------------------------------------------

@pytest.mark.data
def test_real_files_parse_core_questions():
    if not (paths.GLOSSARY_CSV.exists() and paths.SURAH_NAMES_CSV.exists()):
        pytest.skip("glossary files missing")
    q = parse_query("From Ibn Kathir's explanation of the story of Adam and Iblis (Satan)")
    assert "shaytan" in next(c for c in q.concepts if c.typed == "iblis").aliases
    assert parse_query("Surah Al-Fatiha verse 5").verses == [(1, 5, 5)]
    assert parse_query("Explain Surah `Abasa").surahs == [80]
    assert parse_query("Al-Falaq").surahs == [] and parse_query("Surah Al-Falaq").surahs == [113]


@pytest.mark.parametrize("text, compare", [
    ("Contrast the believers with the hypocrites", True),
    ("What is the difference between Salah and Sawm?", True),
    ("Musa versus Fir`awn", True),
    ("The story of Adam and Iblis", False),
])
def test_comparison_questions_are_flagged(text, compare):
    assert parse(text).compare is compare
