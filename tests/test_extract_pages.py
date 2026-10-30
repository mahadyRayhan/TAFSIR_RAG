import json
import re

import pytest

from pipeline import paths
from pipeline.extract_pages import (
    blocks_from_page_dict,
    build_vocabulary,
    classify_line,
    clean_text,
    join_lines,
    page_text,
    repair_alt_encoding,
)


def span(text, font="Helvetica", size=10.0):
    return {"text": text, "font": font, "size": size}


def line(*spans):
    return {"spans": list(spans)}


# --- clean_text / join_lines -------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Ibn ` Abbas said", "Ibn `Abbas said"),
    ("Mu` adh and Ya` la", "Mu`adh and Ya`la"),
    ("as in (41 :34-36) and ( 2 : 30 )", "as in (41:34-36) and (2:30)"),
    ("see (2:30 – 33)", "see (2:30-33)"),
    ("the Prophet , said", "the Prophet, said"),
    ("too   many   spaces ", "too many spaces"),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent():
    s = "Ibn ` Abbas  said (41 :34-36) , then"
    assert clean_text(clean_text(s)) == clean_text(s)


def test_join_lines_keeps_name_hyphens():
    assert join_lines(["recorded by Imam Al-", "Bukhari and", "Muslim. "]) == "recorded by Imam Al-Bukhari and Muslim."


# --- line classification ---------------------------------------------------------------

def test_classify_body_heading_arabic():
    assert classify_line(line(span("Allah said,")))[0] == "body"
    assert classify_line(line(span("The Story of Adam", "Helvetica-Bold", 13.5)))[0] == "heading"
    kind, _, arabic, _ = classify_line(line(span("ぴ∇ク⌒ま∠ヱび", "TimesNewRomanPSMT", 30)))
    assert kind == "arabic" and "ぴ" in arabic


def test_arabic_span_inside_english_line_is_dropped():
    kind, english, _, _ = classify_line(line(span("Allah's statement, "), span("ゅ∠", "ArialMT"), span("and then")))
    assert kind == "body"
    assert "ゅ" not in english and "statement" in english and "and then" in english


def test_blocks_from_page_dict_merges_arabic_and_labels_script():
    page = {"blocks": [
        {"lines": [line(span("A Heading", "Helvetica-Bold", 13.5))]},
        {"lines": [line(span("Qatadah commented on Allah's statement,"))]},
        {"lines": [line(span("ぴ∇ク⌒ま", "TimesNewRomanPSMT", 30))]},
        {"lines": [line(span("∠ヱび", "TimesNewRomanPSMT", 30))]},
        {"lines": [line(span("(And (remember) when We said to the angels.) (2 :34)"))]},
        {"lines": [line(span("»ゅ∠ャ«", "TimesNewRomanPSMT", 30))]},
    ]}
    blocks = blocks_from_page_dict(page)
    assert [b["kind"] for b in blocks] == ["heading", "body", "arabic", "body", "arabic"]
    assert blocks[0]["size"] == 13.5
    assert blocks[2]["script"] == "verse"
    assert blocks[3]["text"].endswith("(2:34)")
    assert blocks[4]["script"] == "hadith"
    assert page_text({"blocks": blocks}).count("\n\n") == 2


# --- alternate encoding repair ---------------------------------------------------------

VOCAB = {"al-hasan", "hasan", "al-basri", "basri", "ad-dahhak", "dahhak", "hadith", "allah"}


@pytest.mark.parametrize("raw, expected", [
    ("Al-H\x1f asan Al-Bas\x1f ri said", "Al-Hasan Al-Basri said"),
    ("Mujahid and Ad-D\x1f ah\x1f h\x1f a0k said", "Mujahid and Ad-Dahhak said"),
    ("from Alla0h\x1f )2:140(. Then", "from Allah (2:140). Then"),
    ("(So when you \x1f want to\x1f recite the Qur'an", "(So when you (want to) recite the Qur'an"),
    ("the Ayah \x1f 3:10\x1f, and", "the Ayah (3:10), and"),
    ("dream.) \x1f 20: 110\x1f. Then", "dream.) (20:110). Then"),
    ("In the H\x1f ad0th of Jar0r and Ibra0h0m", "In the Hadith of Jarir and Ibrahim"),
    ("Abu Dawud and An-Nasa ®299\x1f. Allah", "Abu Dawud and An-Nasa. Allah"),
    ("listed: \x1f \x1f \x1f \x1f \x1f:", "listed::"),
])
def test_repair_alt_encoding(raw, expected):
    assert repair_alt_encoding(raw, VOCAB) == expected


def test_repair_leaves_clean_text_alone():
    s = "Ibn `Abbas said (2:30) in 1400 H."
    assert repair_alt_encoding(s, VOCAB) == s


def test_build_vocabulary_ignores_damaged_text_and_singletons():
    vocab = build_vocabulary(["Al-Hasan said", "Al-Hasan said", "Alla0h said", "Alla0h said", "rare"])
    assert "al-hasan" in vocab and "said" in vocab
    assert "alla0h" not in vocab and "rare" not in vocab


# --- full build output (run `python -m pipeline.extract_pages` first) --------------------

@pytest.fixture(scope="module")
def pages():
    if not paths.PAGES_JSONL.exists():
        pytest.skip("data/build/pages.jsonl not built")
    with paths.PAGES_JSONL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def full_text(pages):
    return "\n".join(page_text(r) for r in pages)


def by_page(pages, n):
    return next(r for r in pages if r["page"] == n)


@pytest.mark.data
def test_every_page_present_in_order(pages):
    assert [r["page"] for r in pages] == list(range(1, 5699))


@pytest.mark.data
def test_no_encoding_junk_left(full_text):
    assert not re.search(r"[぀-ヿ∀-⏿＀-￯]", full_text), "Arabic glyph junk"
    assert "\x1f" not in full_text and "®" not in full_text
    assert not re.search(r"[A-Za-z]0", full_text), "unrepaired long-vowel marks"


@pytest.mark.data
def test_broken_words_are_rare(full_text):
    for broken in ("th e ", "S urat", "wh ich", "Al lah"):
        assert broken not in full_text
    stray = len(re.findall(r"(?<![A-Za-z`'\-])[b-hj-z](?![A-Za-z`'\-.)])", full_text))
    assert stray / (len(full_text) / 1e6) < 100  # PyPDF2 text had ~3,400 per million chars


@pytest.mark.data
def test_references_are_normalized(full_text):
    assert len(re.findall(r"\(\d{1,3}:\d{1,3}(?:-\d{1,3})?\)", full_text)) > 3000
    assert not re.search(r"\(\s+\d{1,3}:|\(\d{1,3}\s+:|\)\d{1,3}:\d{1,3}\(", full_text)


@pytest.mark.data
def test_known_passages(pages, full_text):
    p161 = by_page(pages, 161)
    headings = [b["text"] for b in p161["blocks"] if b["kind"] == "heading"]
    assert "The Prostration was before Adam but the Obedience was to Allah" in headings
    assert "Ibn `Abbas" in page_text(p161) and "(12:100)" in page_text(p161)
    assert "Al-Hasan Al-Basri" in page_text(by_page(pages, 367))
    assert "(30. And (remember) when your Lord said to the angels" in full_text


@pytest.mark.data
def test_arabic_markers_and_empty_pages(pages):
    verses = sum(1 for r in pages for b in r["blocks"] if b["kind"] == "arabic" and b["script"] == "verse")
    assert verses > 20000
    empty = [r for r in pages if not any(b["kind"] != "arabic" for b in r["blocks"])]
    assert len(empty) <= 20
    assert all(any(b["kind"] == "arabic" for b in r["blocks"]) for r in empty)
