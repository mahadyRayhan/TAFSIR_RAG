"""Step 1: extract clean English text from the Tafsir PDF, one JSON record per page.

The PDF stores Arabic (Qur'an verses and hadith) in fonts with a custom encoding that
extracts as junk such as "ぴ∇ク⌒ま…び". Those spans are dropped by font and each run is
replaced with an `arabic` marker block, labelled `verse` (wrapped in ぴ…び, the ﴿…﴾
ornaments) or `hadith` (wrapped in »…«), so later steps know where a verse stood.

Output record:
    {"page": 161, "blocks": [
        {"kind": "body", "text": "..."},
        {"kind": "heading", "text": "...", "size": 13.5},
        {"kind": "arabic", "script": "verse"}]}

Usage:
    python -m pipeline.extract_pages [--pdf PATH] [--out PATH] [--pages 1-200]
"""
import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import pymupdf

from pipeline import paths

ENGLISH_FONTS = {"Helvetica", "Helvetica-Bold", "Times-Bold"}
ARABIC_FONTS = {"TimesNewRomanPSMT", "ArialMT", "Arial-BoldMT"}

VERSE_ORNAMENTS = set("ぴび")
HADITH_ORNAMENTS = set("»«")

_REF_RE = re.compile(r"\(\s*(\d{1,3})\s*:\s*(\d{1,3})(?:\s*[-–—]\s*(\d{1,3}))?\s*\)")
_AYN_RE = re.compile(r"`\s+(?=[A-Za-z])")
_SPACES_RE = re.compile(r"[ \t ]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([,.;:!?])")


def _normalize_ref(m: re.Match) -> str:
    surah, start, end = m.group(1), m.group(2), m.group(3)
    return f"({surah}:{start}-{end})" if end else f"({surah}:{start})"


def clean_text(text: str) -> str:
    """Normalize one paragraph of English text."""
    text = unicodedata.normalize("NFKC", text)
    text = _REF_RE.sub(_normalize_ref, text)       # "(41 :34 - 36)" -> "(41:34-36)"
    text = _AYN_RE.sub("`", text)                  # "Ibn ` Abbas"   -> "Ibn `Abbas"
    text = _SPACES_RE.sub(" ", text)
    # The ﷺ glyph after "the Prophet" is not extractable and leaves "Prophet ,".
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


# --- Repair of the alternate transliteration encoding --------------------------------
# About 800 pages were typeset with a font whose glyphs map to odd code points:
#   "\x1f" is either a parenthesis ("\x1f 8:29\x1f" = "(8:29)") or an under-dot
#          diacritic ("H\x1f asan" = "Hasan", "Ad-D\x1f ah\x1f h\x1f a0k" = "Ad-Dahhak");
#   "0"    is a long vowel: after a vowel it is a macron ("Alla0h" = "Allah"), after a
#          consonant it stands for i ("Jar0r" = "Jarir", "Ibra0h0m" = "Ibrahim");
#   ")2:140(" is a reversed reference and "®363"-style codes are unmapped glyphs.
# The two uses of "\x1f" share one glyph, so a diacritic is told apart by checking
# whether the joined word occurs in the cleanly encoded pages (`build_vocabulary`).

ALT_MARK = "\x1f"
_WORD_RE = re.compile(r"[A-Za-z`'\-]+")
_MARK_RUN_RE = re.compile(r"\x1f(?: ?\x1f){2,}")
_REVERSED_REF_RE = re.compile(r"\)\s*(\d{1,3}\s*:\s*\d{1,3}(?:\s*-\s*\d{1,3})?)\s*\(")
_MACRON_AFTER_VOWEL_RE = re.compile(r"(?<=[AaEeIiOoUu])0")
_LONG_I_RE = re.compile(r"(?<=[A-Za-z])0")
_UNMAPPED_GLYPH_RE = re.compile(r" ?[®£]\d{3}")
_OPENERS = set(" \n\"'`(")


def needs_repair(text: str) -> bool:
    return ALT_MARK in text or bool(_UNMAPPED_GLYPH_RE.search(text)) or bool(re.search(r"[A-Za-z]0", text)) \
        or bool(_REVERSED_REF_RE.search(text))


def build_vocabulary(texts: Iterable[str]) -> set:
    """Lower-cased words (hyphenated names kept whole) from text that needs no repair."""
    counts: Counter = Counter()
    for text in texts:
        if not needs_repair(text):
            counts.update(w.lower().strip("-'`") for w in _WORD_RE.findall(text))
    return {w for w, n in counts.items() if n >= 2 and w}


def _diacritic_chain(text: str, i: int, vocab: set) -> Optional[int]:
    """If the mark at text[i] (and any marks chained after it) sit inside one word that
    exists in `vocab` once the marks are removed, return the end index of that chain."""
    left = re.search(r"[A-Za-z`'\-]+$", text[:i])
    if not left:
        return None
    pieces, j, best = [left.group()], i, None
    while j < len(text) and text[j] == ALT_MARK:
        k = j + 1
        if k < len(text) and text[k] == " ":
            k += 1
        right = re.match(r"[a-z][A-Za-z`'\-]*", text[k:]) if k < len(text) else None
        if not right:
            break
        pieces.append(right.group())
        end = k + right.end()
        word = "".join(pieces)
        # Look up the whole word and, for "Ad-Dahhak"-style names, its last component.
        if word.lower().strip("-'`") in vocab or word.split("-")[-1].lower() in vocab:
            best = end
        j = end
    return best


def repair_alt_encoding(text: str, vocab: set) -> str:
    if not needs_repair(text):
        return text
    text = _UNMAPPED_GLYPH_RE.sub("", text)
    text = _REVERSED_REF_RE.sub(r"(\1)", text)
    text = _MARK_RUN_RE.sub(" ", text)
    text = _MACRON_AFTER_VOWEL_RE.sub("", text)
    text = _LONG_I_RE.sub("i", text)

    out: List[str] = []
    i = 0
    open_paren = False
    while i < len(text):
        ch = text[i]
        if ch != ALT_MARK:
            out.append(ch)
            i += 1
            continue
        prev = out[-1] if out else " "
        if prev in _OPENERS:
            out.append("(")
            open_paren = True
            i += 2 if text[i + 1:i + 2] == " " else 1
            continue
        chain_end = _diacritic_chain(text, i, vocab)
        if chain_end is not None:
            out.append(text[i:chain_end].replace(ALT_MARK + " ", "").replace(ALT_MARK, ""))
            i = chain_end
            continue
        if open_paren:
            out.append(")")
            open_paren = False
        i += 1
    return clean_text("".join(out))


def join_lines(lines: Iterable[str]) -> str:
    """Join the lines of a paragraph. A trailing hyphen is kept and joined without a
    space, since in this book it almost always splits a name ("Al-" / "Baqarah")."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out or out.endswith("-"):
            out += line
        else:
            out += " " + line
    return out


def font_class(font: str) -> str:
    if font in ARABIC_FONTS:
        return "arabic"
    return "english"


def classify_line(line: dict) -> Tuple[str, str, str, float]:
    """Return (kind, english_text, arabic_text, max_english_size) for a PyMuPDF line dict.

    kind is "arabic" when the line holds more Arabic than English characters, "heading"
    when every English character is bold, and "body" otherwise.
    """
    english, arabic = [], []
    bold_chars = english_chars = 0
    size = 0.0
    for span in line.get("spans", []):
        text = span.get("text", "")
        n = len(text.strip())
        if font_class(span.get("font", "")) == "arabic":
            arabic.append(text)
            english.append(" ")  # keep word separation where an Arabic span was dropped
            continue
        english.append(text)
        if n:
            english_chars += n
            size = max(size, float(span.get("size", 0.0)))
            if "Bold" in span.get("font", ""):
                bold_chars += n
    arabic_text = "".join(arabic)
    arabic_chars = len(arabic_text.replace(" ", ""))
    if arabic_chars > english_chars:
        kind = "arabic"
    elif english_chars and bold_chars == english_chars:
        kind = "heading"
    else:
        kind = "body"
    return kind, "".join(english), arabic_text, size


def arabic_script(raw: str) -> str:
    chars = set(raw)
    if chars & VERSE_ORNAMENTS:
        return "verse"
    if chars & HADITH_ORNAMENTS:
        return "hadith"
    return "other"


def blocks_from_page_dict(page_dict: dict) -> List[dict]:
    """Turn a `page.get_text("dict")` result into a list of clean blocks.

    Consecutive lines of the same kind inside one PDF block form one output block;
    consecutive Arabic runs are merged even across PDF blocks, because a single verse
    is often laid out as several blocks.
    """
    out: List[dict] = []
    for pdf_block in page_dict.get("blocks", []):
        groups: List[dict] = []
        for line in pdf_block.get("lines", []):
            kind, english, arabic, size = classify_line(line)
            if kind != "arabic" and not english.strip():
                continue
            if not groups or groups[-1]["kind"] != kind:
                groups.append({"kind": kind, "english": [], "arabic": [], "size": 0.0})
            group = groups[-1]
            group["english"].append(english)
            group["arabic"].append(arabic)
            group["size"] = max(group["size"], size)

        for group in groups:
            kind, size = group["kind"], group["size"]
            english_lines, arabic_parts = group["english"], group["arabic"]
            if kind == "arabic":
                script = arabic_script("".join(arabic_parts))
                if out and out[-1]["kind"] == "arabic":
                    prev = out[-1]["script"]
                    if prev == "other" or script == "verse":
                        out[-1]["script"] = script
                else:
                    out.append({"kind": "arabic", "script": script})
                continue
            text = clean_text(join_lines(english_lines))
            if not text:
                continue
            block = {"kind": kind, "text": text}
            if kind == "heading":
                block["size"] = round(size, 1)
            out.append(block)
    return out


def page_text(record: dict) -> str:
    """English text of a page record, paragraphs separated by blank lines."""
    return "\n\n".join(b["text"] for b in record["blocks"] if b["kind"] != "arabic")


def iter_pages(pdf_path: Path, page_range: Optional[Tuple[int, int]] = None) -> Iterator[dict]:
    with pymupdf.open(pdf_path) as doc:
        first, last = page_range or (1, doc.page_count)
        for number in range(first, min(last, doc.page_count) + 1):
            page = doc[number - 1]
            yield {"page": number, "blocks": blocks_from_page_dict(page.get_text("dict"))}


def extract(pdf_path: Path, page_range: Optional[Tuple[int, int]] = None) -> Tuple[List[dict], int]:
    """Extract all pages and repair alternately encoded paragraphs.
    Returns (records, number_of_repaired_blocks)."""
    records = list(iter_pages(pdf_path, page_range))
    text_blocks = [b for r in records for b in r["blocks"] if b["kind"] != "arabic"]
    vocab = build_vocabulary(b["text"] for b in text_blocks)
    repaired = 0
    for b in text_blocks:
        fixed = repair_alt_encoding(b["text"], vocab)
        if fixed != b["text"]:
            b["text"] = fixed
            repaired += 1
    return records, repaired


def unknown_fonts(pdf_path: Path, page_range: Optional[Tuple[int, int]] = None) -> Counter:
    """Characters per font that is neither in ENGLISH_FONTS nor ARABIC_FONTS."""
    seen: Counter = Counter()
    with pymupdf.open(pdf_path) as doc:
        first, last = page_range or (1, doc.page_count)
        for number in range(first, min(last, doc.page_count) + 1):
            for block in doc[number - 1].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if span["font"] not in ENGLISH_FONTS | ARABIC_FONTS:
                            seen[span["font"]] += len(span["text"].strip())
    return seen


def parse_range(value: str) -> Tuple[int, int]:
    first, _, last = value.partition("-")
    return int(first), int(last or first)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pdf", type=Path, default=paths.SOURCE_PDF)
    parser.add_argument("--out", type=Path, default=paths.PAGES_JSONL)
    parser.add_argument("--pages", type=parse_range, help="e.g. 1-200 (default: all)")
    args = parser.parse_args(argv)

    start = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".jsonl.tmp")
    records, repaired = extract(args.pdf, args.pages)
    kinds: Counter = Counter()
    scripts: Counter = Counter()
    pages = english_chars = 0
    empty_pages: List[int] = []
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            pages += 1
            for b in record["blocks"]:
                kinds[b["kind"]] += 1
                if b["kind"] == "arabic":
                    scripts[b["script"]] += 1
                else:
                    english_chars += len(b["text"])
            if not any(b["kind"] != "arabic" for b in record["blocks"]):
                empty_pages.append(record["page"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    print(f"Wrote {pages} pages to {args.out} in {time.time() - start:.0f}s")
    print(f"  blocks: {dict(kinds)}  arabic: {dict(scripts)}")
    print(f"  English characters: {english_chars:,}  repaired blocks: {repaired}")
    print(f"  pages without English text: {len(empty_pages)} {empty_pages[:20]}")
    others = unknown_fonts(args.pdf, args.pages)
    if others:
        print(f"  WARNING unclassified fonts (treated as English): {dict(others)}")


if __name__ == "__main__":
    main()
