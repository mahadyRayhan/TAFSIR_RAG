"""Step 2: split the book into verse sections and topic passages.

Structure of the book (found by inspection, see README):
  * each surah opens with a "(Chapter N)" heading (110 of 114 surahs); surahs 2, 31, 113
    and 114 lack it and are found where verse numbering restarts;
  * a block of verse translations "(30. And (remember) ...) (31. ...)" introduces a group of
    verses, and the commentary that follows belongs to it until the next such block;
  * Ibn Kathir's bold topic headings ("The Prostration was before Adam ...") divide long
    commentaries.

Every verse section gets the id "S:A-B" from the verse numbers printed in the book. A
section's range is extended up to the verse before the next section, because commentary on
verses whose translation does not start a block still sits inside it. Long sections are split
into passages at topic headings, then at paragraph breaks; passages keep the section's id.

Usage:
    python -m pipeline.sections [--pages PATH] [--out PATH]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pipeline import paths
from pipeline.extract_pages import page_text
from pipeline.quran import SURAH_COUNT, verse_count

MAX_PASSAGE_CHARS = 6000
MIN_PASSAGE_CHARS = 600

_CHAPTER_HEADING_RE = re.compile(r"(?:^|\()\s*Chapter\s*-?\s*(\d{1,3})\s*\)")
_ANCHORED_TITLE_RE = re.compile(r"^The Tafsir of Sura", re.I)
# Also matches "The Tafsir of Al-Mu`awwidhatayn (Surahs Al-Falaq and An-Nas)".
_RESET_TITLE_RE = re.compile(r"^(?:The Tafsir of\b|Sura[th]?\b)", re.I)
# "(30. And ...)" and the occasional "(1.Ta Sin ...)"; the lookahead rules out "(1.5".
_VERSE_HEADER_RE = re.compile(r"\(\d{1,3}\.\s*(?=[^\d\s])")
_VERSE_MARKER_RE = re.compile(r"(?:^|(?<=\))|(?<=\) ))\((\d{1,3})\.\s*(?=[^\d\s])")

Block = Tuple[int, dict]  # (page number, block from pages.jsonl)


def chapter_anchor(block: dict) -> Optional[int]:
    """Surah number announced by a "(Chapter N)" heading, else None. Only headings and the
    "The Tafsir of Surat ... (Chapter N)" line count: body text also mentions chapters."""
    text = block.get("text", "")
    if block["kind"] == "heading" or (block["kind"] == "body" and text.startswith("The Tafsir of Sur")):
        m = _CHAPTER_HEADING_RE.search(text)
        if m and 1 <= int(m.group(1)) <= SURAH_COUNT:
            return int(m.group(1))
    return None


def verse_markers(text: str) -> List[int]:
    """Verse numbers of a translation block "(1. ...) (2. ...)"; empty for other text."""
    if not _VERSE_HEADER_RE.match(text):
        return []
    return [int(m.group(1)) for m in _VERSE_MARKER_RE.finditer(text)]


def _new_section(surah: int, kind: str, start: int = 0) -> dict:
    return {"surah": surah, "kind": kind, "ayah_start": start, "ayah_end": start,
            "marked": [], "blocks": [], "header_open": kind == "verses"}


def _carve_intro(cur: dict, surah: int, title_re=_ANCHORED_TITLE_RE) -> dict:
    """Move the last surah title heading (and what follows it) from `cur` into a new intro
    section for `surah`. Used where a "(Chapter N)" anchor follows the title heading."""
    intro = _new_section(surah, "intro")
    for i in range(len(cur["blocks"]) - 1, -1, -1):
        _, b = cur["blocks"][i]
        if b["kind"] == "heading" and title_re.match(b["text"]):
            intro["blocks"] = cur["blocks"][i:]
            del cur["blocks"][i:]
            break
    return intro


def _carve_intro_after_verses(cur: dict, surah: int) -> dict:
    """For a surah found by the numbering restart (no "(Chapter N)" anchor): its intro starts
    at the first surah title heading after the last verse block of `cur`. The intro can hold
    several title-like headings ("Surat Al-Baqarah" ... "Surat Al-Baqarah was revealed in
    Al-Madinah"), so the earliest one is the boundary."""
    intro = _new_section(surah, "intro")
    blocks = cur["blocks"]
    last_verse = max((i for i, (_, b) in enumerate(blocks) if b["kind"] == "body" and verse_markers(b["text"])),
                     default=-1)
    for i in range(last_verse + 1, len(blocks)):
        _, b = blocks[i]
        if b["kind"] == "heading" and _RESET_TITLE_RE.match(b["text"]):
            intro["blocks"] = blocks[i:]
            del blocks[i:]
            break
    return intro


def segment(blocks: Iterable[Block]) -> Tuple[List[dict], Dict[str, list]]:
    """Group page blocks into front matter, surah intros and verse sections."""
    blocks = list(blocks)
    anchored = {a for _, b in blocks if (a := chapter_anchor(b))}
    sections: List[dict] = []
    report: Dict[str, list] = {"rejected_headers": [], "duplicates": [], "resets": []}

    cur = _new_section(0, "front")
    sections.append(cur)
    surah, prev_end = 0, 0
    skipped: Optional[dict] = None  # collects a duplicated surah copy, which is dropped

    for page, b in blocks:
        anchor = chapter_anchor(b)
        if anchor is not None:
            if anchor <= surah:
                _carve_intro(cur, surah)  # drop the duplicate's title heading
                skipped = _new_section(surah, "duplicate")
                report["duplicates"].append((page, anchor))
                continue
            # The next surah's title heading may sit at the end of a skipped duplicate.
            cur = _carve_intro(skipped or cur, anchor)
            skipped = None
            surah, prev_end = anchor, 0
            sections.append(cur)
            cur["blocks"].append((page, b))
            continue
        if skipped is not None:
            skipped["blocks"].append((page, b))
            continue

        markers = verse_markers(b["text"]) if b["kind"] == "body" else []
        if markers and surah >= 1 and markers[0] <= 3 and markers[0] <= prev_end \
                and surah + 1 not in anchored and surah < SURAH_COUNT:
            surah, prev_end = surah + 1, 0
            report["resets"].append((page, surah))
            cur = _carve_intro_after_verses(cur, surah)
            sections.append(cur)

        valid: List[int] = []
        for m in markers:
            if m > (valid[-1] if valid else prev_end) and surah >= 1 and m <= verse_count(surah):
                valid.append(m)
        if markers and not valid:
            report["rejected_headers"].append((page, surah, markers[:3]))

        if valid:
            if not (cur["kind"] == "verses" and cur["header_open"]):
                cur = _new_section(surah, "verses", valid[0])
                sections.append(cur)
            cur["marked"].extend(valid)
            cur["ayah_end"] = valid[-1]
            prev_end = valid[-1]
        elif b["kind"] != "arabic":
            cur["header_open"] = False
        cur["blocks"].append((page, b))

    sections = [s for s in sections if any(b["kind"] != "arabic" for _, b in s["blocks"])]
    _extend_ranges(sections)
    return sections, report


def _extend_ranges(sections: List[dict]) -> None:
    """Stretch each verse section to the verse before the next one in its surah (the last
    one to the end of the surah). When a surah's first numbered verse block starts after
    verse 1, the commentary on the earlier verses sits in the intro (e.g. 2:1 "Alif Lam Mim"
    under "The Discussion of the Individual Letters"), so the intro takes that range."""
    by_surah: Dict[int, List[dict]] = {}
    intros: Dict[int, dict] = {}
    for s in sections:
        if s["kind"] == "verses":
            by_surah.setdefault(s["surah"], []).append(s)
        elif s["kind"] == "intro":
            intros.setdefault(s["surah"], s)
    for surah, group in by_surah.items():
        for s, nxt in zip(group, group[1:] + [None]):
            limit = nxt["ayah_start"] - 1 if nxt else verse_count(surah)
            s["ayah_end"] = max(s["ayah_end"], limit)
        first = group[0]["ayah_start"]
        if first > 1 and surah in intros:
            intros[surah]["ayah_start"], intros[surah]["ayah_end"] = 1, first - 1


def section_id(s: dict) -> str:
    if s["ayah_start"] == 0:
        return f"{s['surah']}:0"
    a, b = s["ayah_start"], s["ayah_end"]
    return f"{s['surah']}:{a}" if a == b else f"{s['surah']}:{a}-{b}"


def _text_len(blocks: List[Block]) -> int:
    return sum(len(b.get("text", "")) for _, b in blocks)


def split_passages(blocks: List[Block]) -> List[List[Block]]:
    """Split a section at topic headings, merge parts that are too small, and split parts
    that are still too large at paragraph boundaries."""
    parts: List[List[Block]] = [[]]
    for pb in blocks:
        # A heading opens a new part, unless the current part so far holds only headings
        # (headings that wrap across two PDF blocks arrive as consecutive heading blocks).
        if pb[1]["kind"] == "heading" and any(b["kind"] == "body" for _, b in parts[-1]):
            parts.append([])
        parts[-1].append(pb)

    merged: List[List[Block]] = []
    for part in parts:
        if merged and (_text_len(part) < MIN_PASSAGE_CHARS or _text_len(merged[-1]) < MIN_PASSAGE_CHARS) \
                and _text_len(merged[-1]) + _text_len(part) <= MAX_PASSAGE_CHARS:
            merged[-1].extend(part)
        else:
            merged.append(part)

    out: List[List[Block]] = []
    for part in merged:
        total = _text_len(part)
        if total <= MAX_PASSAGE_CHARS:
            out.append(part)
            continue
        pieces = -(-total // MAX_PASSAGE_CHARS)
        target = total / pieces
        chunk: List[Block] = []
        for pb in part:
            if chunk and _text_len(chunk) >= target and _text_len(chunk) + len(pb[1].get("text", "")) > target:
                out.append(chunk)
                chunk = []
            chunk.append(pb)
        if chunk:
            out.append(chunk)
    return [p for p in out if _text_len(p) > 0]


_WRAPPED_HEADING_END_RE = re.compile(
    r"\b(?:the|of|and|in|to|a|an|for|by|with|from|on|that|who|which|his|her|their)$", re.I)


def is_wrapped_heading(first: str, second: str) -> bool:
    """True when `second` continues `first` ("... preceded it in the" + "primary Mushaf")."""
    return bool(_WRAPPED_HEADING_END_RE.search(first)) or second[:1].islower()


def headings_of(blocks: List[Block]) -> List[str]:
    """Heading texts, joining consecutive heading blocks that are one wrapped heading."""
    out: List[str] = []
    prev_heading = False
    for _, b in blocks:
        if b["kind"] == "heading":
            if prev_heading and is_wrapped_heading(out[-1], b["text"]):
                out[-1] += " " + b["text"]
            else:
                out.append(b["text"])
        if b["kind"] != "arabic":
            prev_heading = b["kind"] == "heading"
    return out


def passages(sections: List[dict]) -> List[dict]:
    records = []
    for s in sections:
        sid = section_id(s)
        section_headings = headings_of(s["blocks"])
        parts = split_passages(s["blocks"])
        for i, part in enumerate(parts, 1):
            headings = headings_of(part)
            pages = [p for p, _ in part]
            records.append({
                "id": sid if len(parts) == 1 else f"{sid}#{i}",
                "section_id": sid,
                "kind": s["kind"],
                "surah": s["surah"],
                "ayah_start": s["ayah_start"],
                "ayah_end": s["ayah_end"],
                "marked_verses": s["marked"],
                "part": i,
                "parts": len(parts),
                "title": (headings or section_headings or [None])[0],
                "headings": headings,
                "page_start": min(pages),
                "page_end": max(pages),
                "text": page_text({"blocks": [b for _, b in part]}),
            })
    return records


def coverage(sections: List[dict]) -> Dict[int, dict]:
    """Per surah: verses covered by some section range, and verses with a printed marker."""
    out: Dict[int, dict] = {n: {"covered": set(), "marked": set()} for n in range(1, SURAH_COUNT + 1)}
    for s in sections:
        if s["ayah_start"] > 0:
            out[s["surah"]]["covered"].update(range(s["ayah_start"], s["ayah_end"] + 1))
            out[s["surah"]]["marked"].update(s["marked"])
    return out


def load_blocks(pages_path: Path) -> List[Block]:
    with pages_path.open(encoding="utf-8") as f:
        return [(r["page"], b) for r in map(json.loads, f) for b in r["blocks"]]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pages", type=Path, default=paths.PAGES_JSONL)
    parser.add_argument("--out", type=Path, default=paths.PASSAGES_JSONL)
    args = parser.parse_args(argv)

    sections, report = segment(load_blocks(args.pages))
    records = passages(sections)
    tmp = args.out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    kinds = Counter(s["kind"] for s in sections)
    sizes = sorted(len(r["text"]) for r in records)
    cov = coverage(sections)
    total = sum(verse_count(n) for n in cov)
    covered = sum(len(c["covered"]) for c in cov.values())
    marked = sum(len(c["marked"]) for c in cov.values())
    print(f"Wrote {len(records)} passages from {len(sections)} sections to {args.out}")
    print(f"  sections: {dict(kinds)}  surahs with verse sections: "
          f"{len({s['surah'] for s in sections if s['kind'] == 'verses'})}/{SURAH_COUNT}")
    print(f"  passage chars: median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"  verses covered by a section: {covered}/{total} ({covered / total:.1%}); "
          f"with a printed marker: {marked}/{total} ({marked / total:.1%})")
    uncovered = {n: sorted(set(range(1, verse_count(n) + 1)) - c["covered"]) for n, c in cov.items()}
    uncovered = {n: v for n, v in uncovered.items() if v}
    print(f"  surahs with uncovered verses: {len(uncovered)} {dict(list(uncovered.items())[:10])}")
    print(f"  surahs found by numbering restart: {report['resets']}")
    print(f"  duplicate chapter headings skipped: {report['duplicates']}")
    print(f"  verse blocks rejected (out of order): {len(report['rejected_headers'])} "
          f"{report['rejected_headers'][:8]}")


if __name__ == "__main__":
    main()
