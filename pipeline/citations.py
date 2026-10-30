"""Step 3: the book's own cross-references as a graph between verse sections.

Ibn Kathir constantly quotes other verses, each followed by its reference: "(7:199-200)",
"(8:62,63)", or bare after a quotation "...united them) 8:62,63". Every reference is parsed,
checked against the verse counts, and resolved to the section that comments on that verse.
References next to a hadith source ("An-Nasa'i (Al-Kubra 5:12)") are volume:page, not verses,
and are skipped.

Output (one row per reference):
    {"passage_id": "2:34#2", "section_id": "2:34", "surah": 12, "ayah_start": 100,
     "ayah_end": 100, "raw": "12:100", "targets": ["12:99-101"], "self": false}

Usage:
    python -m pipeline.citations [--passages PATH] [--out PATH]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline import paths
from pipeline.quran import is_valid_verse

REF_RE = re.compile(r"(?<![\d:])(\d{1,3}):(\d{1,3})((?:\s*[-,]\s*\d{1,3}(?![\d:]))*)(?![\d:])")
_PART_RE = re.compile(r"([-,])\s*(\d{1,3})")
_HADITH_SOURCE_RE = re.compile(
    r"(?:Muslim|Bukhari|Ahmad|Kubra|Musnad|Sunan|Fath|Mustadrak|Tabari|Majah|Dawud|Tirmidhi|"
    r"Nasa'?i|Hakim|Volume|Vol\.|Darimi|Bayhaqi|Hibban|Tabarani)[^().\n]{0,12}\(?$")

Ref = Tuple[int, int, int]  # (surah, ayah_start, ayah_end)


def parse_refs(text: str) -> Tuple[List[Tuple[Ref, str]], Dict[str, int]]:
    """Qur'an references in `text` as ((surah, start, end), raw) pairs.
    "2:30-33" is one range; "8:62,63" is two single verses. Also returns counts of
    skipped matches ("invalid" verse numbers, "source" hadith volume:page)."""
    refs: List[Tuple[Ref, str]] = []
    skipped = {"invalid": 0, "source": 0}
    for m in REF_RE.finditer(text):
        if _HADITH_SOURCE_RE.search(text[max(0, m.start() - 25):m.start()]):
            skipped["source"] += 1
            continue
        surah, start = int(m.group(1)), int(m.group(2))
        spans = [[start, start]]
        for sep, n in _PART_RE.findall(m.group(3)):
            if sep == "-":
                spans[-1][1] = int(n)
            else:
                spans.append([int(n), int(n)])
        raw = m.group(0).strip()
        for a, b in spans:
            if a <= b and is_valid_verse(surah, a) and is_valid_verse(surah, b):
                refs.append(((surah, a, b), raw))
            else:
                skipped["invalid"] += 1
    return refs, skipped


def verse_index(passages: List[dict]) -> Dict[Tuple[int, int], str]:
    """(surah, ayah) -> id of the section whose range contains it."""
    index: Dict[Tuple[int, int], str] = {}
    for p in passages:
        if p["ayah_start"] > 0:
            for ayah in range(p["ayah_start"], p["ayah_end"] + 1):
                index[(p["surah"], ayah)] = p["section_id"]
    return index


def citations(passages: List[dict]) -> Tuple[List[dict], Counter]:
    index = verse_index(passages)
    rows: List[dict] = []
    stats: Counter = Counter()
    for p in passages:
        refs, skipped = parse_refs(p["text"])
        stats.update(skipped)
        for (surah, a, b), raw in refs:
            targets = sorted({index[(surah, v)] for v in range(a, b + 1) if (surah, v) in index},
                             key=lambda sid: int(sid.split(":")[1].split("-")[0]))
            rows.append({
                "passage_id": p["id"],
                "section_id": p["section_id"],
                "surah": surah,
                "ayah_start": a,
                "ayah_end": b,
                "raw": raw,
                "targets": targets,
                "self": targets == [p["section_id"]],
            })
            stats["refs"] += 1
            stats["resolved"] += bool(targets)
            stats["self"] += targets == [p["section_id"]]
    return rows, stats


def section_edges(rows: List[dict]) -> Counter:
    """(from section, to section) -> number of references, excluding self references."""
    edges: Counter = Counter()
    for r in rows:
        for t in r["targets"]:
            if t != r["section_id"]:
                edges[(r["section_id"], t)] += 1
    return edges


def load_passages(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--passages", type=Path, default=paths.PASSAGES_JSONL)
    parser.add_argument("--out", type=Path, default=paths.CITATIONS_JSONL)
    args = parser.parse_args(argv)

    rows, stats = citations(load_passages(args.passages))
    tmp = args.out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    edges = section_edges(rows)
    incoming = Counter()
    for (_, dst), n in edges.items():
        incoming[dst] += n
    print(f"Wrote {len(rows)} references to {args.out}")
    print(f"  resolved to a section: {stats['resolved']}/{stats['refs']} "
          f"({stats['resolved'] / max(stats['refs'], 1):.1%}); self references: {stats['self']}")
    print(f"  skipped: {stats['invalid']} invalid verse numbers, {stats['source']} hadith volume:page")
    print(f"  cross-section edges: {len(edges)} distinct, {sum(edges.values())} references")
    print(f"  most cited sections: {incoming.most_common(8)}")


if __name__ == "__main__":
    main()
