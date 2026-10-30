"""Surah names, read from the book, so queries like "Surah Al-Baqarah verse 34" can be parsed.

Names come from the book's own lines:
  opening   "The Tafsir of Surah Alam Nashrah (Surat Ash-Sharh) (Chapter - 94)"
  closing   "This is the end of the Tafsir of Surat Al-Hujurat."
  heading   "Surat Al-Baqarah"
They are written to glossary/surah_names.csv (surah, names, evidence) for review, like the
glossary: re-running keeps any `names` a person edited, so aliases such as "The Cow" can be
added by hand. Surahs 113 and 114 have no line of their own; the book introduces them together
as "The Tafsir of Al-Mu`awwidhatayn (Surahs Al-Falaq and An-Nas)", so their rows are filled
from that line and their first verses ("Lord of Al-Falaq", "Lord of An-Nas").

Usage:
    python -m pipeline.surahs [--passages PATH] [--out PATH]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from pipeline import paths
from pipeline.citations import load_passages
from pipeline.glossary import orthographic_key
from pipeline.quran import SURAH_COUNT

FIELDS = ["surah", "names", "evidence"]

_NAME = r"`?[A-Z][A-Za-z`'\-]*(?: [A-Za-z`'\-]+){0,3}?"  # may start with an ayn: "`Abasa"
_OPENING_RE = re.compile(
    rf"^(?:The )?Tafsir of Sura[ht] ({_NAME})\s*(?:\((?:Surat |Surah )?({_NAME})\))?\s*(?:\(Chapter[^)]*\))?\s*$")
_CLOSING_RE = re.compile(rf"end of the Tafsir of Sura[ht] ({_NAME})[.,]")
_HEADING_RE = re.compile(rf"^Sura[ht] ({_NAME})$")

MANUAL = {
    113: ("Al-Falaq", "Tafsir of Al-Mu`awwidhatayn (Surahs Al-Falaq and An-Nas) [112 tail]; "
                      "113:1 'Lord of Al-Falaq'"),
    114: ("An-Nas", "Tafsir of Al-Mu`awwidhatayn (Surahs Al-Falaq and An-Nas) [112 tail]; "
                    "114:1 'Lord of An-Nas'"),
}


def surah_key(name: str) -> str:
    """Matching key: "Surat Al-Fatihah", "Al-Fatiha" and "fatiha" all give "fatiha"."""
    name = re.sub(r"^(?:the )?sura[ht]?\s+", "", name.strip(), flags=re.I)
    name = re.sub(r"[`'\u2019]", "", name)  # "Nisa'" = "An-Nisa", "`Abasa" = "Abasa"
    # The article may be written with a space too: "Al `Imran" = "Al-Imran".
    name = re.sub(r"^(?:al|an|as|at|ad|ar|az|ash|adh|ath)[\s\-]+(?=\S)", "", name, flags=re.I)
    return re.sub(r"\s+", "", orthographic_key(name)).rstrip("h")


def mine_names(passages: List[dict]) -> Dict[int, Dict[str, str]]:
    """surah -> {name: evidence}"""
    found: Dict[int, Dict[str, str]] = defaultdict(dict)
    for p in passages:
        if p["surah"] < 1:
            continue
        lines = list(p["headings"])
        if p["kind"] == "intro":
            lines += p["text"].split("\n\n")[:3]
        for line in lines:
            line = line.strip()
            m = _OPENING_RE.match(line) or _HEADING_RE.match(line)
            if m:
                for name in filter(None, m.groups()):
                    found[p["surah"]].setdefault(name.strip(), f"{p['id']}: {line}")
        for m in _CLOSING_RE.finditer(p["text"]):
            found[p["surah"]].setdefault(m.group(1).strip(), f"{p['id']}: ...{m.group(0)}")
    return found


def build_rows(passages: List[dict]) -> List[dict]:
    found = mine_names(passages)
    rows = []
    for n in range(1, SURAH_COUNT + 1):
        names = dict(found.get(n, {}))
        if n in MANUAL and not names:
            name, evidence = MANUAL[n]
            names[name] = evidence
        unique: Dict[str, str] = {}
        for name, evidence in names.items():
            unique.setdefault(surah_key(name), name) if surah_key(name) else None
        rows.append({
            "surah": str(n),
            "names": " | ".join(unique.values()),
            "evidence": " || ".join(dict.fromkeys(names[v] for v in unique.values())) or "not found in book",
        })
    return rows


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_with_existing(new_rows: List[dict], existing: List[dict]) -> List[dict]:
    """A reviewer's `names` win over regenerated ones."""
    old = {r["surah"]: r for r in existing}
    for r in new_rows:
        prev = old.get(r["surah"])
        if prev and prev.get("names") and prev["names"] != r["names"]:
            r["names"] = prev["names"]
    return new_rows


def load_surah_names(path: Path = paths.SURAH_NAMES_CSV) -> Dict[str, int]:
    """Matching key -> surah number, for every name in the reviewed file."""
    table: Dict[str, int] = {}
    for r in read_rows(path):
        for name in (r.get("names") or "").split("|"):
            if surah_key(name):
                table[surah_key(name)] = int(r["surah"])
    return table


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--passages", type=Path, default=paths.PASSAGES_JSONL)
    parser.add_argument("--out", type=Path, default=paths.SURAH_NAMES_CSV)
    args = parser.parse_args(argv)

    rows = merge_with_existing(build_rows(load_passages(args.passages)), read_rows(args.out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    missing = [r["surah"] for r in rows if not r["names"]]
    print(f"Wrote {len(rows)} surahs to {args.out}; without a name: {missing or 'none'}")


if __name__ == "__main__":
    main()
