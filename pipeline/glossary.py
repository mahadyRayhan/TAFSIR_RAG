"""Step 5: glossary of equivalent search terms, proposed from the book and approved by a person.

Candidates come only from the text itself:
  * gloss    the translators define terms in parentheses: "Iblis (Satan)", "Salah (prayer)",
             "At-Taqwa (piety, righteousness)". Forms with and without the Arabic article
             (As-Salah / Salah) are one candidate.
  * spelling the same word written two ways that differ only by a hyphen, a doubled letter, or
             y/i, w/u ("Quraysh" / "Quraish", "Hellfire" / "Hell-fire").
An English word list (/usr/share/dict/words) is used only to tell transliterated terms from
ordinary English words.

glossary/glossary.csv is the reviewed file. Set `keep` to y for rows to use and edit `terms`
(pipe-separated) freely; rows with an id starting "custom:" can be added by hand. Re-running
the generator refreshes evidence but keeps every `keep` and `terms` you edited and every row
it no longer proposes. Only rows marked keep=y are ever used for search.

Usage:
    python -m pipeline.glossary [--passages PATH] [--out PATH]
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from pipeline import paths
from pipeline.citations import load_passages
from pipeline.index import search_text

WORDS_FILE = Path("/usr/share/dict/words")
FIELDS = ["id", "keep", "terms", "evidence", "example"]
KEEP_VALUES = {"y", "yes", "x", "1", "true"}

MIN_GLOSS_COUNT = 2     # a definition must appear at least twice
MAX_GLOSS_WORDS = 4
MIN_SPELLING_COUNT = 2

_ARTICLE_RE = re.compile(r"^(?:Al|As|At|Ad|An|Ar|Az|Ash|Adh|Ath|Az|Aw)-(?=[A-Za-z`])")
_TERM = r"[A-Z][A-Za-z`'\-]+"
_GLOSS_RE = re.compile(rf"\b({_TERM}(?: {_TERM}){{0,2}}) \(([A-Za-z][A-Za-z ,'\-]{{2,40}})\)")
_PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "their", "you", "we", "i", "me", "us"}
# Parentheticals starting with these continue the sentence or add an honorific; they do not
# define the term: "Umar (bin Al-Khattab)", "Qur'an (or Muhammad)", "(may Allah ...)".
_NOT_A_DEFINITION_START = {"or", "of", "in", "at", "from", "who", "that", "after", "before", "to",
                           "with", "by", "bin", "ibn", "may", "peace", "up", "through", "over",
                           "and", "as", "for", "on", "which", "when", "saying", "mentioned"}


def load_english(words_file: Path = WORDS_FILE) -> Set[str]:
    if not words_file.exists():
        return set()
    with words_file.open() as f:
        return {w.strip().lower() for w in f}


def is_english(word: str, english: Set[str]) -> bool:
    w = word.lower().strip("'`-")
    if w in english:
        return True
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("d", ""),
                         ("ing", ""), ("ing", "e"), ("ly", "")):
        if w.endswith(suffix) and w[: -len(suffix)] + repl in english:
            return True
    return False


def term_key(term: str) -> str:
    """Grouping key: article dropped, ayn/apostrophes dropped, lower case."""
    return search_text(_ARTICLE_RE.sub("", term)).lower()


def orthographic_key(word: str) -> str:
    """Key under which pure spelling variants collide."""
    w = term_key(word).replace("-", "")
    w = w.replace("y", "i").replace("w", "u")
    return re.sub(r"(.)\1+", r"\1", w)


def clean_gloss(gloss: str, english: Set[str]) -> List[str]:
    """English definitions inside one parenthetical ("piety, righteousness" -> two)."""
    if gloss.split()[0].lower() in _NOT_A_DEFINITION_START:
        return []
    out = []
    for g in re.split(r",\s*|\s+or\s+", gloss.strip(" ,")):
        g = re.sub(r"^(?:the|a|an)\s+", "", g.strip(), flags=re.I)
        words = g.split()
        single_term = len(words) == 1 and re.fullmatch(_TERM, g)  # "Iblis (Shaytan)"
        if len(g) >= 3 and g.lower() not in _PRONOUNS and len(words) <= MAX_GLOSS_WORDS \
                and (single_term or all(is_english(w, english) for w in words)):
            out.append(g)
    return out


def trim_term(term: str, english: Set[str]) -> str:
    """Drop leading English words captured with the term: "Except Iblis" -> "Iblis"."""
    words = term.split()
    while len(words) > 1 and is_english(words[0], english):
        words.pop(0)
    return " ".join(words)


def mine_glosses(passages: Iterable[dict], english: Set[str]) -> Dict[str, dict]:
    found: Dict[str, dict] = defaultdict(lambda: {"forms": Counter(), "glosses": Counter(), "example": {}})
    for p in passages:
        for m in _GLOSS_RE.finditer(p["text"]):
            term, gloss = trim_term(m.group(1), english), m.group(2)
            if is_english(term.split()[-1], english):
                continue
            for g in clean_gloss(gloss, english):
                entry = found[term_key(term)]
                entry["forms"][term] += 1
                entry["glosses"][g] += 1
                entry["example"].setdefault(g, f"{p['id']}: {m.group(0)}")
    return found


def term_frequencies(passages: Iterable[dict]) -> Counter:
    freq: Counter = Counter()
    for p in passages:
        for w in re.findall(r"\b[A-Z][A-Za-z`'\-]{2,}", p["text"]):
            freq[term_key(w.rstrip("'-"))] += 1
    return freq


def mine_spellings(passages: Iterable[dict], english: Set[str]) -> List[Counter]:
    forms: Dict[str, Counter] = defaultdict(Counter)
    for p in passages:
        for w in re.findall(r"\b[A-Za-z][A-Za-z`'\-]{2,}", p["text"]):
            w = w.strip("'-")
            if w[:1].isupper() or "-" in w:
                forms[orthographic_key(w)][term_key(w)] += 1
    groups = []
    for variants in forms.values():
        variants = Counter({v: n for v, n in variants.items() if n >= MIN_SPELLING_COUNT})
        # Skip groups containing an ordinary English word ("said" / "Sayyid", "man" / "Ma`an").
        if len(variants) > 1 and not any(is_english(v.replace("-", ""), english) for v in variants):
            groups.append(variants)
    return groups


def candidates(passages: List[dict], english: Set[str]) -> List[dict]:
    freq = term_frequencies(passages)
    rows = []
    for key, e in mine_glosses(passages, english).items():
        glosses = [(g, n) for g, n in e["glosses"].most_common() if n >= MIN_GLOSS_COUNT]
        if not glosses:
            continue
        forms = [f for f, _ in e["forms"].most_common()]
        terms = list(dict.fromkeys(forms + [g for g, _ in glosses]))
        rows.append({
            "id": f"gloss:{key}",
            "keep": "",
            "terms": " | ".join(terms),
            "evidence": f"term in book x{freq[key]}; defined as " +
                        "; ".join(f"'{g}' x{n}" for g, n in glosses),
            "example": e["example"][glosses[0][0]],
            "_rank": freq[key],
        })
    for variants in mine_spellings(passages, english):
        ordered = [v for v, _ in variants.most_common()]
        rows.append({
            "id": f"spelling:{ordered[0]}",
            "keep": "",
            "terms": " | ".join(ordered),
            "evidence": "spelling variants: " + "; ".join(f"{v} x{n}" for v, n in variants.most_common()),
            "example": "",
            "_rank": sum(variants.values()),
        })
    rows.sort(key=lambda r: -r.pop("_rank"))
    return rows


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_with_existing(new_rows: List[dict], existing: List[dict]) -> List[dict]:
    """Keep the reviewer's `keep`/`terms` for rows that are proposed again, and keep rows
    that are no longer proposed (hand-added or from an earlier run) at the end."""
    old = {r["id"]: r for r in existing}
    merged = []
    for r in new_rows:
        if r["id"] in old:
            prev = old.pop(r["id"])
            r["keep"] = prev.get("keep", "")
            if prev.get("keep") or prev.get("terms") != r["terms"]:
                r["terms"] = prev.get("terms", r["terms"])
        merged.append(r)
    merged.extend(old.values())
    return merged


def write_rows(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_term(term: str) -> str:
    """Form used for matching: article dropped, ayn/apostrophes dropped, lower case."""
    return " ".join(term_key(w) for w in term.split())


def load_glossary(path: Path = paths.GLOSSARY_CSV) -> List[Set[str]]:
    """Approved alias groups (rows with keep=y). Groups sharing a term are merged."""
    groups: List[Set[str]] = []
    for r in read_rows(path):
        if (r.get("keep") or "").strip().lower() not in KEEP_VALUES:
            continue
        terms = {normalize_term(t) for t in (r.get("terms") or "").split("|") if t.strip()}
        overlapping = [g for g in groups if g & terms]
        for g in overlapping:
            terms |= g
            groups.remove(g)
        if terms:
            groups.append(terms)
    return groups


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--passages", type=Path, default=paths.PASSAGES_JSONL)
    parser.add_argument("--out", type=Path, default=paths.GLOSSARY_CSV)
    args = parser.parse_args(argv)

    english = load_english()
    if not english:
        print(f"WARNING: {WORDS_FILE} not found; English words cannot be filtered out")
    rows = merge_with_existing(candidates(load_passages(args.passages), english), read_rows(args.out))
    write_rows(rows, args.out)
    kinds = Counter(r["id"].split(":")[0] for r in rows)
    approved = sum((r.get("keep") or "").strip().lower() in KEEP_VALUES for r in rows)
    print(f"Wrote {len(rows)} rows to {args.out}: {dict(kinds)}; approved (keep=y): {approved}")


if __name__ == "__main__":
    main()
