"""Step 6: turn a question into verse references and search concepts, without any model.

    parse_query("From Ibn Kathir's explanation of the story of Adam and Iblis (Satan)")
      verses:   []
      concepts: story | adam | iblis (+ shaytan, satan, devil from the approved glossary)

    parse_query("What does Surah Al-Baqarah verse 34 say about the angels?")
      verses:   [(2, 34, 34)]
      concepts: angels

Verse references are read from "2:34", "2:30-39", "Surah Al-Baqarah verse 34",
"chapter 18 verse 50" and "verses 30 to 39 of Al-Baqarah"; "Surah Yusuf" alone refers to the
whole surah. A bare name such as "Yusuf" is kept as a concept (the prophet), because many
surahs share their name with a prophet. Surah names come from glossary/surah_names.csv and
aliases from the approved rows of glossary/glossary.csv; both were taken from the book and
reviewed. The words the user typed are kept apart from glossary expansions so retrieval can
weight them higher.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pipeline.citations import REF_RE, parse_refs
from pipeline.glossary import load_glossary, normalize_term
from pipeline.index import search_text
from pipeline.quran import SURAH_COUNT, is_valid_verse, verse_count
from pipeline.surahs import load_surah_names, surah_key

Ref = Tuple[int, int, int]  # (surah, ayah_start, ayah_end)

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it
its itself just me more most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves
""".split())

# Words that frame the question rather than name its subject.
SCAFFOLDING = set("""
explain explains explained explanation explanations according describe describes described
description tell tells mention mentions mentioned say says said discuss discusses discussed
discussion please give provide show based regarding concerning detail details detailed
summary summarize summarise commentary interpretation interpret view views opinion opinions
tafsir meaning means mean teaching teachings teach teaches understand understanding
concept concepts throughout contrast compare comparison versus vs difference differences
""".split())

_IBN_KATHIR_RE = re.compile(r"\bIbn\s+Kathir(?:'s|s)?\b", re.I)
_VERSE_WORD = r"(?:verses?|ayahs?|ayat|ayaat|aya|v\.)"
_SURAH_WORD = r"(?:surahs?|surat|suras?|chapters?)"
_RANGE = r"(\d{1,3})(?:\s*(?:-|–|to|through)\s*(\d{1,3}))?"
_TOKEN_RE = re.compile(r"`?[A-Za-z0-9][A-Za-z0-9`'’\-]*")
_SURAH_THEN_VERSE_RE = re.compile(rf"\s*,?\s*(?:{_VERSE_WORD}\s*{_RANGE}|:\s*{_RANGE})", re.I)
_VERSE_OF_SURAH_RE = re.compile(rf"\b{_VERSE_WORD}\s*{_RANGE}\s+(?:of|in|from)\s+(?:the\s+)?", re.I)
_SURAH_WORD_RE = re.compile(rf"\b{_SURAH_WORD}\s+", re.I)
_BARE_VERSE_RE = re.compile(rf"\b{_VERSE_WORD}\s*{_RANGE}", re.I)
_ARTICLE = r"(?:al|an|as|at|ad|ar|az|ash|adh|ath)"
# "in Al-Baqarah": a surah name with an article after in/of/from. Surahs named after
# prophets (Yusuf, Nuh, Hud) have no article, so "story of Yusuf" stays a concept.
_IN_SURAH_RE = re.compile(rf"\b(?:in|of|from)\s+(?=(?:the\s+)?{_ARTICLE}[-\s]`?[A-Za-z])", re.I)
# Arabic construct names joined to an "al-" word: "Ayat Al-Kursi", "Laylat Al-Qadr".
_NAME_PHRASE_NEXT_RE = re.compile(rf"^{_ARTICLE}-`?[A-Za-z]", re.I)
MAX_NAME_TOKENS = 4


@dataclass
class Concept:
    typed: str                                   # normalized form of what the user wrote
    aliases: List[str] = field(default_factory=list)  # approved glossary equivalents
    proper: bool = False                         # capitalized in the question or in the glossary

    @property
    def terms(self) -> List[str]:
        return [self.typed] + self.aliases


@dataclass
class ParsedQuery:
    text: str
    verses: List[Ref] = field(default_factory=list)   # explicit verse references
    surahs: List[int] = field(default_factory=list)   # whole surahs named without verses
    concepts: List[Concept] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)    # e.g. a verse number out of range


def fts_terms(terms: Sequence[str]) -> str:
    """FTS5 expression matching any of `terms` (multi-word terms as phrases)."""
    words = [re.sub(r'["\s]+', " ", t).strip() for t in terms]
    return "(" + " OR ".join(f'"{w}"' for w in words if w) + ")"


@lru_cache(maxsize=1)
def _default_glossary() -> Tuple[frozenset, ...]:
    return tuple(frozenset(g) for g in load_glossary())


@lru_cache(maxsize=1)
def _default_surah_names() -> Tuple[Tuple[str, int], ...]:
    return tuple(load_surah_names().items())


def _resolve_surah(text: str, start: int, names: Dict[str, int]) -> Optional[Tuple[int, int]]:
    """Surah named by the tokens at text[start:], as (surah, end index), longest name first.
    A number ("chapter 18") is accepted too."""
    tokens = list(_TOKEN_RE.finditer(text, start))[:MAX_NAME_TOKENS]
    if not tokens or text[start:tokens[0].start()].strip():
        return None
    if tokens[0].group().isdigit():
        n = int(tokens[0].group())
        return (n, tokens[0].end()) if 1 <= n <= SURAH_COUNT else None
    for k in range(len(tokens), 0, -1):
        # tokens must be contiguous words ("Al `Imran", "Ya Sin")
        if any(text[a.end():b.start()].strip() not in ("", "`") for a, b in zip(tokens[:k], tokens[1:k])):
            continue
        key = surah_key(text[tokens[0].start():tokens[k - 1].end()])
        if key in names:
            return names[key], tokens[k - 1].end()
    return None


def _verse_range(surah: int, a: str, b: Optional[str], notes: List[str]) -> Optional[Ref]:
    start, end = int(a), int(b or a)
    if start > end:
        start, end = end, start
    if is_valid_verse(surah, start) and is_valid_verse(surah, end):
        return surah, start, end
    notes.append(f"Surah {surah} has {verse_count(surah)} verses; ignored verse {a}{'-' + b if b else ''}")
    return None


def _blank(text: str, a: int, b: int) -> str:
    return text[:a] + " " * (b - a) + text[b:]


def extract_references(text: str, names: Dict[str, int]) -> Tuple[List[Ref], List[int], str, List[str]]:
    """Verse and surah references, plus `text` with them blanked out."""
    verses: List[Ref] = []
    surahs: List[int] = []
    notes: List[str] = []

    for m in REF_RE.finditer(text):                                  # "2:34", "(2:30-39)"
        verses += [r for r, _ in parse_refs(m.group())[0]]
        text = _blank(text, m.start(), m.end())

    for m in list(_VERSE_OF_SURAH_RE.finditer(text)):                # "verses 30 to 39 of ..."
        after = m.end()
        w = _SURAH_WORD_RE.match(text, after)
        found = _resolve_surah(text, w.end() if w else after, names)
        if found:
            surah, end = found
            ref = _verse_range(surah, m.group(1), m.group(2), notes)
            if ref:
                verses.append(ref)
            else:
                surahs.append(surah)
            text = _blank(text, m.start(), end)

    for m in list(_SURAH_WORD_RE.finditer(text)):                    # "Surah Al-Baqarah (verse 34)"
        found = _resolve_surah(text, m.end(), names)
        if not found:
            continue
        surah, end = found
        v = _SURAH_THEN_VERSE_RE.match(text, end)
        if v:
            a, b = (v.group(1), v.group(2)) if v.group(1) else (v.group(3), v.group(4))
            ref = _verse_range(surah, a, b, notes)
            if ref:
                verses.append(ref)
            else:
                surahs.append(surah)
            end = v.end()
        else:
            surahs.append(surah)
        text = _blank(text, m.start(), end)

    for m in list(_BARE_VERSE_RE.finditer(text)):                    # "Al-Baqarah verse 34"
        before = list(_TOKEN_RE.finditer(text[:m.start()]))[-MAX_NAME_TOKENS:]
        for k in range(len(before)):
            found = _resolve_surah(text[:m.start()], before[k].start(), names)
            if found and not text[found[1]:m.start()].strip(" ,"):
                ref = _verse_range(found[0], m.group(1), m.group(2), notes)
                if ref:
                    verses.append(ref)
                text = _blank(text, before[k].start(), m.end())
                break

    for m in list(_IN_SURAH_RE.finditer(text)):                      # "... in Al-Baqarah"
        start = m.end() + (4 if text[m.end():m.end() + 4].lower() == "the " else 0)
        found = _resolve_surah(text, start, names)
        if found and found[0] not in surahs:
            surahs.append(found[0])
            text = _blank(text, m.start(), found[1])

    return list(dict.fromkeys(verses)), list(dict.fromkeys(surahs)), text, notes


def extract_concepts(text: str, groups: Sequence[Set[str]]) -> List[Concept]:
    term_group: Dict[str, int] = {t: i for i, g in enumerate(groups) for t in g}
    longest = max((len(t.split()) for t in term_group), default=1)
    tokens = [m.group() for m in _TOKEN_RE.finditer(_IBN_KATHIR_RE.sub(" ", text))]
    tokens = [re.sub(r"(?:'s|’s|['’`])$", "", t) for t in tokens]

    # Name phrases ("Ayat Al-Kursi") become one token, searched as a phrase with the article.
    merged: List[str] = []
    for t in tokens:
        if merged and _NAME_PHRASE_NEXT_RE.match(t) and merged[-1][:1].isupper() \
                and merged[-1].lower() not in STOPWORDS and normalize_term(t) not in term_group:
            merged[-1] = f"{merged[-1]} {t}"
        else:
            merged.append(t)
    tokens = merged

    concepts: List[Concept] = []
    used_groups: Set[int] = set()
    seen: Set[str] = set()

    def add(norm: str, capitalized: bool) -> None:
        group = term_group.get(norm)
        if norm in seen or (group is not None and group in used_groups):
            return
        aliases = sorted(groups[group] - {norm}) if group is not None else []
        concepts.append(Concept(norm, aliases, proper=group is not None or capitalized))
        seen.add(norm)
        if group is not None:
            used_groups.add(group)

    i = 0
    while i < len(tokens):
        if " " in tokens[i]:                                   # name phrase
            add(search_text(tokens[i]).lower(), True)
            i += 1
            continue
        # Longest approved multi-word glossary term starting here ("waiting period").
        n = next((n for n in range(min(longest, len(tokens) - i), 1, -1)
                  if normalize_term(" ".join(tokens[i:i + n])) in term_group), 1)
        norm = normalize_term(" ".join(tokens[i:i + n]))
        if n > 1 or not (norm in STOPWORDS or norm in SCAFFOLDING or len(norm) < 2 or norm.isdigit()):
            add(norm, i > 0 and tokens[i][:1].isupper())
        i += n
    return concepts


def parse_query(text: str, glossary: Optional[Sequence[Set[str]]] = None,
                surah_names: Optional[Dict[str, int]] = None) -> ParsedQuery:
    groups = [set(g) for g in (glossary if glossary is not None else _default_glossary())]
    names = surah_names if surah_names is not None else dict(_default_surah_names())
    verses, surahs, rest, notes = extract_references(text, names)
    return ParsedQuery(text=text, verses=verses, surahs=surahs,
                       concepts=extract_concepts(rest, groups), notes=notes)
