"""Step 7: retrieve passages for a parsed question by fusing several ranked lists.

A concept is "important" when it appears in at most 20% of the book's passages: "Iblis"
(8%) and "hypocrites" (6%) are the subject of a question, "Allah" (96%) and "people" (53%)
are not. This is measured on the book, not guessed from capitalization.

Ranked lists (each searched with BM25 over the step 4 index, or looked up directly):
  verse              passages of the sections that comment on verses named in the question
                     (or, for "Explain Surah X" alone, the surah's sections in order)
  coverage_typed     passages ranked by the important concepts they contain, each weighted by
                     its rarity in the book, words as typed
  coverage_expanded  the same, counting approved glossary aliases
  concept:<x>        one important concept at a time, expanded: (iblis OR shaytan OR ...)
  any                any concept at all, important or not
  citations          sections the top results cite or are cited by (the book's own
                     references), kept only when they also mention an important concept
  neighbours         the sections just before/after top results in the same surah, when that
                     surah has several top results (a story told over consecutive sections),
                     kept only when they mention an important concept

Reciprocal rank fusion combines them: score = sum(weight / (RRF_K + rank)). Words the user
typed weigh more than glossary aliases. When the question names verses or a surah, keyword
lists are limited to those surahs. Passages scoring under a quarter of the best one are
dropped; the rest are cut to a size budget and returned in the order of the Qur'an, each
with the reasons it was chosen.
"""
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pipeline.index import TITLE_WEIGHT
from rag.query import Concept, ParsedQuery, fts_terms

RRF_K = 60
LIST_LIMIT = 50
WEIGHTS = {
    "verse": 3.0,
    "coverage_typed": 1.0,
    "coverage_expanded": 0.8,
    "concept": 0.5,
    "any": 0.25,
    "citations": 0.4,
    "neighbours": 0.8,
}
IMPORTANT_MAX_SHARE = 0.20  # concepts in more passages than this are general vocabulary
COVERAGE_POOL = 300         # hits per concept considered when counting coverage
COMMON_WORD_FACTOR = 0.5    # weight of uncapitalized, non-glossary words in coverage
COMPARE_CONCEPT_FACTOR = 2.0  # per-concept lists count double for "contrast A with B"
# When the question names verses, those sections are the answer; keyword lists only add
# supporting passages, at this fraction of their usual weight.
VERSE_FOCUS = 0.25
CITATION_SEEDS = 10       # top fused sections whose references are followed
MAX_PASSAGES = 30
MAX_CHARS = 120_000       # ~30k tokens of context for the writer
# Passages scoring below this share of the best passage are padding (e.g. any mention of
# "angels" in Al-Baqarah when the question is about 2:34), not evidence.
MIN_RELATIVE_SCORE = 0.25
SURAH_ONLY_LIMIT = 60     # passages returned for "Explain Surah X" without other words

_VERSE_LINE_RE = re.compile(r"^\(\d{1,3}\.")


@dataclass
class Retrieved:
    id: str
    section_id: str
    surah: int
    ayah_start: int
    ayah_end: int
    title: Optional[str]
    page_start: int
    page_end: int
    text: str
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    verse_text: str = ""     # the section's verse translations "(30. ...) (31. ...)"

    @property
    def citation(self) -> str:
        """How the answer cites this passage: Surah:Ayah, plus the PDF page."""
        if self.ayah_start == 0:
            ref = f"Surah {self.surah}, introduction"
        elif self.ayah_start == self.ayah_end:
            ref = f"{self.surah}:{self.ayah_start}"
        else:
            ref = f"{self.surah}:{self.ayah_start}-{self.ayah_end}"
        pages = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return f"{ref}, {pages}"


@dataclass
class RetrievalResult:
    query: ParsedQuery
    passages: List[Retrieved]
    lists: Dict[str, List[str]]           # every ranked list, for inspection
    notes: List[str] = field(default_factory=list)


def _row(r: sqlite3.Row) -> Retrieved:
    return Retrieved(r["id"], r["section_id"], r["surah"], r["ayah_start"], r["ayah_end"], r["title"],
                     r["page_start"], r["page_end"], r["text"])


def keyword_list(conn: sqlite3.Connection, match: str, surahs: Sequence[int] = (),
                 sections: Sequence[str] = (), limit: int = LIST_LIMIT) -> List[str]:
    """Passage ids for an FTS5 expression, best BM25 first, optionally within surahs/sections."""
    sql = (f"SELECT p.id FROM passages_fts JOIN passages p ON p.rowid = passages_fts.rowid "
           f"WHERE passages_fts MATCH ?")
    args: list = [match]
    if surahs:
        sql += f" AND p.surah IN ({','.join('?' * len(surahs))})"
        args += list(surahs)
    if sections:
        sql += f" AND p.section_id IN ({','.join('?' * len(sections))})"
        args += list(sections)
    sql += f" ORDER BY bm25(passages_fts, {TITLE_WEIGHT}, 1.0) LIMIT ?"
    args.append(limit)
    return [r[0] for r in conn.execute(sql, args)]


def verse_list(conn: sqlite3.Connection, verses: Iterable[Tuple[int, int, int]]) -> List[str]:
    """Passages of every section whose verse range overlaps a requested range, in book order."""
    ids: List[str] = []
    for surah, a, b in verses:
        ids += [r[0] for r in conn.execute(
            "SELECT id FROM passages WHERE surah = ? AND ayah_start > 0 AND ayah_start <= ? "
            "AND ayah_end >= ? ORDER BY rowid", (surah, b, a))]
    return list(dict.fromkeys(ids))


def surah_list(conn: sqlite3.Connection, surahs: Iterable[int], limit: int = SURAH_ONLY_LIMIT) -> List[str]:
    """For "Explain Surah X": the intro's first passage, then the first passage of every verse
    section (verse text and the start of its commentary), then the remaining passages."""
    ids: List[str] = []
    for s in surahs:
        rows = conn.execute("SELECT id, kind, part FROM passages WHERE surah = ? ORDER BY rowid", (s,)).fetchall()
        intro_first = [pid for pid, kind, part in rows if kind == "intro" and part == 1][:1]
        sections_first = [pid for pid, kind, part in rows if kind == "verses" and part == 1]
        ids += intro_first + sections_first + [pid for pid, _, _ in rows if pid not in intro_first + sections_first]
    return ids[:limit]


def concept_share(conn: sqlite3.Connection, concept: Concept) -> float:
    """Share of all passages that mention the concept (or an approved alias)."""
    total = conn.execute("SELECT count(*) FROM passages").fetchone()[0]
    hits = conn.execute("SELECT count(*) FROM passages_fts WHERE passages_fts MATCH ?",
                        (fts_terms(concept.terms),)).fetchone()[0]
    return hits / max(total, 1)


def coverage_list(per_concept: Sequence[List[str]], weights: Optional[Sequence[float]] = None,
                  limit: int = LIST_LIMIT) -> List[str]:
    """Passages ranked by the weight of the concepts they match, then by their mean rank in
    those concepts' lists. Takes one ranked hit list per concept, and each concept's weight
    (its rarity in the book, log(1/share); equal weights by default). With several concepts,
    a passage is listed only if what it matches weighs at least as much as the rarest concept
    alone: "Musa" + "story" does not outrank "Khidr" for "the story of Musa and Al-Khidr"."""
    weights = list(weights) if weights is not None else [1.0] * len(per_concept)
    ranks: Dict[str, List[int]] = defaultdict(list)
    weight: Dict[str, float] = defaultdict(float)
    for hits, w in zip(per_concept, weights):
        for rank, pid in enumerate(hits, 1):
            ranks[pid].append(rank)
            weight[pid] += w
    need = max(weights) + (1e-9 if len(set(weights)) == 1 and len(weights) > 1 else 0) if len(weights) > 1 else 0
    found = [pid for pid in ranks if weight[pid] >= need]
    return sorted(found, key=lambda pid: (-weight[pid], sum(ranks[pid]) / len(ranks[pid])))[:limit]


def citation_list(conn: sqlite3.Connection, seed_sections: Sequence[str], concepts: Sequence[Concept],
                  surahs: Sequence[int] = ()) -> List[str]:
    """Passages of sections linked to the seeds by the book's references (either direction),
    most-linked first, kept only if they mention an important concept."""
    if not seed_sections or not concepts:
        return []
    marks = ",".join("?" * len(seed_sections))
    links: Dict[str, int] = defaultdict(int)
    for target, n in conn.execute(
            f"SELECT target_section_id, count(*) FROM citations WHERE section_id IN ({marks}) "
            f"AND is_self = 0 GROUP BY target_section_id", list(seed_sections)):
        links[target] += n
    for source, n in conn.execute(
            f"SELECT section_id, count(*) FROM citations WHERE target_section_id IN ({marks}) "
            f"AND is_self = 0 GROUP BY section_id", list(seed_sections)):
        links[source] += n
    for s in seed_sections:
        links.pop(s, None)
    return _linked_passages(conn, links, concepts, surahs)


def neighbour_list(conn: sqlite3.Connection, seed_sections: Sequence[str], concepts: Sequence[Concept],
                   surahs: Sequence[int] = ()) -> List[str]:
    """A story runs through consecutive sections (2:30, 2:31-33, 2:34, ...). In every surah
    with at least two seed sections, the sections just before and after each seed are
    added, kept only if they mention an important concept."""
    if not seed_sections or not concepts:
        return []
    seeds = set(seed_sections)
    by_surah: Dict[int, List[str]] = defaultdict(list)
    for sid in seeds:
        by_surah[int(sid.split(":")[0])].append(sid)
    links: Dict[str, int] = defaultdict(int)
    for surah, members in by_surah.items():
        if len(members) < 2:
            continue
        order = [r[0] for r in conn.execute(
            "SELECT section_id FROM passages WHERE surah = ? GROUP BY section_id ORDER BY min(rowid)", (surah,))]
        for i, sid in enumerate(order):
            if sid in seeds:
                for j in (i - 1, i + 1):
                    if 0 <= j < len(order) and order[j] not in seeds:
                        links[order[j]] += 1
    return _linked_passages(conn, links, concepts, surahs)


def _linked_passages(conn: sqlite3.Connection, links: Dict[str, int], concepts: Sequence[Concept],
                     surahs: Sequence[int]) -> List[str]:
    """Passages of the linked sections that mention a concept, most-linked section first."""
    if not links:
        return []
    match = " OR ".join(fts_terms(c.terms) for c in concepts)
    hits = keyword_list(conn, match, surahs=surahs, sections=list(links), limit=LIST_LIMIT * 4)
    order = {pid: i for i, pid in enumerate(hits)}
    section_of = dict(_section_ids(conn, hits))
    return sorted(hits, key=lambda pid: (-links[section_of[pid]], order[pid]))[:LIST_LIMIT]


def fuse(lists: Dict[str, List[str]], verse_focus: bool = False,
         compare: bool = False) -> Dict[str, Tuple[float, List[str]]]:
    """Reciprocal rank fusion: passage id -> (score, names of lists it appeared in)."""
    fused: Dict[str, Tuple[float, List[str]]] = {}
    for name, ids in lists.items():
        weight = WEIGHTS[name.split(":")[0]]
        if compare and name.startswith("concept:"):
            weight *= COMPARE_CONCEPT_FACTOR
        if verse_focus and name != "verse":
            weight *= VERSE_FOCUS
        for rank, pid in enumerate(ids, 1):
            score, reasons = fused.get(pid, (0.0, []))
            fused[pid] = (score + weight / (RRF_K + rank), reasons + [f"{name}#{rank}"])
    return fused


def verse_text(conn: sqlite3.Connection, section_id: str) -> str:
    """The verse translations that open a section ("(30. ...) (31. ...)")."""
    row = conn.execute("SELECT text FROM passages WHERE section_id = ? ORDER BY part LIMIT 1",
                       (section_id,)).fetchone()
    if not row:
        return ""
    lines = []
    for para in row[0].split("\n\n"):
        if _VERSE_LINE_RE.match(para):
            lines.append(para)
        elif lines:
            break
    return "\n".join(lines)


def retrieve(conn: sqlite3.Connection, query: ParsedQuery, max_passages: int = MAX_PASSAGES,
             max_chars: int = MAX_CHARS) -> RetrievalResult:
    lists: Dict[str, List[str]] = {}
    notes = list(query.notes)
    concepts = query.concepts
    shares = {c.typed: concept_share(conn, c) for c in concepts}
    important = [c for c in concepts if 0 < shares[c.typed] <= IMPORTANT_MAX_SHARE]
    if not important and concepts:
        rarest = min(concepts, key=lambda c: shares[c.typed] or 1.0)
        important = [rarest] if shares[rarest.typed] > 0 else []
        if important and not query.verses and not query.surahs:
            notes.append("The question names no specific subject (every word is common in the book); "
                         "results are broad")
    for c in concepts:
        if shares[c.typed] == 0:
            notes.append(f"'{c.typed}' does not occur in the book")
    scope = sorted({s for s, _, _ in query.verses} | set(query.surahs))

    if query.verses:
        lists["verse"] = verse_list(conn, query.verses)
    if query.surahs and not concepts and not query.verses:
        lists["verse"] = surah_list(conn, query.surahs)
        if len(lists["verse"]) == SURAH_ONLY_LIMIT:
            notes.append(f"Surah has more passages than shown; only the first {SURAH_ONLY_LIMIT} were used")

    if important:
        typed_hits = [keyword_list(conn, fts_terms([c.typed]), scope, limit=COVERAGE_POOL) for c in important]
        expanded_hits = [keyword_list(conn, fts_terms(c.terms), scope, limit=COVERAGE_POOL) for c in important]
        # Rarity in the book; halved for plain lowercase words ("story") next to names the
        # user capitalized or glossary terms ("Adam", "Iblis"), which are the subject.
        rarity = [math.log(1 / shares[c.typed]) * (1.0 if c.proper else COMMON_WORD_FACTOR) for c in important]
        if query.compare:
            rarity = [1.0] * len(important)  # rank by how many sides a passage covers
        lists["coverage_typed"] = coverage_list(typed_hits, rarity)
        lists["coverage_expanded"] = coverage_list(expanded_hits, rarity)
        for c, hits in zip(important, expanded_hits):
            lists[f"concept:{c.typed}"] = hits[:LIST_LIMIT]
    if concepts:
        lists["any"] = keyword_list(conn, " OR ".join(fts_terms(c.terms) for c in concepts), scope)
    lists = {name: ids for name, ids in lists.items() if ids}

    fused = fuse(lists, verse_focus=bool(query.verses), compare=query.compare)
    if important:
        seeds = _top_sections(conn, fused, CITATION_SEEDS)
        lists["citations"] = citation_list(conn, seeds, important, scope)
        lists["neighbours"] = neighbour_list(conn, seeds, important, scope)
        lists = {name: ids for name, ids in lists.items() if ids}
        fused = fuse(lists, verse_focus=bool(query.verses), compare=query.compare)

    if not fused:
        notes.append("Nothing in the book matched the question")
        return RetrievalResult(query, [], lists, notes)

    ranked = sorted(fused.items(), key=lambda kv: -kv[1][0])
    floor = ranked[0][1][0] * MIN_RELATIVE_SCORE
    ranked = [(pid, v) for pid, v in ranked if v[0] >= floor]
    rows = _fetch(conn, [pid for pid, _ in ranked])
    chosen: List[Retrieved] = []
    total = 0
    for pid, (score, reasons) in ranked:
        r = rows[pid]
        if len(chosen) >= max_passages or (chosen and total + len(r.text) > max_chars):
            continue
        r.score, r.reasons = score, reasons
        chosen.append(r)
        total += len(r.text)

    order = {pid: i for i, pid in enumerate(rows)}
    chosen.sort(key=lambda r: (r.surah, r.ayah_start, order[r.id]))
    for sid in {r.section_id for r in chosen}:
        text = verse_text(conn, sid)
        for r in chosen:
            if r.section_id == sid:
                r.verse_text = text
    return RetrievalResult(query, chosen, lists, notes)


def _top_sections(conn: sqlite3.Connection, fused: Dict[str, Tuple[float, List[str]]], n: int) -> List[str]:
    ranked = [pid for pid, _ in sorted(fused.items(), key=lambda kv: -kv[1][0])]
    sections: List[str] = []
    for pid, sid in _section_ids(conn, ranked):
        if sid not in sections:
            sections.append(sid)
        if len(sections) == n:
            break
    return sections


def _section_ids(conn: sqlite3.Connection, ids: List[str]) -> List[Tuple[str, str]]:
    found = dict(conn.execute(
        f"SELECT id, section_id FROM passages WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()) if ids else {}
    return [(pid, found[pid]) for pid in ids if pid in found]


def _fetch(conn: sqlite3.Connection, ids: List[str]) -> Dict[str, Retrieved]:
    """Rows for `ids`, keyed in book order."""
    if not ids:
        return {}
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM passages WHERE id IN ({','.join('?' * len(ids))}) ORDER BY rowid",
                        ids).fetchall()
    return {r["id"]: _row(r) for r in rows}


def main(argv: Optional[List[str]] = None) -> None:
    """python -m rag.retrieve "question" [--verbose]: show what would be retrieved."""
    import argparse
    from pipeline.index import connect
    from rag.query import parse_query

    parser = argparse.ArgumentParser(description="Show the passages retrieved for a question")
    parser.add_argument("question")
    parser.add_argument("--verbose", action="store_true", help="also list why each passage was chosen")
    args = parser.parse_args(argv)

    q = parse_query(args.question)
    res = retrieve(connect(), q)
    print(f"verses: {q.verses}  surahs: {q.surahs}")
    print("concepts: " + "; ".join(c.typed + (f" (+{', '.join(c.aliases)})" if c.aliases else "") for c in q.concepts))
    for note in res.notes:
        print(f"note: {note}")
    print(f"{len(res.passages)} passages, {sum(len(r.text) for r in res.passages):,} chars")
    for r in res.passages:
        print(f"  {r.score:.4f}  {r.id:<14} {r.citation:<26} {r.title or ''}")
        if args.verbose:
            print(f"           {', '.join(r.reasons)}")


if __name__ == "__main__":
    main()
