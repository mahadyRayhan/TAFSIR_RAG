# TAFSIR_RAG

Question answering over *Tafsir Ibn Kathir* (abridged English, 10 volumes, 5,698 PDF pages).
Answers are written as a narrative with every claim cited by Surah:Ayah (e.g. `2:34`).

Retrieval is built from the book itself — its verse structure, a keyword index and its own
cross-references — with no embeddings and no use of the model's own knowledge to choose passages.
The LLM only writes the final answer from retrieved passages, and its citations are checked.

## Layout

```text
main.py, tafsir.py, templates/   current Flask app (legacy vector RAG; replaced in step 11)
pipeline/                        offline build steps (PDF -> pages -> sections -> index)
rag/                             runtime: query parsing, retrieval, answering
glossary/                        reviewed, tracked: glossary.csv, surah_names.csv
tests/                           pytest suite
scripts/                         manual tools (api_smoke_test.py hits a running server)
legacy/hadith/                   earlier Imam Nawawi hadith prototype
data/        (git-ignored)       source/ PDFs, build/ pipeline outputs, legacy/ old vector CSVs
logs/        (git-ignored)
```

## Build steps

| Step | Module | Output | Tag |
|------|--------|--------|-----|
| 1 Clean text extraction | `pipeline.extract_pages` | `data/build/pages.jsonl` | `step-1` |
| 2 Verse sections + passages | `pipeline.sections` | `data/build/passages.jsonl` | `step-2` |
| 3 Citation graph | `pipeline.citations` | `data/build/citations.jsonl` | `step-3` |
| 4 Keyword index (SQLite FTS5) | `pipeline.index` | `data/build/tafsir.db` | `step-4` |
| 5 Glossary (human-reviewed) | `pipeline.glossary` | `glossary/glossary.csv` (tracked) | `step-5` |
| 6 Query parsing | `rag.query` (+ `pipeline.surahs`) | `glossary/surah_names.csv` (tracked) | `step-6` |
| 7 Retrieval + RRF | `rag.retrieve` (`python -m rag.retrieve "question"`) | ranked passages | `step-7` |
| 8 Test set / recall | | | |
| 9 Answer writing | | | |
| 10 Citation checks | | | |

**Step 1 notes.** PyMuPDF replaces PyPDF2 (broken words like `th e` fell from ~3,400 to ~30 per
million characters). Arabic text uses a custom-encoded font and is dropped by font name; each run
becomes an `arabic` block labelled `verse` or `hadith`. About 800 pages use an alternate encoding
(`\x1f` for parentheses and under-dots, `0` for long vowels, reversed `)2:140(` references); these
are repaired, using the book's own vocabulary to tell `H\x1f asan` (= Hasan) from a parenthesis.
Known limit: ~180 repaired paragraphs keep an unmatched parenthesis where the PDF lost the glyph.

**Step 2 notes.** Surahs are found from the book's `(Chapter N)` headings (110 of 114); surahs
2, 31, 113 and 114 have none and are found where verse numbering restarts. A verse-translation
block `(30. ...) (31. ...)` opens a section; its commentary runs to the next such block. Section
ranges are stretched to the next section, so all 6,236 verses fall in exactly one section
(95.1% have their own printed marker). Long sections are split at Ibn Kathir's topic headings,
then at paragraph breaks (max ~6,000 chars); each passage keeps the section id, e.g. `2:34#2`
"The Prostration was before Adam but the Obedience was to Allah". The PDF prints Surah Al-Ma`un
twice; the second copy is skipped. `pipeline/quran.py` holds the standard verse count per surah,
used only to validate parsing.

**Step 3 notes.** 5,113 Qur'an references (`(7:199-200)`, `8:62,63`) are parsed from the
passages and all resolve to the section commenting on that verse, giving 4,405 distinct
section-to-section edges (e.g. the 2:34 prostration passage cites 18:50, "Iblis was one of the
Jinn"). Volume:page citations next to hadith sources (`Al-Kubra 5:12`) are skipped.

**Step 4 notes.** `tafsir.db` (SQLite, ~31 MB, builds in under a second) holds the passages,
an FTS5 BM25 index over headings + text (Porter stemming, diacritics removed, ayn/apostrophes
dropped, headings weighted 4x) and the citation edges. A plain `"Adam" AND "Iblis"` search
returns the story in surahs 2, 7, 15, 17, 18, 20 and 38 from the book alone. Deployment note:
App Engine standard rejects files over 32 MB, so the database is close to that limit.

**Step 5 notes.** `python -m pipeline.glossary` proposes equivalent search terms from the book
only: definitions the translators give in parentheses (`Iblis (Shaytan)`, `Salah (prayer)`,
`At-Taqwa (piety, righteousness)`, each seen at least twice) and pure spelling variants
(`Quraysh`/`Quraish`). **Nothing is used until a person approves it**: in
`glossary/glossary.csv` set `keep` to `y`, edit `terms` (pipe-separated) as needed, and add
your own rows with an id starting `custom:`. Re-running the generator keeps all edits.
Approved rows that share a term are merged into one group.

**Step 6 notes.** `rag.query.parse_query()` reads verse references (`2:34`, `2:30-39`,
"Surah Al-Baqarah verse 34", "verses 30 to 39 of Al-Baqarah", "chapter 18 verse 50"), whole
surahs ("Surah Yusuf", "in Al-Baqarah") and search concepts, each expanded only by approved
glossary groups; typed words are kept apart from expansions. Bare prophet names ("story of
Yusuf") stay concepts. Arabic name phrases ("Ayat al-Kursi") are searched whole. Surah names
come from the book's own opening/closing lines (`python -m pipeline.surahs` writes
`glossary/surah_names.csv`; add aliases such as "The Cow" there by hand). Building this step
also fixed two section boundaries from step 2: Al-Baqarah's intro (pp. 64-69) had stayed in
1:7, and the duplicated Al-Ma`un copy had leaked its title into surah 108; surah 31's heading
printed as `Chapter - 31)` is now recognised.

**Step 7 notes.** `rag.retrieve.retrieve()` fuses ranked lists with reciprocal rank fusion:
direct verse lookup; coverage (passages ranked by the question's subject concepts they
contain, each weighted by its rarity in the book, typed words above glossary aliases);
one list per concept; any concept; and two structural lists from the book itself, its
cross-references and the neighbouring sections of a story told over consecutive sections.
A concept is a subject when it occurs in at most 20% of passages ("Iblis" 8%, "Allah" 96%).
Named verses dominate; "in Al-Baqarah" limits the search to that surah; "contrast A with B"
covers each side separately. Passages under 25% of the best score are dropped; at most 30
passages / 120k characters are returned in Qur'an order, each with its reasons and the
section's verse translations. "Adam and Iblis" returns 2:30-36, 7:11-27, 15:28-44, 17:61-65,
18:50, 20:115-126 and 38:71-85 in ~70 ms. The weights were set on a handful of questions only;
step 8's test set is what should tune them.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
# put the PDF at data/source/ibn_kathir/
python -m pipeline        # steps 1-4 + glossary/surah-name candidates (~25 s, no API calls)
pytest
```
