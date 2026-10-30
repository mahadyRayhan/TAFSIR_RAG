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
| 4 Keyword index (SQLite FTS5) | | | |
| 5 Glossary | | | |
| 6 Query parsing | | | |
| 7 Retrieval + RRF | | | |
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

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
# put the PDF at data/source/ibn_kathir/
python -m pipeline.extract_pages
python -m pipeline.sections
python -m pipeline.citations
pytest
```
