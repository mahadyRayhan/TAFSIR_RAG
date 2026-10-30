# TAFSIR_RAG

Question answering over *Tafsir Ibn Kathir* (abridged English, 10 volumes, 5,698 PDF pages).
Answers are written as a narrative with every claim cited by Surah:Ayah (e.g. `2:34`).

Retrieval is built from the book itself — its verse structure, a keyword index and its own
cross-references — with no embeddings and no use of the model's own knowledge to choose passages.
The LLM only writes the final answer from retrieved passages, and its citations are checked.

## Layout

```
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
| 2 Verse sections | | | |

**Step 1 notes.** PyMuPDF replaces PyPDF2 (broken words like `th e` fell from ~3,400 to ~30 per
million characters). Arabic text uses a custom-encoded font and is dropped by font name; each run
becomes an `arabic` block labelled `verse` or `hadith`. About 800 pages use an alternate encoding
(`\x1f` for parentheses and under-dots, `0` for long vowels, reversed `)2:140(` references); these
are repaired, using the book's own vocabulary to tell `H\x1f asan` (= Hasan) from a parenthesis.
Known limit: ~180 repaired paragraphs keep an unmatched parenthesis where the PDF lost the glyph.
| 3 Citation graph | | | |
| 4 Keyword index (SQLite FTS5) | | | |
| 5 Glossary | | | |
| 6 Query parsing | | | |
| 7 Retrieval + RRF | | | |
| 8 Test set / recall | | | |
| 9 Answer writing | | | |
| 10 Citation checks | | | |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
# put the PDF at data/source/ibn_kathir/
python -m pipeline.extract_pages
pytest
```
