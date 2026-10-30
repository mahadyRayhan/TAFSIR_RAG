# Unfinished edits to the vector-RAG app (June 2025)

Snapshot of edits that were never committed, kept for reference only. Nothing imports these
files; the running app still uses the committed `tafsir.py` at the repo root until it is
replaced by the vectorless pipeline.

- `tafsir.py`: renames chunk metadata to `chunk_on_page`, builds ChromaDB ids from
  document/page/chunk, simplifies the CSV similarity search, and asks the final-answer prompt
  to cite `(Source: Document, Page, ChunkOnPage)`. **Note:** it also re-enables a debug limit,
  `chunk_texts = chunk_texts[:50]`, which would rebuild the vector DB with only 50 chunks.
- `api_smoke_test.py`: two of the three test questions commented out.
- `uncommitted_changes.patch`: the exact diff against the last committed versions
  (`git apply` it at commit 8096054 or earlier to restore the edits in place).
