import py_compile
from pathlib import Path

import pytest

from pipeline import paths


def test_paths_live_inside_repo():
    for p in (paths.DATA_DIR, paths.SOURCE_DIR, paths.BUILD_DIR, paths.SOURCE_PDF, paths.PAGES_JSONL):
        assert paths.ROOT in p.parents


@pytest.mark.data
def test_source_pdf_present():
    if not paths.SOURCE_DIR.exists():
        pytest.skip("data/source not present")
    assert paths.SOURCE_PDF.is_file()


def test_python_files_compile():
    files = [f for d in ("pipeline", "scripts", "legacy") for f in (paths.ROOT / d).rglob("*.py")]
    files += [paths.ROOT / "main.py", paths.ROOT / "tafsir.py"]
    for f in files:
        py_compile.compile(str(f), doraise=True)
