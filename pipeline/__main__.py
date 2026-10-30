"""Run every build step in order: python -m pipeline"""
from pipeline import citations, extract_pages, glossary, index, sections, surahs

# glossary and surahs keep every reviewed edit when they regenerate their files.
for step in (extract_pages, sections, citations, index, glossary, surahs):
    print(f"== {step.__name__}")
    step.main([])
