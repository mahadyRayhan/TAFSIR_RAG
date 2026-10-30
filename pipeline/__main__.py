"""Run every build step in order: python -m pipeline"""
from pipeline import citations, extract_pages, index, sections

for step in (extract_pages, sections, citations, index):
    print(f"== {step.__name__}")
    step.main([])
