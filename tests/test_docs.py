"""Run the Python code blocks of documentation pages, to catch examples that drift from the API.

The blocks of each page are executed in order, in a shared namespace. Blocks that need a
GUI or a running server, or that are meant to raise, are skipped.
"""

from pathlib import Path
import re

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"

PAGES = [
    "tutorial/python.md",
    "how-to/regions.md",
    "concepts/layers-and-stacks.md",
    "concepts/coordinates.md",
]

SKIP_MARKERS = ["to_napari", "sk.serve(", "sk.Client(", "client.", ".info()", "# Raises"]

CODE_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _code_blocks(page: str) -> list:
    text = (DOCS_DIR / page).read_text(encoding="utf-8")
    blocks = CODE_BLOCK.findall(text)
    return [b for b in blocks if not any(marker in b for marker in SKIP_MARKERS)]


@pytest.mark.parametrize("page", PAGES)
def test_docs_page_code_runs(page):
    blocks = _code_blocks(page)
    assert blocks, f"No runnable code blocks found in {page}"

    namespace = {}
    for block in blocks:
        exec(compile(block, f"{page}", "exec"), namespace)
