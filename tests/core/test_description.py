import inspect
import webbrowser
from pathlib import Path

import imaging_server_kit as sk


def test_explicit_description_wins():
    @sk.algorithm(description="Custom")
    def algo():
        """Docstring."""

    assert algo.algo_info["description"] == "Custom"


def test_docstring_fallback():
    @sk.algorithm
    def algo():
        """First line.

        More details.
        """

    assert algo.algo_info["description"] == "First line.\n\nMore details."


def test_docstring_fallback_with_decorator_args():
    @sk.algorithm(tags=["a"])
    def algo():
        """Docstring."""

    assert algo.algo_info["description"] == inspect.getdoc(algo) == "Docstring."


def test_no_description_no_docstring():
    @sk.algorithm
    def algo():
        pass

    assert algo.algo_info["description"] == ""


def test_info_writes_doc_page(tmp_path, monkeypatch):
    @sk.algorithm(description="Custom")
    def algo(x: int = 1):
        pass

    opened = []
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(webbrowser, "open", opened.append)

    algo.info()

    output_path = tmp_path / ".serverkit" / "output.html"
    assert (tmp_path / ".serverkit" / "static" / "css" / "info.css").is_file()
    assert "Custom" in output_path.read_text(encoding="utf-8")
    assert opened == [output_path.resolve().as_uri()]
