import inspect

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
