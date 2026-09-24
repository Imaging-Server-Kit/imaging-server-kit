"""Small formatting helpers used by the `__repr__` / `__str__` methods of public classes."""

from typing import Any, Optional, Sequence


def fmt_num(value: Any) -> str:
    """Format a number compactly: integral floats as ints, others with 4 significant digits."""
    if isinstance(value, bool):
        return str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4g}"


def fmt_tuple(values: Optional[Sequence]) -> str:
    """Format a sequence of numbers as a compact tuple, e.g. (0.0, 512.0) -> '(0, 512)'."""
    if values is None:
        return "None"
    return f"({', '.join(fmt_num(v) for v in values)})"


def fmt_slices(coords_min: Optional[Sequence], coords_max: Optional[Sequence]) -> str:
    """Format an extent in slice notation, e.g. '[0:512, 100:356]'."""
    if coords_min is None or coords_max is None:
        return "undefined"
    return f"[{', '.join(f'{fmt_num(a)}:{fmt_num(b)}' for a, b in zip(coords_min, coords_max))}]"


def truncate(text: str, max_len: int = 40) -> str:
    """Truncate a string to `max_len` characters, adding an ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
