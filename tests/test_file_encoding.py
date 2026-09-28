"""
Every text file the package reads or writes names its encoding.

Without ``encoding=``, ``open()`` uses the locale codec: UTF-8 on Linux and
macOS, cp1252 on most Windows machines. That is symmetric on one machine, so
it passes every CI job -- but a results directory written on a cp1252 box is
not readable on a UTF-8 one, and a non-cp1252 character in a target or
covariate name raises on write. Nothing at runtime would catch it; this scan
does.

pandas is exempt: ``read_csv`` / ``to_csv`` default to UTF-8 whatever the
locale, so only the builtin ``open`` and ``Path.read_text`` / ``write_text``
are checked. Binary modes carry no encoding and are skipped.
"""
import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "cccpm"

TEXT_IO_METHODS = {"read_text", "write_text"}


def _mode(call):
    """The mode string of an ``open()`` call, or 'r' when it is not given."""
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        return call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return "r"


def _unencoded_text_io(source, filename):
    """Yield (lineno, spelling) for each text-mode file access without ``encoding=``."""
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            if "b" not in str(_mode(node)):
                yield node.lineno, "open()"
        elif isinstance(func, ast.Attribute) and func.attr in TEXT_IO_METHODS:
            yield node.lineno, f".{func.attr}()"


def test_package_text_io_names_its_encoding():
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert files, f"no Python files found under {PACKAGE_ROOT}"

    violations = [
        f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {spelling}"
        for path in files
        for lineno, spelling in _unencoded_text_io(
            path.read_text(encoding="utf-8"), str(path))
    ]
    assert not violations, (
        "Text file access without an explicit encoding -- this uses the locale "
        "codec (cp1252 on Windows), so results written on one machine may not "
        "read on another:\n  " + "\n  ".join(violations)
        + "\n\nPass encoding='utf-8'."
    )
