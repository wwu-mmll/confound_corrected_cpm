"""Scans for text I/O missing encoding='utf-8': without it, open() uses the locale
codec (UTF-8 on Linux/macOS, cp1252 on Windows) -- symmetric on one machine so it
passes CI, but a results dir written on a cp1252 box won't read on a UTF-8 one.
pandas is exempt (read_csv/to_csv always default to UTF-8); binary modes are skipped."""
import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "cccpm"

TEXT_IO_METHODS = {"read_text", "write_text"}


def _mode(call):
    """Mode string of an open() call, or 'r' when not given."""
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        return call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return "r"


def _unencoded_text_io(source, filename):
    """Yield (lineno, spelling) for each text-mode file access without encoding=."""
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
