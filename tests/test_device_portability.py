"""Guards against torch.cuda.* calls that run fine on a CUDA dev box but raise on
CPU-only CI runners -- regression: two bare torch.cuda.synchronize() calls in
cpm_analysis._select_edges failed six tests on CPU while passing on GPU."""
import ast
from pathlib import Path



PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "cccpm"

# torch.cuda attributes that do NOT require an available GPU: is_available
# returns False rather than raising; the nvtx no-ops work with no device;
# mem_get_info is guarded by a device.type == 'cuda' check at its call site.
CPU_SAFE_CUDA_CALLS = {
    "torch.cuda.is_available",
    "torch.cuda.nvtx.range",
    "torch.cuda.nvtx.range_push",
    "torch.cuda.nvtx.range_pop",
    "torch.cuda.mem_get_info",
}


def _dotted_name(node):
    """Dotted spelling of an attribute chain rooted in a Name, else None (e.g. f().cuda.foo)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _cuda_attribute_uses(source, filename):
    """Yield (lineno, dotted_name) for every torch.cuda.* attribute used. ast.walk visits
    every prefix of a chain (torch.cuda.nvtx.range also yields torch.cuda.nvtx); the
    caller keeps the longest spelling per line."""
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        name = _dotted_name(node)
        if name is None or not name.startswith("torch.cuda."):
            continue
        yield node.lineno, name


def _package_python_files():
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert files, f"no Python files found under {PACKAGE_ROOT}"
    return files


def test_no_gpu_only_cuda_calls():
    """One test over every module, not one per module: the failure message already
    names file and line, so per-file parametrisation only inflated the count."""
    violations = []
    for path in _package_python_files():
        source = path.read_text(encoding="utf-8")
        if "torch.cuda" not in source:
            continue

        by_line = {}
        for lineno, name in _cuda_attribute_uses(source, str(path)):
            current = by_line.get(lineno)
            if current is None or len(name) > len(current):
                by_line[lineno] = name

        violations += [
            f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {name}"
            for lineno, name in sorted(by_line.items())
            if name not in CPU_SAFE_CUDA_CALLS
        ]

    assert not violations, (
        "GPU-only torch.cuda call(s) found -- these raise on CPU-only machines "
        "(all CI runners) while passing locally on a CUDA box:\n  "
        + "\n  ".join(violations)
        + "\n\nEither remove the call, guard it behind a device-type check, or "
        "add it to CPU_SAFE_CUDA_CALLS if it is genuinely safe without a GPU."
    )
