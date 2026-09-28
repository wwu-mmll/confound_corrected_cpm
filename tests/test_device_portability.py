"""
Guard against CUDA-only code paths that crash on machines without a GPU.

Most contributors develop on a CUDA machine, where a bare ``torch.cuda.*`` call
runs fine. On a CPU-only machine -- which is every GitHub Actions runner -- the
same call raises (``RuntimeError: No CUDA GPUs are available`` on a CUDA-enabled
torch build with no device, ``AssertionError: Torch not compiled with CUDA
enabled`` on the CPU-only wheel). That is invisible locally and breaks CI.

This happened: two bare ``torch.cuda.synchronize()`` calls in
``cpm_analysis._select_edges`` failed six tests on CPU while the full suite
passed on GPU. The scan below catches the whole class rather than that one
instance.

Only calls that are genuinely safe without a GPU are allowed; see
``CPU_SAFE_CUDA_CALLS``.
"""
import ast
from pathlib import Path



PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "cccpm"

# ``torch.cuda`` attributes that do NOT require an available GPU.
CPU_SAFE_CUDA_CALLS = {
    # The capability check itself -- returns False rather than raising.
    "torch.cuda.is_available",
    # A no-op annotation when no profiler is attached; verified to work with
    # CUDA_VISIBLE_DEVICES="" on a CUDA-enabled build.
    "torch.cuda.nvtx.range",
    "torch.cuda.nvtx.range_push",
    "torch.cuda.nvtx.range_pop",
    # Guarded by an explicit `device.type == 'cuda'` check in
    # memory.available_memory_bytes.
    "torch.cuda.mem_get_info",
}


def _dotted_name(node):
    """Return the dotted source spelling of an attribute chain, or None.

    ``torch.cuda.nvtx.range`` -> "torch.cuda.nvtx.range". Returns None for
    anything not rooted in a plain name (e.g. ``f().cuda.foo``).
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _cuda_attribute_uses(source, filename):
    """Yield (lineno, dotted_name) for every ``torch.cuda.*`` attribute used."""
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        name = _dotted_name(node)
        if name is None or not name.startswith("torch.cuda."):
            continue
        # ast.walk visits every prefix of a chain, so torch.cuda.nvtx.range
        # also yields torch.cuda.nvtx. The caller keeps the longest spelling
        # per line.
        yield node.lineno, name


def _package_python_files():
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert files, f"no Python files found under {PACKAGE_ROOT}"
    return files


def test_no_gpu_only_cuda_calls():
    """No ``torch.cuda.*`` call in the package may require an available GPU.

    One test over every module rather than one test per module: the report below
    already names the file and line, so a per-file parametrisation only inflated
    the collected count (37 tests for one check).
    """
    violations = []
    for path in _package_python_files():
        source = path.read_text(encoding="utf-8")
        if "torch.cuda" not in source:
            continue

        # ast.walk visits every prefix of a chain, so torch.cuda.nvtx.range also
        # yields torch.cuda.nvtx. Keep the longest spelling per line.
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
