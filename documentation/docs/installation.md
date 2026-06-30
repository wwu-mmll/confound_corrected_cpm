# Installation Guide

CCCPM runs on **macOS, Linux, and Windows** with **Python 3.10–3.14**. It uses
[PyTorch](https://pytorch.org/) for computation and works on CPU out of the box;
GPU (CUDA) and Apple Silicon (MPS) acceleration are used automatically when available.

## Quick install

```bash
pip install cccpm
```

We strongly recommend installing into a clean virtual environment:

=== "venv (any OS)"

    ```bash
    python -m venv .venv
    # macOS / Linux:
    source .venv/bin/activate
    # Windows (PowerShell):
    .venv\Scripts\Activate.ps1

    pip install cccpm
    ```

=== "conda"

    ```bash
    conda create -n cccpm python=3.11
    conda activate cccpm
    pip install cccpm
    ```

Verify the installation:

```bash
python -c "import cccpm; print(cccpm.__version__)"
```

## Requirements

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14.
- **PyTorch**: installed automatically as a dependency. By default you get the build
  appropriate for your platform (CPU on Windows, CPU/CUDA on Linux, CPU/MPS on macOS).

## CPU vs GPU

You do **not** need a GPU — CCCPM is fast on CPU for typical connectome sizes.

- **NVIDIA GPU (Linux/Windows):** pass `device="cuda"` to `CPMAnalysis`. If the default
  PyTorch wheel doesn't match your CUDA driver, install the matching build first by
  following the [official PyTorch instructions](https://pytorch.org/get-started/locally/),
  then `pip install cccpm`.
- **Apple Silicon (M-series Macs):** PyTorch's MPS backend is available automatically.
- **CPU only:** the default — nothing extra to do.

## Platform notes

=== "macOS"

    - On **Apple Silicon (M1/M2/M3/…)**, make sure you are using a **native arm64**
      Python. Modern PyTorch no longer ships x86_64 macOS wheels, so an Intel/Rosetta
      Python will fail to install `torch` (you'll see *"no matching distribution"* or
      skipped wheels). Check with:

      ```bash
      python -c "import platform; print(platform.machine())"   # should print 'arm64'
      ```

      If it prints `x86_64`, install a native arm64 Python (e.g. via
      [Homebrew](https://brew.sh): `brew install python@3.11`) and recreate your
      virtual environment with it.

=== "Linux"

    - The default `torch` wheel includes CUDA support and is large. For a smaller,
      CPU-only install, follow the PyTorch CPU instructions before installing CCCPM.

=== "Windows"

    - Use a recent 64-bit Python from [python.org](https://www.python.org/) or the
      Microsoft Store. Installing into a virtual environment (above) avoids most
      permission and PATH issues.

## Development install

To work on CCCPM itself (or use the latest unreleased changes), install from source
with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/wwu-mmll/confound_corrected_cpm.git
cd confound_corrected_cpm
poetry install            # add --with docs to build the documentation
poetry run pytest         # run the test suite
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` after install** — your environment
  couldn't install PyTorch (most often the Apple Silicon / Rosetta issue above, or an
  unsupported Python version). Confirm `python --version` is 3.10–3.14 and, on macOS,
  that `platform.machine()` is `arm64`.
- **`pip install cccpm` resolves very slowly or fails on dependency versions** — make
  sure `pip` is up to date (`python -m pip install --upgrade pip`) and that you are on
  a supported Python version.
- **CUDA out of memory / wrong CUDA version** — install the matching PyTorch build from
  the [official instructions](https://pytorch.org/get-started/locally/) first, then
  install CCCPM.
- **`_tkinter.TclError: Can't find a usable tk.tcl` (or `init.tcl`) when running an
  analysis or generating the report** — this is a plotting error, usually on **Windows**.
  CCCPM saves all figures to disk, but matplotlib defaults to an interactive GUI backend
  (Tk), and some Python installs ship without a working Tcl/Tk. The simplest fix is to
  tell matplotlib to use the non-interactive `Agg` backend, which needs no GUI:

    === "Windows (PowerShell)"
        ```powershell
        $env:MPLBACKEND = "Agg"
        python your_script.py
        ```

    === "Windows (cmd)"
        ```bat
        set MPLBACKEND=Agg
        python your_script.py
        ```

    === "Inside a script (any OS)"
        ```python
        import matplotlib
        matplotlib.use("Agg")   # before importing cccpm or matplotlib.pyplot
        import cccpm
        ```

    Alternatively, reinstall Python from [python.org](https://www.python.org/downloads/)
    and make sure the **"tcl/tk and IDLE"** component is selected in the installer, which
    provides a working Tk.
