# Contributing to CCCPM

Thanks for your interest in improving CCCPM! Contributions of all kinds are
welcome — bug reports, feature requests, documentation, and code.

## Reporting issues

Please open an issue on
[GitHub](https://github.com/wwu-mmll/confound_corrected_cpm/issues) and include:

- what you expected to happen and what actually happened,
- a minimal example that reproduces the problem (ideally using simulated data),
- your operating system, Python version, and `cccpm` version
  (`python -c "import cccpm; print(cccpm.__version__)"`).

## Development setup

CCCPM uses [Poetry](https://python-poetry.org/) and a `src/` layout.

```bash
git clone https://github.com/wwu-mmll/confound_corrected_cpm.git
cd confound_corrected_cpm
poetry install                # add --with docs for the documentation tools
```

On Apple Silicon, make sure you are using a native arm64 Python (see the
[installation guide](https://wwu-mmll.github.io/confound_corrected_cpm/installation/)).

## Running the tests

```bash
poetry run pytest                       # full suite
poetry run pytest --cov=cccpm           # with coverage
poetry run pytest tests/test_scoring.py # a single file
```

All tests should pass before you open a pull request. New functionality should
come with tests; the suite includes end-to-end "ground truth" tests on simulated
data with known structure, and example scripts in `examples/` are run as part of
`tests/test_integration.py`.

## Building the documentation

```bash
poetry install --with docs
poetry run mkdocs serve     # live preview at http://127.0.0.1:8000
poetry run mkdocs build --strict
```

Documentation lives in `documentation/docs/`. Example tutorials embed the actual
scripts from `examples/` via snippets, so update the script and the docs stay in
sync.

## Pull requests

1. Create a feature branch off `develop`.
2. Make your change with tests and documentation.
3. Ensure `poetry run pytest` and `poetry run mkdocs build --strict` pass.
4. Open a pull request against `develop` describing the change and why.

## Code style

Match the style of the surrounding code: NumPy-style docstrings, clear names, and
keep the public API (exported from `cccpm/__init__.py`) stable and documented.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).
