# Installation Guide

Follow these steps to install the ccCPM Python package directly from GitHub.

## Prerequisites

- tested with Python 3.10 or later
- `pip` (Python's package manager)

## Installation Steps

Clone the GitHub repository:

```bash
git clone https://github.com/mmll/cpm_python.git
```

Navigate to the repository directory:

```bash
cd cpm_python
```

Install the package:

```bash
pip install .
```

To install in development mode, use:

```bash
pip install -e .
```

Verify the installation:

```python
import cccpm

print(cccpm.__version__)
```

You should see the package version printed without errors.