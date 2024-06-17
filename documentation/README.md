# Documentation How-To

## Use documentation webpage locally

To make life easier, you can simply try and install everything inside your current Python environment. If you don't want to do that, feel free to create a dedicated Python environment first. Make sure to use Python 3.7, as I had problems with the latest Python 3.8.

### Installation

Install mkdocs (Python package) through PyPI (inside your Python environment, if you're using one). We also need to install the material theme https://squidfunk.github.io/mkdocs-material/ and some custom plugins we're using to render Latex and so on.

```bash
pip install mkdocs mkdocs-material pymdown-extensions mkdocstrings-python mkdocs-jupyter
```

### How to use it

Within the terminal, navigate to the documentation folder and type:

```bash
PYTHONPATH=.. mkdocs serve
```

If you were using a dedicated Python environment, make sure to activate it before. Once you've run make doc-serve, mkdocs will start a local webserver and host the documentation website under:

http://127.0.0.1:8000/

Just follow that link or copy and paste it to your browser. You should now be able to use the documentation from your browser.

## Build the website

### Build site and export PDF

To see the website while you are working on some of the markdown files (not the ones automatically generated, but the ones for general descriptions of the toolkit and so on), you can cd to the documentation folder and run `mkdocs serve`. You can then work on the markdowns and the changes live in the browser.

To build the website, cd to the documentation folder and run `mkdocs gh-deploy`. This will create the website in the default gh-pages branch and push everything to the repo. Unfortunately, this will also create a docs folder in the master branch. Make sure to delete this folder.
