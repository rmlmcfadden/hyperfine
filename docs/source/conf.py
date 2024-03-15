# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomli
import datetime

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../../"))

from hyperfine import __version__


# get the metadata from the package's pyproject.toml 
with open("../../pyproject.toml", "rb") as f:
    toml = tomli.load(f)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = toml["project"]["name"]
author = ", ".join([author["name"] for author in toml["project"]["authors"]])
version = __version__
release = __version__
copyright = f"2022-{datetime.date.today().year:d}, {author}"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

#
autodoc_default_flags = ["members"]
autosummary_generate = True

# napoleon specific flags
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_templates",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
