import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "pbr.sphinxext",
    # ... other extensions
    "sphinxcontrib.apidoc",
]

apidoc_module_dir = "../.."
apidoc_output_dir = "reference"
apidoc_excluded_paths = [
    "tests",
    "_*",
    "setup.py",
]
apidoc_separate_modules = True
