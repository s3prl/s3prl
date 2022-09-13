# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import inspect
import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "S3PRL"
copyright = "2022, S3PRL Team"
author = "S3PRL Team"

# The full version, including alpha/beta/rc tags
with (Path(__file__).parent.parent.parent / "s3prl" / "version.txt").open() as f:
    release = f.read()


def linkcode_resolve(domain, info):
    def find_source():
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        if isinstance(obj, property):
            return None

        file_parts = Path(inspect.getsourcefile(obj)).parts
        reversed_parts = []
        for part in reversed(file_parts):
            if part == "s3prl":
                reversed_parts.append(part)
                break
            else:
                reversed_parts.append(part)
        fn = "/".join(reversed(reversed_parts))

        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None

    tag = "master" if "dev" in release else ("v" + release)  # s3prl github version

    try:
        filename = "%s#L%d-L%d" % find_source()  # specify file page with line number
    except Exception:
        filename = (
            info["module"].replace(".", "/") + ".py"
        )  # cannot find corresponding codeblock, use the file page instead

    return "https://github.com/s3prl/s3prl/blob/%s/%s" % (tag, filename)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# add extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

html_js_files = [
    "js/custom.js",
]
html_css_files = ["css/custom.css"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Uncomment the following if you want to document __call__
#
# def skip(app, what, name, obj, would_skip, options):
#     if name == "__call__":
#         return False
#     return would_skip
#
# def setup(app):
#     app.connect("autodoc-skip-member", skip)

autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_member_order = "bysource"
