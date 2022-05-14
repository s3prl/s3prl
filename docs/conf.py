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

for x in os.walk(".."):
    sys.path.insert(0, x[0])


# -- Project information -----------------------------------------------------

project = "S3PRL"
copyright = "2022, S3PRL"
author = "S3PRL"

# The full version, including alpha/beta/rc tags
release = "0.4.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
]

html_js_files = [
    "js/custom.js",
]


def linkcode_resolve(domain, info):
    def find_source():
        obj = sys.modules[info["module"]]
        if info["fullname"] == "InitConfig.args":
            return None
        if info["fullname"] == "InitConfig.kwargs":
            return None
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        if isinstance(obj, property):
            return None

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(os.path.abspath(__file__))[:-4])

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

    return "https://github.com/s3prl/s3prl-private/blob/%s/%s" % (tag, filename)


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_mock_imports = ["fairseq", "torch"]
