"""Settings shared by the English and Japanese Sphinx projects."""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
ROOT = DOCS_DIR.parent


def _release() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"\s*$', text)
    if match is None:
        raise RuntimeError("Could not read version from pyproject.toml")
    return match.group(1)


def configure(namespace, *, language, other_language, other_label, source_directory):
    src = ROOT / "src"
    src_path = str(src)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    release = _release()

    def setup(app):
        def add_language_link(app, pagename, templatename, context, doctree):
            ups = "../" * (pagename.count("/") + 1)
            context["language_switch_url"] = f"{ups}{other_language}/{pagename}.html"
            context["language_switch_label"] = other_label
            context["language_switch_hreflang"] = other_language

        app.connect("html-page-context", add_language_link)

    namespace.update(
        project="FXTensor",
        author="Tadachika Oki",
        copyright="2025, Tadachika Oki",
        release=release,
        version=release,
        language=language,
        master_doc="index",
        extensions=[
            "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx.ext.napoleon",
            "sphinx.ext.viewcode",
        ],
        templates_path=["../_templates"],
        html_static_path=["../_static"],
        html_css_files=["language-switch.css"],
        html_theme="furo",
        html_title="FXTensor",
        html_baseurl=f"https://salmonsnare.github.io/fxtensor-salmon/{language}/",
        html_theme_options={
            "announcement": other_label,
            "source_repository": "https://github.com/salmonsnare/fxtensor-salmon/",
            "source_branch": "main",
            "source_directory": source_directory,
            "top_of_page_buttons": ["view", "edit"],
            "navigation_with_keys": True,
        },
        exclude_patterns=["_build", "Thumbs.db", ".DS_Store"],
        autodoc_member_order="bysource",
        autodoc_typehints="description",
        autodoc_inherit_docstrings=True,
        autoclass_content="class",
        autosummary_generate=False,
        napoleon_google_docstring=True,
        napoleon_numpy_docstring=True,
        napoleon_preprocess_types=False,
        pygments_style="sphinx",
        setup=setup,
    )
