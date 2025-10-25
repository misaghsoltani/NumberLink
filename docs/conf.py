"""Sphinx configuration for the NumberLink documentation."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import metadata as _pkg_metadata, version as _pkg_version
import inspect
import os
from os import path
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.metadata import PackageMetadata
    from types import ModuleType

    from sphinx.application import Sphinx

dist_name: str = "numberlink"
project: str = "NumberLink"

# Strings for Sphinx:
release: str = _pkg_version(dist_name)  # full version, e.g. "1.2.3"
version: str = release  # or: ".".join(release.split(".")[:2])
meta: PackageMetadata = _pkg_metadata(dist_name)
author: str = meta.get("Author", "") or meta.get("Author-email", "") or "NumberLink authors"

copyright: str = f"{time.localtime().tm_year} {author}"
description: str = "NumberLink Puzzle Environment for Gymnasium"
short_title: str = "NumberLink"
long_title: str = "NumberLink Puzzle Environment for Gymnasium"
links: dict[str, str] = {
    "GitHub": "https://github.com/misaghsoltani/NumberLink",
    "Documentation": "https://misaghsoltani.github.io/NumberLink/",
}


here: Path = Path(__file__).resolve()
repo_root: Path = here.parents[1]  # .../NumberLink
pkg_dir: Path = repo_root / "numberlink"  # package location


extensions: list[str] = [
    "sphinx.ext.intersphinx",
    # Note: viewcode and linkcode are mutually exclusive - we use linkcode for GitHub links
    "sphinx.ext.linkcode",  # GitHub "view source" per-object links
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

optional_extensions: list[str] = ["myst_parser", "sphinx_copybutton", "sphinx_sitemap", "autodoc2"]
for _ext in optional_extensions:
    __import__(_ext)
    extensions.append(_ext)

# MyST (Markdown)
myst_enable_extensions: list[str] = ["colon_fence", "deflist", "fieldlist", "linkify", "substitution"]
myst_heading_anchors: int = 3

# sphinx-autodoc2 (static analysis API docs)
autodoc2_packages: list[dict[str, str | list[str]]] = [
    {
        "path": path.relpath(str(pkg_dir), start=str(here.parent)).replace(os.sep, "/"),
        "module": "numberlink",
        "exclude_files": [],
    }
]
autodoc2_output_dir: str = "apidocs"
autodoc2_render_plugin: str = "rst"  # "myst" for Markdown output
autodoc2_module_summary: bool = True
# Include inherited members (override default that hides them)
autodoc2_hidden_objects: list[str] = []  # default is {"inherited"}
autodoc2_sort_names: bool = False  # keep by-source ordering
# Pretty-print long type names as short aliases in annotations:
autodoc2_replace_annotations: list[tuple[str, str]] = [
    ("numberlink.types.RenderMode", "RenderMode"),
    ("numberlink.types.Coord", "Coord"),
    ("numberlink.types.Lane", "Lane"),
    ("numberlink.types.CellLane", "CellLane"),
    ("numberlink.types.RGBInt", "RGBInt"),
    ("numberlink.types.ActType", "ActType"),
    ("numberlink.types.ObsType", "ObsType"),
    ("numpy.bool_", "bool"),
    ("numpy.bool", "bool"),
    ("np.bool_", "bool"),
    ("np.bool", "bool"),
    ("NDArray[np.bool_]", "NDArray[bool]"),
    ("NDArray[numpy.bool_]", "NDArray[bool]"),
    ("NDArray[np.bool | np.bool_]", "NDArray[bool]"),
    ("ipywidgets.Layout", "ipywidgets.widgets.widget_layout.Layout"),
    ("ipywidgets.Button", "ipywidgets.widgets.widget_button.Button"),
    ("ipywidgets.GridBox", "ipywidgets.widgets.widget_box.GridBox"),
]

# Intersphinx
intersphinx_mapping: dict[str, tuple[str, None]] = {
    "python": ("https://docs.python.org/3", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pygame": ("https://www.pygame.org/docs/", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "imageio": ("https://imageio.readthedocs.io/en/stable/", None),
    "IPython": ("https://ipython.readthedocs.io/en/stable/", None),
    "ipyevents": ("https://ipyevents.readthedocs.io/en/latest/", None),
}

# Theme
html_theme: str = "furo"
html_title: str = f"{short_title} Documentation"
html_baseurl: str = links["Documentation"]  # needed by sphinx-sitemap
html_copy_source: bool = False
html_show_sourcelink: bool = True  # Enable source links: they will use linkcode_resolve to point to GitHub
html_static_path: list[str] = ["_static"]
html_css_files: list[str] = ["custom.css"]
# html_logo: str = "_static/numberlink-logo.svg"
html_favicon: str = "_static/numberlink-favicon.ico"
html_extra_path: list[str] = []
html_theme_options: dict[str, bool | str | dict[str, str] | list[str] | list[dict[str, str]]] = {
    "navigation_with_keys": True,
    "light_css_variables": {"color-brand-primary": "#0f172a", "color-brand-content": "#0f172a"},
    "dark_css_variables": {"color-brand-primary": "#60a5fa", "color-brand-content": "#e6eefc"},
    "light_logo": "numberlink-logo-light.svg",
    "dark_logo": "numberlink-logo-dark.svg",
    "source_repository": links["GitHub"],  # Furo buttons ("View page", "Edit this page")
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view"],
    "source_view_link": links["GitHub"] + "/blob/main/docs/{filename}",
    # Add GitHub link and version to announcement banner
    "announcement": (
        '<a href="https://pypi.org/project/numberlink/" style="color: inherit; text-decoration: none;">'
        f"📦 <strong>NumberLink v{release}</strong></a> | "
        '<a href="' + links["GitHub"] + '" style="color: inherit; text-decoration: none;">'
        '<img src="_static/github-icon.svg" alt="GitHub" '
        'style="display: inline-block; vertical-align: middle; width: 1em; height: 1em; '
        'margin-right: 0.35em;"/>'
        "⭐ <strong>Star us on GitHub</strong></a>"
    ),
    # Footer icon linking to the repository for a visible, always-present link
    "footer_icons": [
        {
            "name": "GitHub",
            "url": links["GitHub"],
            "html": ('<img src="_static/github-icon.svg" alt="GitHub"/>'),
            "class": "",
        }
    ],
}

# Copy button
# Strip common prompts from copied code
copybutton_prompt_is_regexp: bool = True
copybutton_prompt_text: str = r">>> |\.\.\. |\$ "


templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky: bool = True
suppress_warnings: list[str] = [
    "ref.python"  # occasionally helpful if we re-export Python stdlib names
]

exclude_patterns.append("_autosummary")
exclude_patterns.append("_includes")

nitpick_ignore: list[tuple[str, str]] = [
    ("py:class", "RGBInt"),
    ("py:class", "RenderMode"),
    ("py:class", "Coord"),
    ("py:class", "Lane"),
    ("py:class", "CellLane"),
    ("py:class", "ActType"),
    ("py:class", "ObsType"),
    ("py:class", "InfoDict"),
    ("py:class", "numberlink.env.InfoDict"),
    ("py:class", "numberlink.vector_env.InfoDict"),
    ("py:class", "numpy.typing.NDArray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.bool_"),
    ("py:class", "numpy.bool"),
    ("py:class", "numpy.uint8"),
    ("py:class", "numpy.int8"),
    ("py:class", "numpy.int16"),
    ("py:class", "numpy.intp"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.int64"),
    ("py:class", "NDArray"),
    ("py:class", "np.bool"),
    ("py:class", "np.uint8"),
    ("py:class", "np.integer"),
    ("py:class", "typing.Optional"),
    ("py:class", "optional"),
    ("py:class", "module"),
    ("py:class", "gymnasium.core.RenderFrame"),
    ("py:class", "gymnasium.core.ActType"),
    ("py:class", "gymnasium.core.ObsType"),
    ("py:class", "gymnasium.vector.AutoresetMode"),
    ("py:class", "gymnasium.Space"),
    ("py:class", "numberlink.types.RenderFrame"),
    ("py:class", "RenderFrame"),
    ("py:class", "VariantConfig"),
    ("py:class", "GeneratorConfig"),
    ("py:class", "RewardConfig"),
    ("py:class", "RenderConfig"),
    ("py:obj", "numpy.float32 | numpy.bool_"),
    ("py:mod", "gymnasium.vector"),
    ("py:func", "upscale_with_endpoint_borders"),
    ("py:func", "register_numberlink_v0"),
    ("py:func", "generate_level"),
    ("py:func", "build_level_template"),
    ("py:func", "main"),
    ("py:attr", "_closed"),
    ("py:attr", "_grid_codes"),
    ("py:attr", "_cell_switch_mask"),
    ("py:attr", "_heads"),
    ("py:attr", "_render_cfg"),
    ("py:attr", "_cell_action_size"),
    ("py:attr", "variant.cell_switching_mode"),
    ("py:attr", "variant.must_fill"),
    ("py:attr", "numberlink.env.NumberLinkRGBEnv._endpoint_mask"),
    ("py:attr", "numberlink.env.NumberLinkRGBEnv._palette_stack"),
    ("py:attr", "numberlink.env.NumberLinkRGBEnv._dirs"),
    ("py:data", "LEVELS"),
]


def link_to(fn: str, start: int | None = None, end: int | None = None) -> str | None:
    """Return a GitHub URL for the given file and line range, or None if not possible."""
    rel: str = Path(fn).resolve().relative_to(repo_root).as_posix()
    base: str = f"{links['GitHub']}/blob/main/{rel}"
    if start and end:
        # Use single line format for better auto-scroll, or range format for multi-line
        return f"{base}#L{start}" if start == end else f"{base}#L{start}-L{end}"
    return base


# Link to GitHub source per object
# Uses sphinx.ext.linkcode to link to line ranges on GitHub
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Return a GitHub URL for the documented object, or None if unknown."""
    if domain != "py" or not info.get("module"):
        return None

    try:
        mod: ModuleType = import_module(info["module"])
    except Exception:
        return None

    obj = mod
    fullname: str = info.get("fullname", "")

    # Navigate to the object
    for part in fullname.split(".") if fullname else []:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            # If we can't find the object, try to link to the module at least
            obj = mod
            break

    # First try: link to the exact object (works for functions/classes/methods)
    try:
        fn: str = inspect.getsourcefile(obj) or inspect.getfile(obj)  # may raise TypeError
        src: list[str]
        lineno: int
        src, lineno = inspect.getsourcelines(obj)  # may raise OSError/TypeError
        start: int = lineno
        end: int = lineno + max(len(src) - 1, 0)
        url: str | None = link_to(fn, start, end)
        if url:
            return url

    except (TypeError, OSError, AttributeError):
        pass  # not inspectable (typing alias, builtin, C ext, etc.)

    # Fallback: try to find the object's definition by searching the module source
    # This helps with module-level constants, attributes, properties, etc.
    try:
        mod_fn: str = inspect.getsourcefile(mod) or inspect.getfile(mod)
        if fullname and obj is not mod:
            # Try to find where this name is defined in the module
            with open(mod_fn, encoding="utf-8") as f:
                source_lines: list[str] = f.readlines()

            # Look for the attribute name at the module or class level
            # Matches various patterns:
            # - "name = ..." (simple assignment)
            # - "name: type = ..." (typed assignment)
            # - "class name" (class definition)
            # - "def name(" (function/method definition)
            # - "@property" followed by "def name" (property decorator)
            attr_name: str = fullname.rsplit(".", maxsplit=1)[-1]  # Get the last part of the name

            # Pattern to catch properties and decorated methods
            patterns: list[str] = [
                rf"^{attr_name}\s*[:=]",  # Direct assignment
                rf"^class\s+{attr_name}\b",  # Class definition
                rf"^def\s+{attr_name}\s*\(",  # Function/method definition
                rf"^\s+def\s+{attr_name}\s*\(",  # Indented method in class
                rf"^\s+{attr_name}\s*[:=]",  # Indented attribute in class
            ]

            # Also look for @property decorator above the method
            for i, line in enumerate(source_lines, start=1):
                stripped: str = line.strip()
                # Check all patterns
                for pattern in patterns:
                    if re.match(pattern, line):
                        return link_to(mod_fn, i, i)

                # Special case: check if this is a @property decorator
                # and the next line matches the method name
                if stripped.startswith("@") and i < len(source_lines):
                    next_line: str = source_lines[i]  # i is 1-indexed, list is 0-indexed, so i points to next
                    if re.match(rf"^\s+def\s+{attr_name}\s*\(", next_line):
                        # Link to the line with the decorator for better context
                        return link_to(mod_fn, i, i)

        # Last resort: link to the module file without line numbers
        return link_to(mod_fn)

    except Exception:
        return None


def replace_types_content(app: Sphinx) -> None:
    """Replace autodoc2-generated types.rst with our hand-written include.

    This runs after autodoc2 generates files, replacing the auto-generated
    content with our hand-written documentation.
    """
    stub: Path = here.parent / "apidocs" / "numberlink" / "numberlink.types.rst"
    if stub.exists():
        stub.write_text(".. include:: ../../_includes/types.rst\n", encoding="utf8")


def setup(app: Sphinx) -> None:
    """Add custom CSS and other assets to the Sphinx build."""
    app.add_css_file("custom.css")
    # Replace autodoc2's types.rst with our hand-written version after it generates files
    app.connect("builder-inited", replace_types_content)
