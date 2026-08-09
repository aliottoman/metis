"""Serving the Metis design language to a generated application's pages.

Every build used to invent its own visual language. The instruction said
"follow the frontend design language" and shipped nothing behind those words,
so each model reached for its own prior: one build came back warm and
minimal, the next cool and boxy, a third with a CDN framework. All of them
were defensible and none of them matched Metis.

The design language is now a file, vendored like every other piece of
infrastructure: ``appkit/static/theme.css``. A page links it and composes the
documented classes; it never declares its own colors, fonts, radii or
shadows. Because it lives under ``appkit/``, model writes to it are refused,
so a build cannot drift the theme by editing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Mounted away from the project's own /static so a build can keep writing its
# own assets there without colliding with — or overwriting — the theme.
MOUNT_PATH = "/appkit"
THEME_HREF = f"{MOUNT_PATH}/theme.css"

THEME_LINK = f'<link rel="stylesheet" href="{THEME_HREF}">'


def mount_appkit_static(app: Any) -> None:
    """Serve appkit's own static files (the theme) at ``/appkit``.

    Call once, next to where the application mounts its own static files::

        from appkit.web import mount_appkit_static
        mount_appkit_static(app)

    Then link ``/appkit/theme.css`` first in every page's ``<head>``.
    """
    from fastapi.staticfiles import StaticFiles

    app.mount(MOUNT_PATH, StaticFiles(directory=str(STATIC_DIR)), name="appkit")


def page(title: str, body: str, *, head: str = "") -> str:
    """A complete HTML document already wearing the design language.

    The whole point is that the parts a build gets wrong — the charset, the
    viewport, the theme link, the ``page`` wrapper — are not written by hand
    each time. ``body`` is the page's own markup, composed from the classes
    the theme documents.
    """
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"{THEME_LINK}\n"
        f"{head}\n"
        "</head>\n"
        '<body>\n<div class="page">\n'
        f"{body}\n"
        "</div>\n</body>\n</html>\n"
    )
