"""Serve every visualization at its own URL from one process.

    python dev/serve_views.py

pn.serve accepts a dict whose keys become URL slugs, so each view gets an isolated
endpoint such as http://localhost:5007/image_view or http://localhost:5007/traces_view.
Working on one visualization then means loading one page, and layout_lint.py can be
pointed at a single endpoint rather than a page holding everything.

Each entry is wrapped in a zero-argument function so sessions do not share Bokeh models.
"""

from __future__ import annotations

import panel as pn
from views import VIEWS, report_missing, resolve

PORT = 5007


def _session(name: str):
    def build():
        return resolve(name)

    return build


if __name__ == "__main__":
    report_missing()
    apps = {name: _session(name) for name in VIEWS}
    print(f"serving {len(apps)} views on port {PORT}:")
    for name in apps:
        print(f"  http://localhost:{PORT}/{name}")
    pn.serve(apps, port=PORT, show=False)
