"""The view registry shared by the dev tools.

Every visualization in large_array_viz.py is a named module-level variable. This module
maps each name to a URL slug so the server, the checker, and the screenshotter all cover
the same set. Adding a view to the notebook means adding one name here.

A value may be either a HoloViews or Panel object, or a zero-argument callable that
builds one. Callables are used for views whose data is expensive, so importing the
notebook module stays cheap.

Names not yet defined in the notebook are reported and skipped, so the harness is usable
while the notebook is still being built.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import large_array_viz as lav  # noqa: E402

# Notebook section each view belongs to, in notebook order.
NAMES = [
    "fast_hvplot_view",
    "fast_holoviews_view",
    "image_raw_view",
    "image_view",
    "image_widgets_view",
    "image_levels_view",
    "traces_view",
    "traces_downsampled_view",
    "traces_minimap_view",
    "pyramid_app",
    "spike_raster_view",
    "spike_detail_view",
    "spike_field_view",
    "servable_app",
    "full_resolution_view",
]

# A name set to None is an optional view whose dependency is missing.
VIEWS = {
    name: getattr(lav, name)
    for name in NAMES
    if getattr(lav, name, None) is not None
}
MISSING = [name for name in NAMES if name not in VIEWS]


def resolve(name: str):
    """Return the view object, calling it first if it is a lazy builder."""
    view = VIEWS[name]
    return view() if callable(view) else view


def report_missing() -> None:
    if MISSING:
        print(f"not yet defined in large_array_viz.py: {', '.join(MISSING)}")
