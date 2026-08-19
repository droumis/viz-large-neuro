"""Headless correctness checks for every registered view.

    python dev/check_views.py

No server and no browser, so this is the cheapest feedback available and the thing to run
on every edit. Two tiers:

Tier 1 renders the Bokeh model and reads geometry off it. A degenerate (0, 1) axis range
is the usual signature of an empty element, and a renderer count below the number of
overlaid elements means a layer was silently dropped.

Tier 2 drives the operations and streams directly to verify the behavioral claims the
notebook teaches. Downsampling holds the returned point count near the plot width, and
zooming changes the selected pyramid level. Neither claim is checkable from a screenshot.
"""

from __future__ import annotations

import sys
import traceback

import holoviews as hv
import numpy as np
import panel as pn
from views import VIEWS, report_missing, resolve

import large_array_viz as lav

FAILURES: list[str] = []


def check(label: str):
    """Run a check body, recording a failure instead of aborting the whole run."""

    def decorator(fn):
        try:
            fn()
        except Exception as exc:
            FAILURES.append(f"{label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"ok   {label}")
        return fn

    return decorator


def figures(obj) -> list:
    """Every Bokeh figure reachable from a HoloViews or Panel object.

    Templates and dynamic Tabs do not expose their contents as model children, so Panel
    objects are rendered into a Bokeh Document and the plots are selected from it.
    """
    from bokeh.document import Document
    from bokeh.models import Plot

    if isinstance(obj, (hv.core.Dimensioned, hv.core.spaces.DynamicMap)):
        return _descend(hv.render(obj, backend="bokeh"))

    if isinstance(obj, pn.template.base.BasicTemplate):
        doc = Document()
        obj.server_doc(doc=doc, title="check")
        return list(doc.select({"type": Plot}))

    model = obj.get_root() if isinstance(obj, pn.viewable.Viewable) else obj
    found = _descend(model)
    if not found and model.document is not None:
        found = list(model.document.select({"type": Plot}))
    return found


def _descend(model) -> list:
    from bokeh.models import GridPlot, LayoutDOM, Plot

    found: list = []
    seen: set[int] = set()

    def walk(node):
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Plot):
            found.append(node)
        children = []
        if isinstance(node, GridPlot):
            children = [child for child, *_ in node.children]
        elif isinstance(node, LayoutDOM):
            children = list(getattr(node, "children", []) or [])
        for child in children:
            walk(child if not isinstance(child, tuple) else child[0])

    walk(model)
    return found


def assert_range(fig, axis: str, low: float, high: float, label: str) -> None:
    rng = fig.x_range if axis == "x" else fig.y_range
    start, end = rng.start, rng.end
    assert start is not None and end is not None, f"{label}: {axis} range unset"
    assert (start, end) != (0, 1), f"{label}: degenerate {axis} range (0, 1), empty element"
    assert low <= start and end <= high, f"{label}: {axis} range {start}, {end} outside {low}, {high}"


# --- Tier 1: every view renders and carries a sensible model ------------------------


def tier1() -> None:
    for name in VIEWS:
        @check(f"tier1 renders {name}")
        def _(name=name):
            figs = figures(resolve(name))
            assert figs, f"{name}: no Bokeh figure in the rendered model"
            for fig in figs:
                for axis in ("x", "y"):
                    rng = fig.x_range if axis == "x" else fig.y_range
                    if rng.start is None:
                        continue
                    assert (rng.start, rng.end) != (0, 1), (
                        f"{name}: degenerate {axis} range, element is probably empty"
                    )


# --- Tier 1 specifics ---------------------------------------------------------------


def tier1_specifics() -> None:
    if "image_view" in VIEWS:
        @check("tier1 image carries micrometer coordinates")
        def _():
            fig = figures(resolve("image_view"))[0]
            # The sample is about 98 x 97 micrometers. Pixel indices would give a span of
            # 271 x 275, so this is what proves the OME coordinates survived the read.
            assert_range(fig, "x", -1, 120, "image_view")
            assert_range(fig, "y", -1, 120, "image_view")
            # invert_yaxis reverses the y range, so compare magnitudes.
            span_x = abs(fig.x_range.end - fig.x_range.start)
            span_y = abs(fig.y_range.end - fig.y_range.start)
            assert 50 < span_x < 120, f"x span {span_x} is not micrometers"
            assert 50 < span_y < 120, f"y span {span_y} is not micrometers"

        @check("tier1 image applies data_aspect, so a micrometer is square")
        def _():
            for name in ("image_raw_view", "image_view"):
                fig = figures(resolve(name))[0]
                assert fig.frame_width and fig.frame_height, (
                    f"{name}: frame not sized, so data_aspect cannot have been applied"
                )
                span_x = abs(fig.x_range.end - fig.x_range.start)
                span_y = abs(fig.y_range.end - fig.y_range.start)
                px_per_um = (fig.frame_width / span_x, fig.frame_height / span_y)
                ratio = px_per_um[0] / px_per_um[1]
                assert 0.98 < ratio < 1.02, (
                    f"{name}: {px_per_um[0]:.2f} px/um in x against"
                    f" {px_per_um[1]:.2f} in y, so the image is distorted"
                )

    if "traces_view" in VIEWS:
        @check("tier1 raw traces keep one renderer per channel")
        def _():
            fig = figures(resolve("traces_view"))[0]
            assert len(fig.renderers) == len(lav.CHANNELS), (
                f"{len(fig.renderers)} renderers for {len(lav.CHANNELS)} channels"
            )

    if "traces_downsampled_view" in VIEWS:
        @check("tier1 downsampled traces keep one renderer per channel")
        def _():
            fig = figures(resolve("traces_downsampled_view"))[0]
            assert len(fig.renderers) == len(lav.CHANNELS), (
                f"{len(fig.renderers)} renderers for {len(lav.CHANNELS)} channels"
            )

    if "traces_minimap_view" in VIEWS:
        @check("tier1 minimap layout has two figures and symmetric colour limits")
        def _():
            figs = figures(resolve("traces_minimap_view"))
            assert len(figs) == 2, f"expected traces plus minimap, got {len(figs)}"
            low, high = lav.MINIMAP_CLIM
            assert low == -high, f"minimap clim {lav.MINIMAP_CLIM} is not symmetric about zero"


# --- Tier 2: the behavioral claims --------------------------------------------------


def tier2() -> None:
    if hasattr(lav, "traces_overlay"):
        @check("tier2 downsampling holds the point count near the plot width")
        def _():
            from holoviews.operation.downsample import downsample1d

            width = 800
            t0, t1 = float(lav.LFP_TIME[0]), float(lav.LFP_TIME[-1])
            raw = len(next(iter(lav.traces_overlay)))
            reduced = downsample1d(
                lav.traces_overlay, dynamic=False, width=width, x_range=(t0, t1),
                algorithm="minmax-lttb",
            )
            counts = [len(curve) for curve in reduced]
            assert len(counts) == len(lav.CHANNELS), (
                f"{len(counts)} curves survived for {len(lav.CHANNELS)} channels"
            )
            for count in counts:
                assert count <= width * 4, f"{count} points returned for width {width}"
                assert count < raw / 10, f"{count} points is not a reduction from {raw}"

            # A tenth of the range should return a comparable count, not a tenth of one.
            narrow = downsample1d(
                lav.traces_overlay, dynamic=False, width=width,
                x_range=(t0, t0 + (t1 - t0) / 10), algorithm="minmax-lttb",
            )
            narrow_counts = [len(curve) for curve in narrow]
            assert min(narrow_counts) > width / 2, (
                f"zoomed in to a tenth of the range returned only {min(narrow_counts)}"
                f" points for a {width} pixel plot"
            )
            print(
                f"     raw {raw:,} -> {counts[0]} points at width {width},"
                f" {narrow_counts[0]} points over a tenth of the range"
            )

    if hasattr(lav, "pyramid_dmap"):
        @check("tier2 zoom walks down the levels and reaches full resolution")
        def _():
            dmap = lav.pyramid_dmap
            t0 = float(lav.PYRAMID_TIME[0])
            t1 = float(lav.PYRAMID_TIME[-1])
            mid = (t0 + t1) / 2
            width = 1000
            lav.pyramid_size_stream.event(width=width, height=600)

            # Progressively narrower ranges should walk monotonically towards the finest
            # level, which is the whole point of the selection rule.
            records = []
            for span in (t1 - t0, 400.0, 20.0, 1.0):
                low = t0 if span == t1 - t0 else mid
                lav.PYRAMID_TRACE.clear()
                dmap.event(x_range=(low, low + span))
                record = lav.PYRAMID_TRACE[-1]
                record["span"] = span
                record["name"] = str(record["level"])
                records.append(record)

            indexes = [r["level"] for r in records]
            assert indexes == sorted(indexes, reverse=True), (
                f"levels {indexes} are not monotonically finer as the range narrows"
            )
            assert len(set(indexes)) > 2, f"only {len(set(indexes))} distinct levels used"

            for record in records:
                assert record["samples"] >= width or record["level"] == 0, (
                    f"level {record['name']} returned {record['samples']} samples"
                    f" for a {width} pixel plot"
                )
                assert record["samples"] < width * 10, (
                    f"level {record['name']} returned {record['samples']} samples,"
                    f" far more than the {width} pixel plot needs"
                )

            assert records[-1]["level"] == 0, (
                f"a one second window chose level {records[-1]['level']},"
                " not the full-resolution level 0"
            )

            for record in records:
                print(
                    f"     {record['span']:>9.1f} s span -> level"
                    f" {record['name']:>4}, {record['samples']:>5} samples"
                )

    if hasattr(lav, "image_z_slider"):
        @check("tier2 setting the z widget loads a different plane")
        def _():
            def aggregated():
                fig = figures(lav.image_widgets_view)[0]
                assert len(fig.renderers) == 1, f"{len(fig.renderers)} renderers"
                return np.asarray(fig.renderers[0].data_source.data["image"][0])

            original = lav.image_z_slider.value
            before = aggregated()
            lav.image_z_slider.value = original + 40
            after = aggregated()
            assert before.shape == after.shape, (before.shape, after.shape)
            assert not np.array_equal(before, after), "plane unchanged after moving z"
            lav.image_z_slider.value = original

    if hasattr(lav, "image_norm_toggle"):
        @check("tier2 the normalization toggle switches between two mapper types")
        def _():
            eq_pane, linear_pane = lav.image_widgets_view[1], lav.image_widgets_view[2]
            mappers = [
                type(figures(pane)[0].renderers[0].glyph.color_mapper).__name__
                for pane in (eq_pane, linear_pane)
            ]
            assert "EqHist" in mappers[0], f"eq_hist pane uses {mappers[0]}"
            assert mappers[1] == "LinearColorMapper", f"linear pane uses {mappers[1]}"

            # Asserted between assignments, so batching these would defeat the check.
            for value, expected in [
                ("eq_hist", (True, False)),
                ("linear", (False, True)),
                ("eq_hist", (True, False)),
            ]:
                lav.image_norm_toggle.value = value
                assert (eq_pane.visible, linear_pane.visible) == expected, (
                    f"{value} gave visibility {eq_pane.visible}, {linear_pane.visible}"
                )

    if hasattr(lav, "spike_raster_view"):
        @check("tier2 the raster hands over from precomputed counts to exact events")
        def _():
            # The callback is driven directly rather than through the stream. A stream event
            # is deduplicated when the new range matches the one already in use, and the
            # claim under test belongs to the selection logic, not the stream wiring, which
            # the pyramid check above already covers.
            t0, t1 = float(lav.SPIKE_TIMES[0]), float(lav.SPIKE_TIMES[-1])
            mid = (t0 + t1) / 2

            records = []
            for span in (t1 - t0, 400.0, 2.0):
                low = t0 if span > 5000 else mid
                lav.SPIKE_TRACE.clear()
                element = lav.spike_raster(
                    x_range=(low, low + span), width=900, height=420
                )
                assert element is not None
                records.append(lav.SPIKE_TRACE[-1])

            assert "precomputed" in records[0]["source"], (
                f"the whole recording used {records[0]['source']}"
            )
            for record in records[1:]:
                assert "exact" in record["source"], (
                    f"a {record['span']:.0f} s span used {record['source']}"
                )
            # Slicing is the point: visible counts must fall with the span.
            counts = [r["visible"] for r in records]
            assert counts == sorted(counts, reverse=True), counts
            assert counts[-1] < counts[0] / 1000, (
                f"{counts[-1]} visible at 2 s against {counts[0]} at full extent"
            )
            for record in records:
                print(
                    f"     {record['span']:>9.1f} s span -> {record['visible']:>9,}"
                    f" visible, {record['source']}"
                )

        @check("tier2 each unit occupies a band of pixel rows, not a single row")
        def _():
            # Guards the choice of Segments over Points. Points put every spike on exactly
            # one pixel row, leaving a striped plot with empty gaps between units.
            t0 = float(lav.SPIKE_TIMES[0])
            plot = hv.renderer("bokeh").get_plot(lav.spike_raster_view)
            assert plot is not None, "raster did not render"
            lav.spike_size_stream.event(width=600, height=lav.N_UNITS * 3)
            lav.spike_raster_view.event(x_range=(t0, t0 + 200.0))
            fig = plot.state
            image = np.nan_to_num(
                np.asarray(fig.renderers[0].data_source.data["image"][0])
            )
            filled = int((image.sum(axis=1) > 0).sum())
            assert filled > lav.N_UNITS * 2, (
                f"only {filled} of {image.shape[0]} pixel rows carry data for"
                f" {lav.N_UNITS} units, so spikes are landing on single rows"
            )
            print(f"     {filled} of {image.shape[0]} rows filled for {lav.N_UNITS} units")

        @check("tier2 the colorbar label follows the regime the raster switched to")
        def _():
            # HoloViews sets the colorbar title only at plot creation (holoviz/holoviews#5977),
            # so this passes only while the sync_clabel hook is attached. No single screenshot
            # shows it, since the label is only wrong relative to the frame beside it.
            t0, t1 = float(lav.SPIKE_TIMES[0]), float(lav.SPIKE_TIMES[-1])
            mid = (t0 + t1) / 2
            plot = hv.renderer("bokeh").get_plot(lav.spike_raster_view)
            assert plot is not None, "raster did not render"

            # A DynamicMap serves a repeated stream value from its cache without calling the
            # callback, which would leave nothing new in SPIKE_TRACE to compare against. So no
            # span here repeats another, and none is the full extent the plot already rendered
            # with, which also keeps this check independent of the ones above it.
            seen = []
            for low, high in ((t0, t1 - 0.5), (mid, mid + 2.0), (t0, t1 - 1.5)):
                lav.SPIKE_TRACE.clear()
                lav.spike_raster_view.event(x_range=(low, high))
                assert lav.SPIKE_TRACE, f"a {high - low:.0f} s span rendered from cache"
                source = lav.SPIKE_TRACE[-1]["source"]
                title = plot.handles["colorbar"].title
                # The invariant is asserted rather than the exact wording, so rewording a
                # clabel in the notebook does not fail this check with a message that blames
                # the colorbar. A precomputed frame is a rate and has to carry a frequency
                # unit; an exact frame is a per-pixel count and must not.
                precomputed = "precomputed" in source
                assert ("Hz" in title) == precomputed, (
                    f"a {high - low:.0f} s span read from {source} but the colorbar"
                    f" reads {title!r}"
                )
                seen.append(title)
            # A hook that only ever fires on the first frame would pass the checks above.
            assert len(set(seen)) == 2, f"the colorbar label never changed: {seen}"
            print(f"     colorbar tracked {seen[0]!r} -> {seen[1]!r} -> {seen[2]!r}")

    if getattr(lav, "servable_app", None) is not None:
        @check("tier2 the served app exposes both tabs and a plot in the active one")
        def _():
            tabs = lav.servable_app.main[0]
            titles = [pane.name for pane in tabs]
            assert titles == ["Imaging", "Electrophysiology", "Spikes"], titles
            # dynamic=True renders only the active tab, so one figure is the expectation.
            assert figures(lav.servable_app), "no plot rendered in the active tab"

    if hasattr(lav, "image_level_selector"):
        @check("tier2 setting the level selector changes the displayed array shape")
        def _():
            shapes = []
            for level in lav.IMAGE_LEVELS:
                lav.image_level_selector.value = level
                shapes.append(lav.image_level_shape())
            assert len(set(shapes)) == len(shapes), f"level shapes not distinct: {shapes}"
            print(f"     level shapes {shapes}")


if __name__ == "__main__":
    report_missing()
    tier1()
    tier1_specifics()
    tier2()
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  {failure}")
        sys.exit(1)
    print("\nall view checks passed")
