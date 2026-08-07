# Development harness

The committed notebook carries no outputs, so nothing here runs automatically when
someone opens it. Use `pixi run verify-notebook` to confirm it still executes.

None of this directory is published gallery content. It exists so every visualization in
`large_array_viz.py` can be inspected and verified without opening a notebook, and it can
be deleted without affecting the example.

The notebook assigns every visualization to a named module-level variable. That is what
makes this possible: `views.py` imports the notebook as a plain module and looks those
names up. Names not yet defined are reported and skipped, so the harness works while the
notebook is still being written.

## Tools

| Command | What it does |
| --- | --- |
| `pixi run check-views` | Headless correctness checks, no browser. Run this on every edit |
| `pixi run serve-views` | Serves each view at its own URL on port 5007 |
| `pixi run shoot-views [name ...]` | Screenshots served views to `/tmp/lav_<name>.png` |
| `pixi run preflight` | Static checks from the holoviz-skills Panel scripts |
| `pixi run lint-layout <url>` | Rendered-DOM layout and contrast checks |
| `python dev/make_thumbnails.py` | Rebuilds the screenshot previews in `assets/` |
| `python dev/make_diagrams.py` | Rebuilds the concept diagrams in `assets/` |
| `pixi run verify-notebook` | Executes a throwaway copy of the notebook and reports cell errors |

`assets/` is the one thing in this directory's output that **is** published. The notebook
embeds those files with relative paths, so they are committed. Regenerate them whenever a
visualization changes appearance: serve, shoot, then make thumbnails.

## Why the checks are layered this way

`check_views.py` is the primary tool because it is the only one that can verify behaviour.
A screenshot proves a plot drew something, not that it drew the right thing, and a plot
built from an empty selection looks much like a plot that is merely dark.

Tier 1 renders the Bokeh model and reads geometry off it. It catches the failures that are
invisible in an image: an axis range of `(0, 1)` where micrometers were expected, a
renderer count below the number of overlaid channels, a `data_aspect` option that was
silently dropped because it conflicted with responsive sizing.

Tier 2 drives the operations and streams from Python. It is what establishes that
downsampling holds the returned point count near the plot width instead of merely
returning fewer points, and that zooming moves the pyramid to a different level. Neither
claim is checkable from a picture, and both are the actual subject of the notebook.

Tier 3, the screenshots and `layout_lint`, judges what numbers cannot: whether a colormap
reads correctly, whether 35 stacked traces are legible, whether a layout looks intentional.
Use it at milestones rather than on every edit.

## Notes

`spike_field_view` uses a fixed frame width on purpose, because reading spikes against the
LFP requires the two plots to line up on screen and a responsive width cannot guarantee that.

`layout_lint` on `servable_app` reports violations in the `FastListTemplate` chrome, namely
the sidebar toggle, theme toggle, busy indicator, and title link. Those are Panel's own
components. Lint the individual view endpoints to check this notebook's content.

The image endpoints and `spike_field_view` are linted at `--widths 1400,1024`. A microscopy image with
`data_aspect=1` has a fixed pixel size, and below roughly 500 px it cannot both fit a
phone viewport and stay legible. Distorting it to fit would be worse than overflowing.

`HOLOVIZ_SKILLS` is set in `pixi.toml` and defaults to `$HOME/.kilo/skills`. Override it if
the skills collection is installed under `~/.claude/skills` or `~/.copilot/skills`.
