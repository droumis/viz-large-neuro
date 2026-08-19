# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive Visualization of Large Array Data
#
# Reading a terabyte of array data efficiently is a solved problem. Chunk it, compress it, and
# let Dask read only the blocks a computation touches. Displaying it is not, because a plotting
# library asked to draw the result will try to send every value to the browser.
#
# This notebook shows how to plot arrays far larger than the screen, reducing them at three
# points in the path from storage to pixels, across three kinds of neuroscience data. Fluorescence microscopy is read from a
# remote OME-Zarr store, a local field potential recording is read from a multiscale pyramid, and
# nine million spike times are read as a point process.
#
# > **This notebook downloads and writes about 2.9 GB.** One file of 857 MB is downloaded, a
# > 2.1 GB pyramid is generated on first run, and 88 MB of spike times are streamed from a remote
# > archive. Each step is cached and reports its size before it runs. See the README for the
# > full footprint.

# %% [markdown]
# ## Why large arrays need reducing
#
# <p><img src="assets/concept_escalation_compact.svg" alt="Three tiers of reduction, cheapest first" width="560"></p>
#
# However large an array is, a plot has only as many pixels as the window gives it, perhaps a
# thousand or two across a laptop screen. A browser also starts to struggle somewhere between one
# hundred thousand and one million glyphs, because every glyph carries coordinates, style
# attributes, and hit-testing geometry. Neither limit grows when the data does. A single 400
# second window of the recording used below holds 17.5 million points, which is well beyond what
# a browser will draw at interactive speed, and that window is a twenty-fourth of the recording.
#
# The remedy is the one that governs chunked reads, which is to keep only the subset you need.
# The difference at the display step is that the viewport defines the subset, so it has to be
# recomputed whenever the user pans or zooms.
#
# Reduction can happen at three places, and the diagram above summarises them. They are ordered
# by what they cost to set up, so the useful question is always whether the first is already
# enough.
#
# 1. **Reduce what is sent to the browser.** Either aggregate the data into an image the size of
#    the plot, which is *rasterization*, or select the few original samples that preserve the
#    shape of a line, which is *downsampling*. These two are alternatives, and which one applies
#    depends on the kind of data. For many plots this tier is sufficient on its own.
# 2. **Reduce what is read from storage.** A *multiscale pyramid* stores several resolutions in
#    advance, so a viewport becomes a read of the matching level rather than a reduction of the
#    whole array. This becomes necessary once the widest view cannot be reduced quickly enough, or
#    once the array no longer fits in memory.
# 3. **Reduce what is stored.** A *reference* records where bytes already sit in an archive
#    instead of copying them. This tier is different in kind from the other two, because it
#    changes nothing about what is drawn. It is included because a pyramid otherwise duplicates
#    data that already exists, and because reading an archive without copying it is a common need
#    in its own right.
#
# The second tier has a useful property. A pyramid written so that a large image can be loaded at
# all is the same object as a pyramid that makes zooming responsive, so no separate structure is
# needed for display.

# ## Prerequisites
#
# | What? | Why? |
# | --- | --- |
# | [OME-Zarr specification](https://ngff.openmicroscopy.org/latest/) | The imaging data is read as a multiscale OME-Zarr group |
# | [OME-Zarr textbook](https://ome-zarr-book.readthedocs.io/) | Background on multiscale arrays and the metadata this notebook relies on |
# | [Zarr](https://zarr.readthedocs.io/) | Chunked array storage, read here over HTTP and from local disk |
# | [Dask arrays](https://docs.dask.org/en/stable/array.html) | Lazy evaluation, so only the slice on screen is loaded |
# | [Xarray](https://docs.xarray.dev/) | Labeled dimensions and physical coordinates |

# %% [markdown]
# ## Plotting a small array
#
# An unconfigured plot of a labeled array is the starting point that the rest of this notebook
# departs from. Both calls below produce a pannable, zoomable, hoverable Bokeh
# plot from a Dask-backed xarray object without any plotting configuration.

# %%
import shutil
import time
from pathlib import Path

import colorcet  # perceptually uniform colormaps, also registered by name for cmap=
import dask.array as dsa
import h5py
import holoviews as hv
import hvplot.pandas  # noqa: F401 registers .hvplot on DataFrames
import hvplot.xarray  # noqa: F401 registers .hvplot on xarray objects
import numpy as np
import pandas as pd
import panel as pn
import pooch
import xarray as xr
import zarr
from holoviews.operation.datashader import rasterize
from holoviews.operation.downsample import downsample1d
from holoviews.plotting.links import RangeToolLink
from scipy.stats import zscore
from xarray_ome_ngff import DaskArrayWrapper, read_multiscale_group

hv.extension("bokeh")
pn.extension(throttled=True)

# %% [markdown]
# The demonstration array here is simulated two-photon fluorescence, six regions of
# interest sampled at 30 Hz for a minute. It is small on purpose, so the plot below is the
# baseline that the rest of the notebook departs from.

# %%
rng = np.random.default_rng(0)
n_frames, n_rois = 1800, 6
frame_times = np.arange(n_frames) / 30

spikes = rng.random((n_frames, n_rois)) < 0.004
decay = np.exp(-np.arange(120) / 20)
traces = np.stack(
    [np.convolve(spikes[:, i], decay)[:n_frames] for i in range(n_rois)], axis=1
)
traces += rng.normal(0, 0.02, traces.shape)

demo = xr.DataArray(
    traces,
    coords={"time": frame_times, "roi": [f"roi {i}" for i in range(n_rois)]},
    dims=("time", "roi"),
    name="fluorescence",
).chunk({"roi": 1})
demo

# %%
fast_hvplot_view = demo.hvplot.line(
    x="time", by="roi", responsive=True, height=300, xlabel="time (s)",
    ylabel="fluorescence (a.u.)",
)
fast_hvplot_view

# %% [markdown]
# The same plot built from HoloViews elements takes more lines and gives more control. The
# reason to use it here is that the two reduction strategies are HoloViews
# Operations, and an Operation is a function from an element to a transformed element. The
# input to `rasterize` or `downsample1d` is an element like the one below, and the output
# is another element that a plot can render. Nothing about the element changes, so the
# reduction can be added to or removed from a working plot without rewriting it.

# %%
time_dim = hv.Dimension("time", unit="s")
fluorescence_dim = hv.Dimension("fluorescence", unit="a.u.")

fast_holoviews_view = (
    hv.Dataset(demo, ["time", "roi"], [fluorescence_dim])
    .to(hv.Curve, [time_dim])
    .overlay("roi")
    .opts("Curve", responsive=True, height=300)
    .opts("NdOverlay", legend_position="right")
)
fast_holoviews_view

# %% [markdown]
# ## Rasterizing an image
#
# <p><img src="assets/concept_rasterize.svg" alt="A dense cloud of values aggregated into a coarse grid, and the grid sent to a browser" width="460"></p>
#
# The imaging data is a two channel fluorescence volume from the Image Data Resource,
# stored as OME-Zarr 0.4 on the EMBL-EBI Embassy object store and read directly over HTTP.
# Nothing is downloaded. The store is already a multiscale pyramid, so the coarse levels
# arrive quickly.
#
# > **Nothing is downloaded in this section.** The volume is read over HTTP, a plane at a time.
#
# The read goes through `xarray-ome-ngff` rather than opening the arrays with Zarr
# directly. A bare Zarr array carries no coordinates, so the OME scale and translation
# metadata would be discarded and the axes would be pixel indices. Reading through a
# coordinate-aware library turns that metadata into real coordinates in micrometers,
# which is what makes the axis labels, the aspect ratio, and any physical measurement on
# the plot meaningful.

# %%
OME_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"

ome_group = zarr.open_group(OME_URL, mode="r")

# The store keeps one xy plane per chunk, shape (1, 1, y, x). Matching that means
# selecting a plane costs exactly one request instead of stitching many partial ones.
ome_levels = read_multiscale_group(
    ome_group, array_wrapper=DaskArrayWrapper(chunks=(1, 1, -1, -1))
)
# Sorted numerically, since level keys are strings and a lexicographic sort would put "10"
# before "2".
IMAGE_LEVELS = sorted(ome_levels, key=int)

for level in IMAGE_LEVELS:
    array = ome_levels[level]
    spacing = {
        dim: float(array[dim][1] - array[dim][0]) for dim in ("z", "y", "x")
    }
    print(
        f"level {level}: shape {array.shape} {array.dtype}"
        f"  z {spacing['z']:.4f} y {spacing['y']:.4f} x {spacing['x']:.4f} um"
        f"  {array.nbytes / 1e6:.1f} MB"
    )

# %% [markdown]
# The axis order is `(c, z, y, x)`, so a channel and a z index have to be chosen before
# there is a 2D plane to draw. Voxel spacing is about 0.5 micrometers in z against about
# 0.36 in y and x, so the volume is anisotropic by a factor of 1.4. That does not affect
# the plane selected below, which is xy, but any orthogonal view through this volume needs
# its aspect ratio corrected or the morphology will be wrong.

# %%
x_dim = hv.Dimension("x", unit="µm")
y_dim = hv.Dimension("y", unit="µm")
intensity_dim = hv.Dimension("intensity", unit="counts")

MID_Z = ome_levels["0"].sizes["z"] // 2


def image_plane(level="0", channel=0, z=MID_Z):
    """Load one 2D plane, keeping its micrometer coordinates."""
    return ome_levels[level].isel(c=channel, z=z).rename("intensity").load()


def image_element(plane):
    return hv.Image(plane, kdims=[x_dim, y_dim], vdims=[intensity_dim])


plane = image_plane()
print(f"one plane is {plane.size:,} pixels, the whole level 0 array is {ome_levels['0'].size:,} voxels")

# %% [markdown]
# Fluorescence intensity is shown dark to bright, which maps signal to perceived
# luminance and is how single channel fluorescence is conventionally read. A rainbow
# colormap would imply an ordering that intensity does not have and would introduce
# false edges where its hue changes fastest. `data_aspect=1` keeps one micrometer in y
# equal to one micrometer in x so the morphology is not stretched, and `invert_yaxis`
# puts the first row at the top, matching how the acquisition software displays it.
#
# The plot below sends every pixel of the plane to the browser.

# %%
# frame_height sizes the data area, and data_aspect then derives the frame width from the
# coordinates, so one micrometer measures the same on both axes. Combining data_aspect with
# responsive sizing does not achieve this: the aspect ratio is applied to the whole plot
# including the colorbar and axes, which leaves the data area stretched.
IMAGE_OPTS = dict(
    cmap="kgy",
    data_aspect=1,
    invert_yaxis=True,
    colorbar=True,
    frame_height=400,
)

IMAGE_CLIM = (0.0, float(np.percentile(plane.values, 99.5)))

image_raw_view = image_element(plane).opts(
    **IMAGE_OPTS, clim=IMAGE_CLIM, title="every pixel sent",
)
image_raw_view

# %% [markdown]
# Wrapping the same element in `rasterize` moves the reduction to the server. Datashader
# aggregates the array into a grid the size of the plot frame, and the browser receives
# that grid instead of the underlying pixels. The transfer is then fixed by the plot size
# no matter how large the array is, and zooming re-runs the aggregation over the new
# viewport so detail appears as it becomes resolvable.
#
# This plane is small enough that both versions render, and that is what makes them
# comparable. The difference matters at the scale where the raw version stops working. A
# single plane of a whole-brain acquisition is often 20,000 by 20,000 pixels, which is
# 400 million values, and the rasterized call is unchanged.
#
# Both plots use the same linear color limits, so the only thing that differs between them
# is where the reduction happened.

# %%
image_view = rasterize(image_element(plane)).opts(
    **IMAGE_OPTS, clim=IMAGE_CLIM, title="rasterized, plot-sized transfer",
)
image_view

# %% [markdown]
# Channel and z index select which plane is loaded, so they belong to the data and are
# wired through a `DynamicMap`, which patches the plot in place and therefore preserves
# the current zoom.
#
# Normalization is handled differently. `cnorm='eq_hist'` applies a histogram-equalizing
# transform, which spreads the available colors over the range where values actually fall
# and so reveals dim structure next to bright structure. It is nonlinear, so displayed
# brightness stops being proportional to intensity, and the uneven colorbar ticks are the
# visible symptom. The linear version fixes `clim` at the 99.5th percentile instead, which
# is what to use when the reader has to judge relative intensity.
#
# Both versions are built up front from the same plane and the toggle chooses which one is
# shown. Switching `cnorm` on a single existing plot would ask it to replace its color
# mapper with one of a different type, which currently raises in HoloViews 1.23.1.
#
# Only one channel is shown at a time here. Compositing both would call for green and
# magenta rather than green and red, since that pair stays distinguishable under the
# common forms of color vision deficiency and the overlap reads as white.

# %%
n_z = ome_levels["0"].sizes["z"]

# Widgets are given an explicit width and the column a little more, because a column sized
# exactly to its widget clips the widget border.
WIDGET_WIDTH = 220
WIDGET_COLUMN_WIDTH = 250

image_channel_selector = pn.widgets.Select(
    name="channel", options=[0, 1], value=0, width=WIDGET_WIDTH
)
image_z_slider = pn.widgets.IntSlider(
    name="z index", start=0, end=n_z - 1, value=MID_Z, width=WIDGET_WIDTH
)
image_norm_toggle = pn.widgets.RadioButtonGroup(
    name="normalization", options=["eq_hist", "linear"], value="eq_hist",
    width=WIDGET_WIDTH,
)


def image_plane_element(channel, z):
    return image_element(image_plane("0", channel, z))


image_plane_dmap = hv.DynamicMap(
    pn.bind(image_plane_element, image_channel_selector, image_z_slider)
)

image_eq_plot = rasterize(image_plane_dmap).opts(
    **IMAGE_OPTS, cnorm="eq_hist", title="eq_hist, structure over quantity",
)
image_linear_plot = rasterize(image_plane_dmap).opts(
    **IMAGE_OPTS, cnorm="linear", clim=IMAGE_CLIM, title="linear, quantity preserved",
)

image_widgets_view = pn.Row(
    pn.Column(
        image_channel_selector, image_z_slider, image_norm_toggle,
        width=WIDGET_COLUMN_WIDTH,
    ),
    pn.pane.HoloViews(
        image_eq_plot,
        visible=pn.bind(lambda norm: norm == "eq_hist", image_norm_toggle),
    ),
    pn.pane.HoloViews(
        image_linear_plot,
        visible=pn.bind(lambda norm: norm == "linear", image_norm_toggle),
    ),
)
image_widgets_view

# %% [markdown]
# ## Reading a coarser level of an existing pyramid
#
# Everything so far read level 0. The store has three levels, and the reason that matters is
# that rasterization and pyramids solve two different halves of the same problem.
#
# `rasterize` caps what crosses the network to the browser. It does not cap what crosses the
# network to the notebook. Aggregating a zoomed-out view of level 0 still reads every
# full-resolution chunk in view, throws most of the detail away, and sends a small image. The
# transfer to the browser is small and the read behind it is not.
#
# Choosing a coarser level fixes the other half. Nothing does this automatically. Datashader
# is handed an array and does not know the array belongs to a hierarchy, so selecting the
# level is the caller's job.

# %%
for level in IMAGE_LEVELS:
    array = ome_levels[level]
    started = time.perf_counter()
    array.isel(c=0, z=MID_Z).load()
    elapsed = time.perf_counter() - started
    plane_pixels = array.sizes["y"] * array.sizes["x"]
    print(
        f"level {level}: {array.sizes['y']:>3} x {array.sizes['x']:>3}"
        f" = {plane_pixels:>7,} pixels, loaded in {elapsed:.2f}s"
    )

# %% [markdown]
# Only y and x are coarsened. Every level keeps the full set of z planes, which is usual for
# microscopy pyramids, and it means a given z index refers to the same physical depth at every
# level.
#
# The plot below reads whichever level is selected. Level 2 is a sixteenth of the pixels of
# level 0, and at the size it is displayed the difference is hard to see, which is the entire
# argument for reading it instead. Zoom in and the argument reverses, and that is the tradeoff
# a viewer has to manage by requesting a finer level as the viewport narrows.

# %%
image_level_selector = pn.widgets.Select(
    name="pyramid level", options=IMAGE_LEVELS, value=IMAGE_LEVELS[-1],
    width=WIDGET_WIDTH,
)


def image_level_shape():
    """The 2D shape that the selected level currently supplies."""
    level = image_level_selector.value
    return (ome_levels[level].sizes["y"], ome_levels[level].sizes["x"])


def image_level_element(level):
    plane = image_plane(level, 0, MID_Z)
    height, width = plane.shape
    return image_element(plane).opts(
        **IMAGE_OPTS, clim=IMAGE_CLIM, title=f"level {level}, {height} x {width} pixels",
    )


image_levels_view = pn.Row(
    pn.Column(image_level_selector, width=WIDGET_COLUMN_WIDTH),
    pn.pane.HoloViews(
        rasterize(hv.DynamicMap(pn.bind(image_level_element, image_level_selector)))
    ),
)
image_levels_view

# %% [markdown]
# ## Downsampling timeseries
#
# <p><img src="assets/concept_downsample.svg" alt="Many samples per pixel reduced to the minimum and maximum at each pixel" width="460"></p>
#
# Rasterization is the wrong tool for a small number of long traces. Aggregating a line
# into a density image discards the line, and for a voltage trace the shape of the line is
# the measurement. What is needed instead is a subset of the original samples that draws
# the same shape.
#
# Largest Triangle Three Buckets, described by [Steinarsson
# (2013)](https://skemman.is/handle/1946/15343), divides the series into as many buckets as
# there are horizontal pixels and keeps the one point per bucket that forms the largest
# triangle with its neighbours. Points that do not contribute to the visible shape are
# dropped. The `minmax-lttb` variant first reduces each bucket to its minimum and maximum
# before running LTTB, which is faster and preserves the envelope, so a brief spike that
# happens to fall between selected points is not lost.
#
# The recording is extracellular local field potential from a Neuropixels probe in mouse,
# released by the Allen Institute as part of the Visual Coding Neuropixels dataset.
#
# The files are read straight from the Allen Institute's public S3 bucket, and the two small
# index files there make that generalizable. `sessions.csv` is 7.8 KB and lists every released
# session; `probes.csv` is 27 KB and lists every probe, including whether it has LFP data.
# Together they mean any session or probe in the release can be reached by changing two
# integers, without installing the AllenSDK or downloading a catalogue.

# %%
ALLEN_ECEPHYS = (
    "https://allen-brain-observatory.s3.amazonaws.com/"
    "visual-coding-neuropixels/ecephys-cache"
)
SESSION_ID = 754312389
PROBE_ID = 756781563

sessions = pd.read_csv(f"{ALLEN_ECEPHYS}/sessions.csv")
probes = pd.read_csv(f"{ALLEN_ECEPHYS}/probes.csv")
print(f"{len(sessions)} sessions and {len(probes)} probes in the release")

session_row = sessions[sessions["id"] == SESSION_ID].iloc[0]
probe_row = probes[probes["id"] == PROBE_ID].iloc[0]
print(
    f"session {SESSION_ID}: {session_row.session_type},"
    f" {session_row.sex}, {session_row.age_in_days:.0f} days, {session_row.genotype}"
)
print(
    f"probe {PROBE_ID} ({probe_row['name']}): AP band {probe_row.sampling_rate:,.0f} Hz,"
    f" LFP band {probe_row.lfp_sampling_rate:,.1f} Hz,"
    f" subsampled by {probe_row.lfp_temporal_subsampling_factor:.0f}"
)

# Every file in the release follows this layout, so any other session or probe is one edit away.
LFP_URL = f"{ALLEN_ECEPHYS}/session_{SESSION_ID}/probe_{PROBE_ID}_lfp.nwb"
SESSION_URL = f"{ALLEN_ECEPHYS}/session_{SESSION_ID}/session_{SESSION_ID}.nwb"

DATA_DIR = Path("data")
LFP_PATH = DATA_DIR / Path(LFP_URL).name

# The index reports the LFP band before subsampling, so the rate stored in the file is the
# quotient. This is the number the data below is checked against.
EXPECTED_RATE = probe_row.lfp_sampling_rate / probe_row.lfp_temporal_subsampling_factor
print(f"expected rate in the file: {EXPECTED_RATE:.3f} Hz")

# %% [markdown]
# The AP band is worth noticing in that output. The probe sampled at 30 kHz, and what is
# distributed as the LFP band was low-pass filtered and decimated to 2500 Hz, then subsampled
# again by two. The archive therefore already ships a coarse level of a pyramid that the
# acquisition pipeline wrote, which is the same idea a later section applies deliberately.
#
# > **The next cell downloads 857 MB** into `data/`, once, and is skipped if the file is already
# > present. Change `SESSION_ID` and `PROBE_ID` above to fetch a different recording.

# %%
DATA_DIR.mkdir(parents=True, exist_ok=True)
if LFP_PATH.exists():
    print(f"data exists at {LFP_PATH}")
else:
    print(f"downloading {LFP_URL}")
    pooch.retrieve(
        url=LFP_URL, known_hash=None, fname=LFP_PATH.name, path=DATA_DIR,
        progressbar=True,
    )

# %% [markdown]
# The LFP array is in the NWB acquisition group alongside its two dimension vectors, and
# the electrode rows index a separate table that carries the identifiers, the position of
# each site along the probe shank, and the brain region it was assigned to. Reading that
# table is what turns an anonymous channel index into something interpretable.
#
# Amplitudes are stored in volts. Local field potential is conventionally reported in
# microvolts, so the values are scaled on the way in and the axis is labeled to match. The
# file records its own unit, so that scaling is asserted rather than assumed, because a plot
# labeled in the wrong unit is worse than an unlabeled one.

# %%
# Derived from PROBE_ID, so switching probe changes nothing else. Note that
# acquisition/probe_<id>_lfp_data is an HDF5 soft link; the real group is one level deeper,
# which matters later when a reader that does not follow links is used.
LFP_SERIES = f"acquisition/probe_{PROBE_ID}_lfp_data"
LFP_DATA_KEY = f"{LFP_SERIES}/data"
LFP_TIME_KEY = f"{LFP_SERIES}/timestamps"
LFP_ELECTRODE_KEY = f"{LFP_SERIES}/electrodes"
ELECTRODE_TABLE = "general/extracellular_ephys/electrodes"

WINDOW_START_S = 5000.0
WINDOW_DURATION_S = 400.0

with h5py.File(LFP_PATH, "r") as handle:
    # Recorded in the file itself, so there is no need to trust a comment for either of
    # these. conversion is the factor NWB expects a reader to apply before the unit holds.
    assert handle[LFP_DATA_KEY].attrs["unit"] == "volts"
    assert handle[LFP_DATA_KEY].attrs["conversion"] == 1.0
    assert handle[LFP_TIME_KEY].attrs["unit"] == "seconds"

    all_times = handle[LFP_TIME_KEY][:]
    start, stop = np.searchsorted(
        all_times, [WINDOW_START_S, WINDOW_START_S + WINDOW_DURATION_S]
    )
    LFP_TIME = all_times[start:stop]
    # Volts to microvolts, checked against the recorded amplitudes rather than assumed.
    window = handle[LFP_DATA_KEY][start:stop, :] * 1e6

    electrode_rows = handle[LFP_ELECTRODE_KEY][:]
    table = handle[ELECTRODE_TABLE]
    electrode_ids = table["id"][:][electrode_rows]
    electrode_depths = table["probe_vertical_position"][:][electrode_rows]
    electrode_regions = [
        name.decode() or "unassigned" for name in table["location"][:][electrode_rows]
    ]

CHANNELS = [str(identifier) for identifier in electrode_ids]
LFP_WINDOW = pd.DataFrame(
    window, index=pd.Index(LFP_TIME, name="time"), columns=CHANNELS
)

SAMPLING_RATE = 1 / np.median(np.diff(LFP_TIME))
assert abs(SAMPLING_RATE - EXPECTED_RATE) < 0.1, (
    f"file reports {SAMPLING_RATE:.3f} Hz, index implies {EXPECTED_RATE:.3f} Hz"
)
print(f"{LFP_WINDOW.shape[0]:,} samples x {LFP_WINDOW.shape[1]} channels"
      f" = {LFP_WINDOW.size:,} points at {SAMPLING_RATE:.0f} Hz")
print(f"amplitude range {LFP_WINDOW.values.min():.1f} to {LFP_WINDOW.values.max():.1f} µV")
print(f"site positions {electrode_depths.min()} to {electrode_depths.max()} µm along the"
      f" shank, spanning {', '.join(dict.fromkeys(electrode_regions))}")

# %% [markdown]
# The channels are kept in probe order, which is the order the sites occur along the shank.
# The vertical axis of the plot below is therefore anatomical depth, running from
# hippocampal CA1 at the bottom through to visual cortex at the top, and that is the reason
# a stacked layout is the right idiom for this data rather than a stylistic preference.
# Sorting the channels any other way would destroy the spatial structure that makes a
# travelling wave or a laminar current sink visible.
#
# Each column of the DataFrame carries a different channel name, and a channel name is not
# a measurable quantity. Mapping every column to a shared `amplitude` label gives the
# vertical axis a meaning, and gives the hover readout a unit.
#
# `subcoordinate_y` gives each curve its own slice of the vertical axis, so a channel with
# small amplitude stays readable next to one with large amplitude, and it makes the y zoom
# tool act on one channel at a time instead of on the whole stack. Traces are thin black
# lines because channel identity is already carried by vertical position, and coloring them
# by a palette would imply an ordering the colors do not have.

# %%
REPRESENTATIVE_CHANNEL = CHANNELS[len(CHANNELS) // 2]

CURVE_OPTS = dict(
    subcoordinate_y=True,
    subcoordinate_scale=3,
    color="black",
    line_width=1,
    hover_tooltips=[("channel", "$label"), ("time"), ("amplitude")],
    tools=["xwheel_zoom"],
    active_tools=["box_zoom"],
)

# Absolute amplitude matters here, so one channel carries a scale bar that the toolbar
# ruler icon toggles on.
SCALEBAR_OPTS = dict(
    scalebar=True,
    scalebar_location="right",
    scalebar_unit=("µV", "V"),
    scalebar_opts={
        "bar_line_width": 3,
        # The label sits to the right of the bar, in the margin. To its left it would
        # overlap the traces it is measuring.
        "label_location": "right",
        # Nearly opaque, because the bar sits over the traces it is measuring and a
        # translucent background left the label unreadable.
        "background_fill_color": "white",
        "background_fill_alpha": 0.95,
        "length_sizing": "adaptive",
        "bar_length_units": "data",
        "bar_length": 0.8,
    },
)

def build_curves(frame):
    """One Curve per channel, each mapped to a shared amplitude dimension."""
    curves = {}
    for channel in frame.columns:
        amplitude_dim = hv.Dimension(channel, label="amplitude", unit="µV")
        extra = SCALEBAR_OPTS if channel == REPRESENTATIVE_CHANNEL else {}
        curves[channel] = hv.Curve(
            frame, [time_dim], [amplitude_dim], group="LFP", label=channel
        ).opts(**CURVE_OPTS, **extra)
    return curves

# An empty title is deliberate. The default for an overlay of grouped, labeled curves
# concatenates every label, which for 35 channels is an unreadable strip of text.
OVERLAY_OPTS = dict(
    title="",
    ylabel="electrode, deep to superficial",
    show_legend=False,
    padding=0,
    min_height=700,
    responsive=True,
)

traces_overlay = hv.Overlay(
    build_curves(LFP_WINDOW), kdims=[time_dim, "Channel"]
).opts(**OVERLAY_OPTS)

# %% [markdown]
# The overlay above is not displayed, for the reason this section exists. It holds
# 17.5 million points, roughly sixty times what a browser will draw at interactive speed.
# Rendering it takes minutes and transfers well over a hundred megabytes of JSON, and there
# are only about 1500 horizontal pixels to draw it in, so around ten thousand samples land
# on each horizontal pixel, and all but a few of them are painted over.
#
# What is displayed instead is the same construction over the first four seconds. That is
# 175,000 points, still above what a browser handles comfortably, and already sluggish to
# pan. Multiply by a hundred to recover the full window.
#
# Four seconds rather than forty is also a practical choice. Every point drawn this way is
# stored in the notebook file, and this one cell accounts for a few megabytes of it. Raise
# `EXCERPT_S` to observe the degradation directly.

# %%
EXCERPT_S = 4.0
excerpt = LFP_WINDOW.loc[: WINDOW_START_S + EXCERPT_S]

traces_view = hv.Overlay(build_curves(excerpt), kdims=[time_dim, "Channel"]).opts(
    **OVERLAY_OPTS
)
print(f"excerpt: {excerpt.size:,} points against {LFP_WINDOW.size:,} in the full window")
traces_view

# %% [markdown]
# `downsample1d` is applied once to the composed overlay rather than to each curve, so a
# single slice of the shared time index serves every channel. It is a HoloViews Operation,
# so the input is the full window overlay built above and the output is an overlay that
# renders the same way. The full 400 seconds is now interactive.

# %%
traces_downsampled = downsample1d(traces_overlay, algorithm="minmax-lttb")
traces_downsampled_view = traces_downsampled
traces_downsampled_view

# %% [markdown]
# The number of points sent is now set by the width of the plot rather than by the length
# of the recording, and it stays there as you zoom. Zooming in does not reveal more of the
# same points, it triggers a new selection over the narrower range, so detail sharpens
# instead of merely magnifying.
#
# A minimap makes the zoomed-in view navigable by showing the whole window alongside it. This
# is the case rasterization was built for. The z-scored amplitude matrix is 17.5 million
# cells, there is no line to preserve, and density across channels is exactly what the
# reader needs, so it is rasterized into a plot-sized image.
#
# Z scores are signed and centred on zero, so the colormap is diverging and the limits are
# symmetric, which puts zero at the neutral midpoint. Histogram equalization is deliberately
# not used here, because it would move the color midpoint away from zero, and a diverging map
# would then show a change of sign where there is none.
#
# Symmetric limits still have to be chosen against the data, and the relevant data is what
# rasterization produces rather than what went into it. Datashader aggregates with a mean by
# default, and the whole window compressed to about 1300 pixels wide puts 380 samples behind
# each one. Averaging 380 samples of a zero-mean oscillation largely cancels it, so the
# displayed values are far smaller than the raw z scores. Limits taken from the raw values
# would leave the minimap almost uniformly white.
#
# The limits below are therefore computed from a block mean at the width the minimap is
# actually drawn at. Zero stays at the midpoint, and the contrast lands where the aggregated
# values are. What the minimap shows is the local mean of the signal, not its envelope, and
# which has to be known before structure is read into it.

# %%
channel_positions = range(len(CHANNELS))
channel_ticks = [(i, channel) for i, channel in enumerate(CHANNELS)]
z_scored = zscore(LFP_WINDOW.values, axis=0).T

MINIMAP_WIDTH = 1300
per_column = z_scored.shape[1] // MINIMAP_WIDTH
aggregated = z_scored[:, : per_column * MINIMAP_WIDTH].reshape(
    len(CHANNELS), MINIMAP_WIDTH, per_column
).mean(axis=2)

minimap_limit = float(np.percentile(np.abs(aggregated), 98))
MINIMAP_CLIM = (-minimap_limit, minimap_limit)
print(
    f"{per_column} samples behind each pixel shrinks the 98th percentile of |z|"
    f" from {np.percentile(np.abs(z_scored), 98):.2f} to {minimap_limit:.2f}"
)

minimap = rasterize(
    hv.Image(
        (LFP_TIME, channel_positions, z_scored), [time_dim, "Channel"], ["amplitude"]
    )
).opts(
    cmap="RdBu_r",
    clim=MINIMAP_CLIM,
    colorbar=False,
    xlabel="",
    yticks=[channel_ticks[0], channel_ticks[-1]],
    toolbar="disable",
    height=120,
    responsive=True,
)

# The box on the minimap sets the initial view of the main plot. Thirty five channels in
# 700 pixels leaves 20 pixels each, so the view opens on half of them and the minimap is
# how you reach the rest.
RangeToolLink(
    minimap,
    traces_downsampled,
    axes=["x", "y"],
    boundsx=(WINDOW_START_S, WINDOW_START_S + WINDOW_DURATION_S / 8),
    boundsy=(-0.5, len(CHANNELS) // 2 + 0.5),
    # The box is dragged rather than resized, because resize handles on a box this short are
    # difficult to grab and the vertical extent is meant to stay fixed.
    use_handles=False,
)

traces_minimap_view = (traces_downsampled + minimap).opts(shared_axes=False).cols(1)
traces_minimap_view

# %% [markdown]
# ## Writing a multiscale pyramid
#
# <p><img src="assets/concept_pyramid.svg" alt="Stored resolution levels covering the same extent, with the zoom selecting one" width="460"></p>
#
# Downsampling worked because the whole 400 second window was already in memory, so
# selecting from it was cheap. The full recording is 12,094,337 samples across 35 channels,
# which is 1.7 GB as float32. Loading that to select 1500 points from it, on every zoom, is
# wasteful, and it becomes impossible once the recording is longer than memory.
#
# A pyramid inverts the order. The reduction is done once, ahead of time, at a handful of
# fixed resolutions. Interaction then becomes a read of the level that matches the current
# zoom, which is a slice of a small array rather than a reduction of a large one.
#
# Both ends of the range of levels are chosen against the display rather than picked for
# neatness. The coarsest level has to be coarse enough that the entire recording fits on
# screen, and all 9675 seconds in a 1500 pixel plot means about 1500 samples, so a factor of
# 4096 leaves 2952. Stopping at 256 would still hand the browser 47,000 samples for the full
# view. The finest level is the original 1250 Hz, so that zooming in far enough reaches real
# samples rather than running out of resolution.
#
# Including full resolution makes the pyramid a complete, self-contained copy. It accounts for
# 1.7 GB of the 2.1 GB the store occupies on disk, and in return the pyramid continues to work
# if the NWB file is moved, archived, or deleted. The cell below prints the compressed size, so
# the figure quoted here can be checked against it.

# > **The next cell writes 2.1 GB** into `data/` and takes roughly a minute. It is skipped if
# > the pyramid is already present.

# %%
from ndpyramid import pyramid_coarsen  # noqa: E402 kept next to the code that uses it

PYRAMID_PATH = DATA_DIR / "lfp_pyramid.zarr"
FACTORS = [1, 4, 16, 64, 256, 1024, 4096]

# Chunk length along time for the written levels. The source file is chunked one channel at a
# time over 47,244 samples, which suits writing a recording and not reading 35 channels at
# once. Rechunking on the way in is what makes a narrow window cheap to read later.
LEVEL_CHUNK = 16384

if PYRAMID_PATH.exists():
    print(f"pyramid exists at {PYRAMID_PATH}")
else:
    print(f"building pyramid at {PYRAMID_PATH}, this takes a few minutes")
    with h5py.File(LFP_PATH, "r") as handle:
        # The HDF5 dataset is wrapped rather than read. Nothing is loaded until to_zarr
        # pulls it chunk by chunk, so peak memory stays at a few chunks.
        lfp = dsa.from_array(handle[LFP_DATA_KEY], chunks=(LEVEL_CHUNK * 32, -1)) * 1e6
        source = xr.Dataset(
            {"lfp": (("time", "channel"), lfp)},
            coords={"time": handle[LFP_TIME_KEY][:], "channel": CHANNELS},
        )
        pyramid = pyramid_coarsen(
            source, factors=FACTORS, dims=["time"], boundary="trim"
        )

        # Coarsening leaves each level with ragged dask chunks, which Zarr will not accept,
        # and the right chunk length differs per level anyway. Rechunking per level fixes
        # both, and caps the coarse levels at their own length so they stay single chunks.
        levels = {}
        for level in pyramid.children:
            data = pyramid[level].to_dataset()
            length = data.sizes["time"]
            levels[f"/{level}"] = data.chunk(
                {"time": min(LEVEL_CHUNK, length), "channel": -1}
            )
        rechunked = xr.DataTree.from_dict(levels)
        rechunked.attrs.update(pyramid.attrs)

        # Written to a temporary path and renamed, so an interrupted build cannot leave a
        # half-finished store that the existence check above would then trust.
        staging = PYRAMID_PATH.with_name(PYRAMID_PATH.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        rechunked.to_zarr(staging, mode="w", consolidated=True)
        staging.rename(PYRAMID_PATH)

# %% [markdown]
# The result is a DataTree, one child per level, read back lazily. `chunks={}` keeps the
# arrays as Dask arrays backed by the store rather than loading them, so opening the pyramid
# costs a metadata read regardless of its size.

# %%
pyramid_tree = xr.open_datatree(PYRAMID_PATH, engine="zarr", chunks={})
PYRAMID_LEVELS = sorted(pyramid_tree.children, key=int)

for level, factor in zip(PYRAMID_LEVELS, FACTORS):
    data = pyramid_tree[level].ds["lfp"]
    times = pyramid_tree[level].ds["time"].values
    rate = 1 / np.median(np.diff(times))
    print(
        f"level {level}: factor {factor:>4}, {len(times):>10,} samples, {rate:8.3f} Hz,"
        f" chunks {data.data.chunksize}, {data.nbytes / 1e6:7.1f} MB"
    )
stored_mb = sum(p.stat().st_size for p in PYRAMID_PATH.rglob("*")) / 1e6
print(f"\n{stored_mb:.0f} MB on disk after compression")

# %% [markdown]
# ### Choosing a level
#
# Given the visible time range and the width of the plot in pixels, the best level is the
# coarsest one that still supplies at least one sample per pixel of width. Anything coarser
# leaves visible gaps, and anything finer sends points that land on a pixel already covered.
#
# The search runs from coarsest to finest and takes the first level that qualifies. Because
# the finest level is the original data, a request can always be satisfied.

# %%
# Time vectors are held in memory, because choosing a level means counting samples in a
# range and a searchsorted on an in-memory vector is what makes that instant. All seven
# together are 130 MB, which is the price of instant level selection.
LEVEL_TIMES = {level: pyramid_tree[level].ds["time"].values for level in PYRAMID_LEVELS}
LEVEL_RATES = {
    level: 1 / np.median(np.diff(times)) for level, times in LEVEL_TIMES.items()
}
PYRAMID_TIME = LEVEL_TIMES[PYRAMID_LEVELS[0]]

# Records which level each call selected. The notebook never reads it, but dev/check_views.py
# asserts against it, which is how the level-selection behaviour is verified.
PYRAMID_TRACE = []


def read_level(level, low, high):
    """The window from one level, as a time by channel frame in microvolts."""
    return pyramid_tree[level].ds["lfp"].sel(time=slice(low, high)).to_pandas()


def samples_in_range(level, low, high):
    times = LEVEL_TIMES[level]
    return int(np.searchsorted(times, high) - np.searchsorted(times, low))


def choose_level(low, high, width):
    for level in reversed(PYRAMID_LEVELS):
        if samples_in_range(level, low, high) >= width:
            return level
    return PYRAMID_LEVELS[0]


def pyramid_overlay(x_range=None, width=None, height=None, **unused):
    width = width or 1000
    low, high = x_range or (PYRAMID_TIME[0], PYRAMID_TIME[-1])
    level = choose_level(low, high, width)

    frame = read_level(level, low, high)
    PYRAMID_TRACE.append(
        {"level": int(level), "samples": len(frame), "x_range": (low, high)}
    )

    return hv.Overlay(build_curves(frame), kdims=[time_dim, "Channel"]).opts(
        **{
            **OVERLAY_OPTS,
            "title": f"level {level}, coarsened by {FACTORS[int(level)]},"
                     f" {LEVEL_RATES[level]:.3g} Hz,"
                     f" {len(frame):,} samples per channel",
        }
    )


pyramid_range_stream = hv.streams.RangeX(
    x_range=(WINDOW_START_S, WINDOW_START_S + WINDOW_DURATION_S)
)
pyramid_size_stream = hv.streams.PlotSize()
pyramid_dmap = hv.DynamicMap(
    pyramid_overlay, streams=[pyramid_range_stream, pyramid_size_stream]
)

pyramid_app = pn.pane.HoloViews(pyramid_dmap, sizing_mode="stretch_width")
pyramid_app

# %% [markdown]
# Zooming in selects progressively finer levels, and the finest holds the original samples. Two
# seconds
# from level 0 is below, where individual cycles and their phase offset between neighbouring
# sites are visible. None of that is recoverable from the 312 Hz level, which is what including
# storing full resolution provides.

# %%
full_resolution_view = downsample1d(
    hv.Overlay(
        build_curves(read_level(PYRAMID_LEVELS[0], WINDOW_START_S, WINDOW_START_S + 2.0)),
        kdims=[time_dim, "Channel"],
    ).opts(**{**OVERLAY_OPTS, "title": "level 0, the original 1250 Hz"}),
    algorithm="minmax-lttb",
)
full_resolution_view

# %% [markdown]
# ## Referencing an archive instead of copying it
#
# The pyramid above holds a complete second copy of the recording, and that is worth
# questioning. The bytes already exist in the NWB file, compressed and chunked. Copying them
# to get a Zarr store means paying for the same samples twice.
#
# A virtual Zarr store avoids that. What a Zarr reader needs is not the bytes but a description
# of where each chunk begins and ends, and `virtualizarr` produces exactly that by reading the
# HDF5 chunk index and writing a manifest of paths, offsets, and lengths. Any Zarr reader then
# treats the manifest as an array, and the original file is never modified or duplicated.
#
# This is the mainstream use of the technique. It began with
# [Kerchunk](https://fsspec.github.io/kerchunk/) and runs at serious scale, including a virtual
# view over the 115 TB GOES-16 archive amounting to 7.1 billion chunk references.
#
# Two details about this file are easy to miss, and both are typical of NWB rather than specific
# to it. The group path is not the one used earlier, because
# `acquisition/probe_756781563_lfp_data` is an HDF5 soft link and the real group is one level
# deeper. h5py follows the link silently and virtualizarr does not. The `electrodes` variable
# also has to be dropped, because NWB is HDF5 without the NetCDF dimension-naming convention,
# so the reader invents names and gives the same one to the 12 million sample axis and the 35
# element electrode axis.

# %%
try:
    from virtualizarr import open_virtual_dataset

    HAVE_VIRTUALIZARR = True
except ImportError:
    HAVE_VIRTUALIZARR = False
    print("virtualizarr is not installed, so this section is skipped")

MANIFEST_PATH = DATA_DIR / "lfp_virtual.json"
LFP_GROUP = f"acquisition/probe_{PROBE_ID}_lfp/probe_{PROBE_ID}_lfp_data"

if HAVE_VIRTUALIZARR:
    virtual = open_virtual_dataset(
        str(LFP_PATH),
        group=LFP_GROUP,
        indexes={},
        loadable_variables=[],
        drop_variables=["electrodes"],
    )
    virtual.virtualize.to_kerchunk(str(MANIFEST_PATH), format="json")

    references = len(virtual["data"].data.manifest.dict())
    manifest_mb = MANIFEST_PATH.stat().st_size / 1e6
    print(f"{references:,} chunk references, each a path, an offset, and a length")
    print(
        f"manifest {manifest_mb:.2f} MB, against {12_094_337 * 35 * 4 / 1e6:.0f} MB"
        f" for the copy in level 0 of the pyramid"
    )

# %% [markdown]
# Read back through the reference filesystem, the manifest behaves like any other lazy Zarr
# array. The values are checked against both the NWB file and the pyramid rather than assumed
# to match, because a manifest pointing at the wrong offsets would produce plausible looking
# noise.
#
# `mask_and_scale=False` is required here, and omitting it corrupts data silently. The HDF5
# dataset declares a fill value of 0.0, which the manifest faithfully records as the Zarr
# `fill_value`. Xarray then applies CF conventions by default, which treat the fill value as
# missing data, so every real sample that happens to be exactly zero is replaced by NaN. On
# this file that silently corrupts about half a percent of the samples. Nothing warns, and the
# result still plots.

# %%
if HAVE_VIRTUALIZARR:
    virtual_store = xr.open_dataset(
        "reference://",
        engine="zarr",
        mask_and_scale=False,
        backend_kwargs={
            "consolidated": False,
            "storage_options": {"fo": str(MANIFEST_PATH)},
        },
        chunks={},
    )
    virtual_lfp = virtual_store["data"].rename(
        {"phony_dim_0": "time", "phony_dim_1": "channel"}
    )

    with h5py.File(LFP_PATH, "r") as handle:
        from_hdf5 = handle[LFP_DATA_KEY][:2000, :]
    from_virtual = virtual_lfp.isel(time=slice(0, 2000)).values
    from_pyramid = read_level(PYRAMID_LEVELS[0], all_times[0], all_times[2000]).values

    print(f"virtual array {virtual_lfp.shape} {virtual_lfp.dtype}")
    print(f"chunk shape {virtual_lfp.data.chunksize}, inherited from the HDF5 layout")
    print(f"matches the NWB file:  {np.array_equal(from_virtual, from_hdf5)}")
    print(
        "matches the pyramid:   "
        f"{np.allclose(from_virtual * 1e6, from_pyramid[:2000], rtol=1e-5)}"
    )
    assert np.array_equal(from_virtual, from_hdf5), "manifest does not match the source"

# %% [markdown]
# Three limits of the technique explain why the written copy is kept.
#
# The chunk shape is inherited and cannot be changed, because the compressed units are stored
# inside the original file. Here that shape is one channel wide and 47,244 samples long, so a window
# across all 35 channels touches 35 chunks however short the window is. Drawing two seconds
# decompresses about 6.6 MB to produce 350 KB of samples. The rechunked level 0 written above
# reads the same window from one or two chunks instead.
#
# A manifest cannot supply a resolution that was never written. Coarsened values are the output
# of a computation and exist nowhere until something stores them, so the coarse levels have to
# be written whatever happens at the fine end. That also explains why the imaging store earlier
# in this notebook needed no such treatment, since it ships its own written levels and every
# one of them is read in place.
#
# The reference is only as durable as the path inside it. Move or rewrite the NWB file and the
# manifest silently points at whatever now occupies those offsets.
# [Icechunk](https://icechunk.io) is the production answer to that last problem, since it stores
# virtual chunk references alongside native ones and records the modification time of each
# referenced chunk so that a stale reference raises instead of returning wrong bytes. It
# requires Zarr 3, which cannot coexist with `xarray-ome-ngff` in this environment.
#
# Use a manifest when the archive is large enough that copying it is the cost you object
# to, and when its chunking already suits the way you intend to read it. At 1.7 GB, and with a
# chunk shape that does not match the access pattern, writing the level was preferable here.

# %%
if HAVE_VIRTUALIZARR:
    for seconds in (0.5, 2.0, 8.0):
        first, last = np.searchsorted(
            all_times, [WINDOW_START_S, WINDOW_START_S + seconds]
        )
        started = time.perf_counter()
        virtual_lfp.isel(time=slice(first, last)).values
        virtual_s = time.perf_counter() - started

        started = time.perf_counter()
        read_level(PYRAMID_LEVELS[0], WINDOW_START_S, WINDOW_START_S + seconds)
        pyramid_s = time.perf_counter() - started
        print(
            f"{seconds:4} s window: manifest {virtual_s:.3f} s,"
            f" rechunked level 0 {pyramid_s:.3f} s"
        )



# %% [markdown]
# ## Spikes
#
# Images are fields, the LFP traces are lines, and spikes are a third shape. A spike train is a
# set of instants, with no value at any of them and nothing defined between them, and that
# changes which strategies are even available.
#
# Most importantly, downsampling has no analogue here. LTTB works by keeping the samples that
# preserve the shape of a line, and a point process has no line. Reducing one means either
# counting events in bins, which is rasterization by another name, or discarding events, which
# misrepresents the data rather than summarising it. Of the two options in the first tier, only
# rasterization applies.
#
# The units below come from the same probe as the LFP, so they sit at the same depths, and they
# are ordered along the shank exactly as the channels were. That makes the vertical axis
# anatomical in both plots and lets them be read against each other.
#
# The spike times are in the session-level NWB file, `SESSION_URL` from earlier, which is
# 1.86 GB. Only about 74 MB of it is needed. The bucket serves byte range requests, so the file
# is opened over HTTP and only the units table and the relevant spike times are transferred.

# > **The next cell transfers about 74 MB** over HTTP and writes an 88 MB cache into `data/`.
# > The remote file is 1.86 GB and is not downloaded in full. It is skipped if the cache is
# > already present.

# %%
SPIKE_PATH = DATA_DIR / f"spikes_probe_{PROBE_ID}.npz"

if SPIKE_PATH.exists():
    print(f"spikes exist at {SPIKE_PATH}")
else:
    import fsspec

    print(f"streaming the units table from {SESSION_URL}")
    remote = h5py.File(fsspec.filesystem("http").open(SESSION_URL, block_size=8 * 2**20), "r")

    table = remote["general/extracellular_ephys/electrodes"]
    on_probe = table["probe_id"][:] == PROBE_ID
    depth_of = dict(zip(table["id"][:][on_probe], table["probe_vertical_position"][:][on_probe]))
    region_of = dict(
        zip(table["id"][:][on_probe],
            [name.decode() for name in table["location"][:][on_probe]])
    )

    units = remote["units"]
    peak_channel = units["peak_channel_id"][:]
    ends = units["spike_times_index"][:]
    begins = np.concatenate([[0], ends[:-1]])
    selected = np.flatnonzero(np.isin(peak_channel, list(depth_of)))
    # Ordered along the shank, so the raster's vertical axis is depth rather than unit id.
    selected = selected[np.argsort([depth_of[peak_channel[i]] for i in selected])]

    all_spikes = units["spike_times"]
    times, unit_index = [], []
    for order, row in enumerate(selected):
        block = all_spikes[begins[row]:ends[row]]
        times.append(block)
        unit_index.append(np.full(block.size, order, np.int16))
    times = np.concatenate(times)
    unit_index = np.concatenate(unit_index)

    # Sorted by time, which is what makes the viewport slice below a binary search.
    order = np.argsort(times, kind="stable")
    np.savez(
        SPIKE_PATH,
        times=times[order],
        unit=unit_index[order],
        unit_id=units["id"][:][selected],
        depth=np.array([depth_of[peak_channel[i]] for i in selected]),
        region=np.array([region_of[peak_channel[i]] for i in selected]),
        firing_rate=units["firing_rate"][:][selected],
    )
    remote.close()

spikes = np.load(SPIKE_PATH)
SPIKE_TIMES = spikes["times"]
SPIKE_UNIT = spikes["unit"].astype(np.float64)
UNIT_DEPTH = spikes["depth"]
UNIT_REGION = spikes["region"]
UNIT_RATE = spikes["firing_rate"]
UNIT_IDS = spikes["unit_id"]
N_UNITS = len(UNIT_DEPTH)

assert np.all(np.diff(SPIKE_TIMES) >= 0), "spike times must be sorted for the slice below"
print(f"{SPIKE_TIMES.size:,} spikes from {N_UNITS} units")
print(f"depths {UNIT_DEPTH.min()} to {UNIT_DEPTH.max()} µm, "
      f"regions {dict(zip(*np.unique(UNIT_REGION, return_counts=True)))}")
print(f"firing rates {UNIT_RATE.min():.2f} to {UNIT_RATE.max():.1f} Hz")

# %% [markdown]
# Three choices decide whether a rasterized raster is readable, and the first two are easy to
# get wrong.
#
# **The primitive.** A spike drawn as a point lands on exactly one pixel row, so with 180 units
# in a 400 pixel plot only 180 of the 400 rows receive anything and the raster comes out as
# stripes separated by empty gaps. Drawing each spike as a short vertical segment spanning its
# own row instead fills the row whatever the pixel density is. It is also the fastest of the
# options here, and `hv.Spikes` is not one of them, because it draws from a baseline up to the
# value and aggregating that fills everything below each unit.
#
# **The vertical budget.** Row identity survives only while the plot has at least as many pixel
# rows as there are units. At 90 pixels for 180 units, two units share every row and the plot
# stops being a raster and becomes a population density map. That is a hard limit, not a
# setting, and it is why a raster of a few thousand units needs scrolling rather than a taller
# figure.
#
# **What gets handed to the aggregator.** Datashader's cost scales with the events it is given,
# not the events on screen, so passing the whole array on every zoom costs the same 0.45 s
# whether the viewport holds 9 million events or 2000. Because the times are sorted, a binary
# search narrows the array to the viewport first, and the aggregation then costs what is
# visible.

# %%
EXACT_LIMIT = 1_000_000  # above this many visible events, read the precomputed counts instead

# The coarse level, which is the pyramid idea applied to a derived field rather than to the
# events themselves. Binning a point process is the only way to reduce it, so the coarse level
# is a count matrix. At 2048 bins it is finer than the plot is wide and costs under 2 MB.
COARSE_BINS = 2048
coarse_edges = np.linspace(SPIKE_TIMES[0], SPIKE_TIMES[-1], COARSE_BINS + 1)
coarse_counts, _, _ = np.histogram2d(
    SPIKE_TIMES, SPIKE_UNIT,
    bins=[coarse_edges, np.arange(N_UNITS + 1) - 0.5],
)
COARSE_SECONDS = float(np.diff(coarse_edges)[0])
# Stored as a rate rather than a count. A count depends on the bin width, so it would mean
# something different at every zoom, whereas a rate in hertz is a property of the units.
coarse_rate = (coarse_counts / COARSE_SECONDS).astype(np.float32)
coarse_centres = (coarse_edges[:-1] + coarse_edges[1:]) / 2

RATE_CEILING = float(np.percentile(coarse_rate[coarse_rate > 0], 95))
print(f"coarse matrix {coarse_rate.shape}, {coarse_rate.nbytes / 1e6:.1f} MB, "
      f"{COARSE_SECONDS:.2f} s per bin, colour ceiling {RATE_CEILING:.0f} Hz")

unit_dim = hv.Dimension("unit", label="unit, deep to superficial")
rate_dim = hv.Dimension("rate", label="firing rate", unit="Hz")
count_dim = hv.Dimension("count", label="spikes per pixel")

# Region boundaries make the vertical axis readable, since a unit index means nothing on its
# own and the two regions are contiguous once units are ordered by depth.
region_ticks = [
    (float(np.flatnonzero(UNIT_REGION == region).mean()), region)
    for region in dict.fromkeys(UNIT_REGION)
]

# A list of two colours would be a two entry palette, not a ramp, and Bokeh would then step
# hard at the midpoint instead of shading. This is a full 256 step reversed grey.
RASTER_CMAP = colorcet.gray[::-1]


# Copies the frame's `clabel` onto the Bokeh colorbar, which HoloViews sets only at plot
# creation (holoviz/holoviews#5977). Without this the raster labels spikes per pixel as hertz.
def sync_clabel(plot, element):
    colorbar = plot.handles.get("colorbar")
    if colorbar is not None and plot.clabel is not None:
        colorbar.title = plot.clabel


RASTER_OPTS = dict(
    cmap=RASTER_CMAP,
    cnorm="linear",
    colorbar=True,
    hooks=[sync_clabel],
    responsive=True,
    min_height=420,
    ylim=(-0.5, N_UNITS - 0.5),
    yticks=region_ticks,
)

# Records which source each call used, for the same reason as PYRAMID_TRACE above.
SPIKE_TRACE = []


def spike_raster(x_range=None, width=None, height=None, **unused):
    width = width or 1000
    height = min(height or 420, N_UNITS * 8)
    low, high = x_range or (SPIKE_TIMES[0], SPIKE_TIMES[-1])
    first, last = np.searchsorted(SPIKE_TIMES, [low, high])
    visible = int(last - first)

    if visible > EXACT_LIMIT:
        lo_bin, hi_bin = np.searchsorted(coarse_centres, [low, high])
        lo_bin, hi_bin = max(lo_bin - 1, 0), min(hi_bin + 1, COARSE_BINS)
        element = hv.Image(
            (coarse_centres[lo_bin:hi_bin], np.arange(N_UNITS),
             coarse_rate[lo_bin:hi_bin].T),
            [time_dim, unit_dim], [rate_dim],
        )
        # Regridding an image averages rates, so the value stays in hertz.
        extra = dict(clim=(0, RATE_CEILING), clabel="firing rate (Hz)")
        source = f"precomputed rate, {hi_bin - lo_bin} bins of {COARSE_SECONDS:.1f} s"
    else:
        times, units = SPIKE_TIMES[first:last], SPIKE_UNIT[first:last]
        element = hv.Segments(
            (times, units - 0.45, times, units + 0.45),
            [time_dim, unit_dim, "time_end", "unit_end"], [count_dim],
        )
        # Counting segments per pixel, which at this zoom is one spike or none.
        extra = dict(clim=(0, 1), clabel="spikes per pixel")
        source = f"{visible:,} exact events"

    SPIKE_TRACE.append({"visible": visible, "source": source, "span": high - low})
    return rasterize(
        element, dynamic=False, width=width, height=height,
        x_range=(low, high), y_range=(-0.5, N_UNITS - 0.5),
    ).opts(**RASTER_OPTS, **extra, title=f"{source}, {high - low:.4g} s in view")


spike_range_stream = hv.streams.RangeX()
spike_size_stream = hv.streams.PlotSize()
spike_raster_view = hv.DynamicMap(
    spike_raster, streams=[spike_range_stream, spike_size_stream]
)
spike_raster_view

# %% [markdown]
# The two regimes are not two renderings of the same quantity, and pretending otherwise would
# be the easiest mistake to make here. Zoomed out, the value is a firing rate in hertz, which is
# a property of each unit and is therefore comparable between one zoom level and another. Zoomed
# in, the value is a count of spikes per pixel, which is zero or one, and dividing that by the
# pixel's width in time would report several hundred hertz for a single spike. So the colorbar
# changes what it measures when the source changes, and the title says which one is in use.
#
# Making the colorbar say so takes a hook. HoloViews writes the colorbar title when the plot is
# created and never rewrites it, so a per-frame `clabel` on its own leaves the label on whichever
# regime rendered first while the numbers beside it change meaning (holoviz/holoviews#5977).
# `clim` needs no hook and switches between the two ranges on its own.
#
# Colour is linear in both regimes, with an explicit ceiling at the 95th percentile of the
# non-zero rates, so twice as dark means twice the rate. `eq_hist` would have revealed more of
# the low-rate units at the cost of that reading, and for a quantity as routinely compared as
# firing rate it is not worth the trade.
#
# One detail that is easy to get wrong. A colormap given as two colours is a two entry palette
# rather than a ramp, and Bokeh then steps abruptly at the midpoint instead of shading, which
# looks like a thresholded image. `RASTER_CMAP` is a full 256 step reversed grey.
#
# What rasterizing gives up is per-event identity. No spike can be hovered and no unit can be
# selected, because by the time it reaches the browser it is an image. The usual answer is the
# two-regime pattern this notebook already used for the LFP, where a rasterized overview hands
# over to real glyphs once the viewport is narrow enough to afford them. Below is that vector
# form, twenty units over two seconds, where each spike is its own mark and can be hovered.

# %%
DETAIL_UNITS = 20
DETAIL_START, DETAIL_SECONDS = WINDOW_START_S, 2.0

detail_first, detail_last = np.searchsorted(
    SPIKE_TIMES, [DETAIL_START, DETAIL_START + DETAIL_SECONDS]
)
detail_times = SPIKE_TIMES[detail_first:detail_last]
detail_units = SPIKE_UNIT[detail_first:detail_last].astype(int)

# The most active units in this window, kept in depth order. Some units fire at 0.01 Hz and
# would contribute an empty row, which is both uninformative and a rendering error waiting to
# happen.
present, counts = np.unique(detail_units, return_counts=True)
busiest = np.sort(present[np.argsort(counts)[::-1][:DETAIL_UNITS]])

# `subcoordinate_y` does not support Spikes in HoloViews 1.23.1, so each unit is placed with
# `position` and given a `spike_length` just short of one row, which is the idiomatic way to
# stack a raster and keeps every spike its own hoverable mark.
#
# Sizing options are passed in rather than applied afterwards. Cloning a responsive plot and
# adding a fixed frame size leaves the original `min_height` in place, and Bokeh then lays the
# canvas out taller than its container and clips the axis off the bottom.
def build_detail_raster(**sizing):
    curves = {}
    for row, order in enumerate(busiest):
        curves[str(UNIT_IDS[order])] = hv.Spikes(
            detail_times[detail_units == order], [time_dim], label=str(UNIT_IDS[order])
        ).opts(
            position=row,
            spike_length=0.85,
            color="black",
            line_width=1,
            hover_tooltips=[("unit", "$label"), ("time")],
        )
    # kdims is declared as a list. HoloViews accepts a bare string and normalises it, but a
    # list matches the declared type and keeps static analysers quiet.
    return hv.NdOverlay(curves, kdims=["unit"]).opts(
        title="",
        ylabel="unit, deep to superficial",
        show_legend=False,
        yticks=[(row + 0.42, str(UNIT_IDS[order])) for row, order in enumerate(busiest)],
        **sizing,
    )


spike_detail_view = build_detail_raster(responsive=True, min_height=360)
spike_detail_view

# %% [markdown]
# Because the units and the channels come from one probe, the two can be read together. The
# layout below shares its time axis, so panning either plot moves both, and the hippocampal
# units at the bottom of the raster sit at the depths where the theta oscillation in the LFP is
# largest.

# %%
# A shared time range links the two plots, but reading one against the other also needs them
# to line up on screen, and that needs an equal frame width rather than a responsive one.
FIELD_FRAME_WIDTH = 900

field_frame = read_level(PYRAMID_LEVELS[0], DETAIL_START, DETAIL_START + DETAIL_SECONDS)
field_plot = downsample1d(
    hv.Overlay(build_curves(field_frame), kdims=[time_dim, "Channel"]).opts(
        **{
            **{k: v for k, v in OVERLAY_OPTS.items() if k not in ("responsive", "min_height")},
            "frame_width": FIELD_FRAME_WIDTH,
            "frame_height": 340,
            "xaxis": None,
            "title": "LFP, 1250 Hz",
        }
    ),
    algorithm="minmax-lttb",
)

spike_field_view = (
    field_plot
    + build_detail_raster(frame_width=FIELD_FRAME_WIDTH, frame_height=280)
).cols(1)
spike_field_view

# %% [markdown]
# ### Waveform shape, and one more rasterization case
#
# A raster answers when each unit fired, and says nothing about the shape of the events, which is
# what distinguishes one unit from another and what sorting algorithms act on. Mean waveforms are
# stored alongside the spike times in the same session file, on every channel the probe recorded,
# so the streaming pattern used above reaches them without a further download.
#
# Plotting them is a rasterization case this notebook does not otherwise cover. Each unit
# contributes one short line per nearby channel, so a few hundred units become several thousand
# curves, and no individual curve matters. Downsampling has nothing to work with, since each
# waveform is only 82 samples long, and the number of lines is the problem rather than their
# length. Datashader also has a categorical aggregator, which counts each category separately and
# so keeps group identity through the reduction even though individual identity is lost. That
# recovers, at the level of the group, what the raster above gave up.
#
# The hvPlot user guide works exactly this case through, on neural waveforms, in [Multiple Lines
# Per Category](https://hvplot.holoviz.org/en/docs/latest/user_guide/Large_Timeseries.html#multiple-lines-per-category-example).
# It also covers the one preparation step that is easy to miss, which is that the curves have to be
# concatenated into a single frame separated by rows of `NaN`, so that the end of one waveform is
# not joined to the start of the next.



# %% [markdown]
# ## Serving the plots as an app
#
# Everything above is a notebook cell, which is where analysis is done but not where a tool
# is handed to someone else. The same objects compose into a Panel template that runs as a
# standalone server with
#
# ```bash
# pixi run serve    # wraps: panel serve large_array_viz.ipynb --show
# ```
#
# The widgets and streams below are new instances rather than the ones used earlier. A Panel
# widget and a Bokeh model belong to one rendered document, so reusing the objects from the
# cells above would put the same model in two documents and leave one of the two dead. The
# functions are reused, since they hold the logic, and only the bindings are rebuilt.
#
# The result is assigned to a variable rather than left as the cell's final expression. A
# template describes a whole page, including a header and a sidebar, and it does not render
# usefully inside a notebook cell. `.servable()` marks it for `panel serve`, which is where
# it is meant to run, and the screenshot at the top of this section is what that looks like.

# %%
app_channel = pn.widgets.Select(name="channel", options=[0, 1], value=0)
app_z = pn.widgets.IntSlider(name="z index", start=0, end=n_z - 1, value=MID_Z)
app_level = pn.widgets.Select(
    name="pyramid level", options=IMAGE_LEVELS, value=IMAGE_LEVELS[0],
)


def app_image_element(level, channel, z):
    plane = image_plane(level, channel, z)
    return image_element(plane).opts(
        **IMAGE_OPTS, clim=IMAGE_CLIM,
        title=f"level {level}, channel {channel}, z {z}",
    )


app_image_plot = rasterize(
    hv.DynamicMap(pn.bind(app_image_element, app_level, app_channel, app_z))
)

app_pyramid_plot = hv.DynamicMap(
    pyramid_overlay, streams=[hv.streams.RangeX(), hv.streams.PlotSize()]
)
app_spike_plot = hv.DynamicMap(
    spike_raster, streams=[hv.streams.RangeX(), hv.streams.PlotSize()]
)

servable_app = pn.template.FastListTemplate(
    title="Large array visualization",
    sidebar=[
        pn.pane.Markdown(
            "A fluorescence volume read over HTTP from a remote OME-Zarr store, and a "
            "local field potential recording served from a multiscale pyramid.\n\n"
            "**Imaging controls**\n\n"
            "These three affect the imaging tab only."
        ),
        app_level,
        app_channel,
        app_z,
        pn.pane.Markdown(
            "**Electrophysiology**\n\n"
            "Neither of those tabs has widgets. The pyramid level and the spike raster's "
            "source are both chosen from the zoom, and each title reports what is in use."
        ),
    ],
    main=[
        # Naming the panes lets Tabs take its titles from them. dynamic keeps the inactive
        # tab unrendered, which matters when the hidden tab holds a large read.
        pn.Tabs(
            pn.pane.HoloViews(app_image_plot, name="Imaging"),
            pn.pane.HoloViews(
                app_pyramid_plot, name="Electrophysiology",
                sizing_mode="stretch_width",
            ),
            pn.pane.HoloViews(
                app_spike_plot, name="Spikes", sizing_mode="stretch_width",
            ),
            dynamic=True,
        )
    ],
    sidebar_width=320,
).servable()

# %% [markdown]
# ## Choosing an approach
#
# <p><img src="assets/concept_escalation.svg" alt="Three tiers of reduction, cheapest first, each with what it reduces" width="640"></p>
#
# Data size is the least useful place to start. A 10 TB image and 10 TB of voltage traces require
# opposite treatments, and a 200 MB set of traces and a 200 GB set require the same treatment
# until memory runs out.
#
# Cost is a better starting point, and it gives the order above. The first tier reduces what is
# sent to the browser, and for most plots it is sufficient by itself, in which case the other two
# tiers never arise. The second tier reduces what is read from storage, and becomes necessary
# once the widest view cannot be reduced quickly enough or the array exceeds memory. The third
# reduces what is stored, and is necessary only when duplicated bytes are themselves the problem.
# Each tier costs more to set up than the one above it, so the useful question is whether the
# cheapest tier is already enough.
#
# Only the first tier requires a choice between alternatives. Rasterizing and downsampling do the
# same job, so one of them is used, and the deciding question is whether individual marks carry
# meaning. If a single trace must remain identifiable, hoverable, and true to its own amplitude,
# aggregating it into a density image destroys the quantity being measured. If the reader is
# instead judging how much is happening where, individual glyphs were never the subject and
# preserving them is wasted effort.
#
# For a point process the choice does not arise, because downsampling requires a line to preserve.
# Spikes are either rasterized or binned, and binning is rasterization with the bins fixed in
# advance.
#
# Two measurable quantities determine where a given plot lands. The resolution the analysis has to
# reach sets the finest level required. The interaction budget, about a tenth of a second, decides
# whether anything must be precomputed, and it should be measured on the **widest** view, since
# that view covers the most data and is also the first one a reader sees.
#
# | Your data | At the browser boundary | At the read boundary | Where the bytes come from |
# | --- | --- | --- | --- |
# | An image, volume, or dense field | Rasterize | Read the level that matches the zoom | Reference it if the archive is already multiscale |
# | Tens of long traces that fit in memory | Downsample | Read the window directly | Nothing extra to store |
# | Tens of long traces that do not fit | Downsample the chosen level | Read a pyramid level | Write the levels, since coarse ones exist nowhere |
# | Millions of discrete events | Rasterize, since downsampling does not apply | Slice by time first, then aggregate | Write binned rates for the coarse end |
#
# Each technique has costs.
#
# Rasterization gives up individual identity. A single trace inside a rasterized bundle
# cannot be hovered, selected, or styled, because by the time it reaches the browser it is
# no longer a line. It also has to be told how to normalize, and `eq_hist` will make faint
# structure visible at the cost of quantitative brightness.
#
# Downsampling uses server CPU on every viewport change. That is
# usually a good trade, but it scales with the number of concurrent users and it needs the
# data in memory to select from.
#
# Pyramids cost disk space and build time, and they become stale when the source changes. The
# one here occupies 2.1 GB and takes about a minute to write, and any change to the source
# recording invalidates it.
# Coarse levels are also averages, so a brief event can be averaged away entirely at the
# level where it is being looked for, which matters when searching for sharp transients rather
# than slow structure. That last cost is the reason the finest level here is
# referenced rather than copied, because the level you most want unaveraged is the one a copy
# is most wasteful for.
#
# The third tier is worth separating from the first two, because it is easy to file under the
# same heading and it does something different. Rasterizing, downsampling, and pyramids all
# reduce what reaches the display. A reference reduces nothing that is drawn. It removes a
# duplicated copy of bytes that already exist, which makes a pyramid cheaper to own without
# making anything faster to draw. It applies when the data is already in a chunked format and
# copying it is the cost you object to.
#
# The tiers compose. This notebook rasterizes the minimap that navigates a downsampled plot,
# and reads a pyramid level, referenced or stored, that is then downsampled again before it is
# drawn.
