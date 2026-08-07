"""Verify the datasets this example depends on.

Run this before the notebook. Remote and archival data drifts, so every property the
notebook relies on is asserted rather than assumed. The constants here mirror the ones
in large_array_viz.py so this script stays runnable on its own.

    python check_data.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pooch
import zarr
from xarray_ome_ngff import DaskArrayWrapper, read_multiscale_group

OME_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"

ALLEN_ECEPHYS = (
    "https://allen-brain-observatory.s3.amazonaws.com/"
    "visual-coding-neuropixels/ecephys-cache"
)
SESSION_ID = 754312389
PROBE_ID = 756781563
LFP_URL = f"{ALLEN_ECEPHYS}/session_{SESSION_ID}/probe_{PROBE_ID}_lfp.nwb"
DATA_DIR = Path(__file__).parent / "data"
LFP_PATH = DATA_DIR / Path(LFP_URL).name

LFP_SERIES = f"acquisition/probe_{PROBE_ID}_lfp_data"
LFP_DATA_KEY = f"{LFP_SERIES}/data"
LFP_TIME_KEY = f"{LFP_SERIES}/timestamps"
LFP_CHANNEL_KEY = f"{LFP_SERIES}/electrodes"

EXPECTED_LFP_SHAPE = (12094337, 35)
EXPECTED_SAMPLING_RATE = 1250.0


def check_ome_zarr() -> None:
    print(f"OME-Zarr: {OME_URL}")
    group = zarr.open_group(OME_URL, mode="r")
    arrays = read_multiscale_group(group, array_wrapper=DaskArrayWrapper(chunks=32))

    keys = sorted(arrays)
    assert len(keys) >= 2, f"expected at least two levels, got {keys}"
    assert {"0", "1"} <= set(keys), f"expected levels '0' and '1', got {keys}"

    a0, a1 = arrays["0"], arrays["1"]
    assert a0.dims == ("c", "z", "y", "x"), a0.dims
    assert a0.shape == (2, 236, 275, 271), a0.shape
    assert str(a0.dtype) == "uint16", a0.dtype
    assert a1.shape == (2, 236, 137, 135), a1.shape

    # y and x halve while z is untouched, which is what makes the volume anisotropic.
    assert a1.sizes["z"] == a0.sizes["z"], (a1.sizes["z"], a0.sizes["z"])

    dz = float(a0.z[1] - a0.z[0])
    dy = float(a0.y[1] - a0.y[0])
    dx = float(a0.x[1] - a0.x[0])
    assert abs(dz - 0.5002) < 0.01, dz
    assert abs(dy - 0.3604) < 0.01, dy
    assert abs(dx - 0.3604) < 0.01, dx

    # The HTTP store cannot list directories, so membership is tested by key.
    assert "labels" in group, "no labels group"

    # A blank slice is the usual way a remote read looks successful and plots nothing.
    sl = a1.isel(c=0, z=a1.sizes["z"] // 2).compute()
    assert float(sl.std()) > 0, "level 1 mid-z slice is constant"

    print(f"  levels {keys}")
    print(f"  level 0 {a0.shape} {a0.dtype}, level 1 {a1.shape}")
    print(f"  spacing z={dz:.4f} y={dy:.4f} x={dx:.4f} micrometers")
    print(f"  mid-z slice range {float(sl.min()):.0f} to {float(sl.max()):.0f}")
    print("  ome-zarr ok")


def download_lfp() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LFP_PATH.exists():
        print(f"LFP: reusing {LFP_PATH}")
        return LFP_PATH
    print(f"LFP: downloading ~900 MB to {LFP_PATH}")
    pooch.retrieve(
        url=LFP_URL,
        known_hash=None,
        fname=LFP_PATH.name,
        path=DATA_DIR,
        progressbar=True,
    )
    return LFP_PATH


def check_lfp() -> None:
    path = download_lfp()
    with h5py.File(path, "r") as f:
        for key in (LFP_DATA_KEY, LFP_TIME_KEY, LFP_CHANNEL_KEY):
            assert key in f, f"missing HDF5 path {key}"

        data = f[LFP_DATA_KEY]
        time = f[LFP_TIME_KEY]
        channels = f[LFP_CHANNEL_KEY][:]

        assert data.shape == EXPECTED_LFP_SHAPE, data.shape
        assert str(data.dtype) == "float32", data.dtype
        assert time.shape == (data.shape[0],), (time.shape, data.shape)
        assert channels.shape == (data.shape[1],), (channels.shape, data.shape)

        t_min, t_max = float(time[0]), float(time[-1])
        assert abs(t_min - 3.677) < 0.1, t_min
        assert abs(t_max - 9679.13) < 1.0, t_max

        # One check that catches a transposed array, a truncated file, and a wrong path.
        rate = data.shape[0] / (t_max - t_min)
        assert abs(rate - EXPECTED_SAMPLING_RATE) < 5, rate

        window = data[: int(EXPECTED_SAMPLING_RATE) * 10, :]
        peak = float(np.abs(window).max())
        assert peak > 0, "first ten seconds are all zeros"
        # The file records its own unit, so assert it rather than inferring from magnitude.
        assert f[LFP_DATA_KEY].attrs["unit"] == "volts"
        assert f[LFP_DATA_KEY].attrs["conversion"] == 1.0
        assert peak < 1e-2, f"amplitude {peak} is larger than expected for volts"

    print(f"  shape {data.shape} {data.dtype}")
    print(f"  time {t_min:.3f} to {t_max:.3f} s, derived rate {rate:.2f} Hz")
    print(f"  channels {channels[:3].tolist()} ... {channels[-1]}")
    print(f"  peak |amplitude| over first 10 s: {peak:.3e} V ({peak * 1e6:.1f} uV)")
    print("  lfp ok")


if __name__ == "__main__":
    check_ome_zarr()
    check_lfp()
    print("all data checks passed")
