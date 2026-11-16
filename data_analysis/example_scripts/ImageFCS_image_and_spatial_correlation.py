"""
ImageFCS measurements (the 'sFCS ImageFCS' tab): the image and the spatial correlation.

An ImageFCS measurement is saved like any scanning-FCS measurement - files
'<template>_angular_<laser>_<time>_<n>.pkl' with '<...>_byte_data.npy' beside each, in the
'solution' folder of the day. Everything below is rebuilt from the photon time-stamps
(data_analysis/image_fcs.py); only NumPy and matplotlib are needed, not the correlator
library.

Set DAY_DIR and TEMPLATE below and run (Spyder, a notebook cell, or 'python <this file>').
Afterwards:

    files[i]       one dict per file: 'runtime' (photon times in laser pulses), 'geom',
                   'image' (a ScanImage: .counts, .rate_khz, .pixel_um, .offset_samples),
                   'corr' (the spatial correlation of that file) and 'scan_settings'
    image_khz      the image summed over the files, as a count rate
    corr           the spatial correlation averaged over the files:
                   corr['lag_um'], corr['g'], corr['g_err'], corr['g_lines'] (per line)

What the spatial correlation is
-------------------------------
Every sweep of the beam along a line is one independent measurement. Its photons are binned
along the line and G(dx) = <n(x) n(x+dx)> / <n>^2 - 1 is computed, without self-pairs (so no
shot-noise spike at dx = 0). The sweeps are averaged with weights equal to their photon
numbers, and sweeps with fewer than MIN_LINE_FRACTION of the photons of the brightest ones
are dropped (they pass outside the object). A sweep takes tens of milliseconds, so G(dx)
is the correlation of the structure as it was during that sweep; repeated sweeps of a
line only add statistics.

Two things to keep in mind when reading G(dx):
 * it contains everything that makes the brightness vary along a line - the structure of
   the object, but also its overall envelope (a 3 um wide object in a 10 um line gives a
   3 um wide pedestal under the fine structure);
 * the first one or two points (dx below ~ speed x 10 us) include detector afterpulsing.

For an X-Z section ('slow axis' = Z) the same code applies; the rows of the image are then
z-steps, and the correlation is still along the fast (in-plane) direction.
"""

import os
from pathlib import Path

import numpy as np

###############################################
# Move to project root to easily import modules
###############################################
PROJECT_ROOT = Path("/Users/oleg/Documents/GitHub/gSTED-sFCS")
if PROJECT_ROOT.exists():
    os.chdir(PROJECT_ROOT)

from matplotlib import pyplot as plt  # noqa: E402

from data_analysis import image_fcs  # noqa: E402
from utilities.file_utilities import load_file_dict  # noqa: E402

######################
# what to process
######################
DAY_DIR = Path("/Users/oleg/Documents/Experiments/tmp/17_11_2026/solution")
TEMPLATE = "imgfcs_angular_exc_*"  # all files of one measurement ('_1', '_2', ...)

# the scan offset (see the docstring of data_analysis/image_fcs.py): None finds it from
# each file; a number (AO samples) uses that for all files
OFFSET_SAMPLES = None

PIXEL_UM = None  # image pixel along the line; None = the line spacing (square pixels)
CORR_BIN_UM = None  # resolution of the correlation; None = half an AO sample of travel
CORR_MAX_LAG_UM = 3.0
MIN_LINE_FRACTION = 0.3  # drop sweeps with fewer photons than this fraction of the bright


def file_number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_files() -> list:
    """Read every file of the measurement and build its image and correlation."""

    paths = sorted(
        (p for p in DAY_DIR.glob(TEMPLATE + ".pkl") if p.stem.split("_")[-1].isdigit()),
        key=file_number,
    )
    if not paths:
        raise FileNotFoundError(f"No files like '{TEMPLATE}.pkl' in '{DAY_DIR}'")

    files = []
    for path in paths:
        full_data = load_file_dict(path)["full_data"]
        scan_settings = full_data["scan_settings"]
        laser_freq_hz = full_data["laser_freq_mhz"] * 1e6
        byte_data = np.load(path.with_name(path.stem + "_byte_data.npy"), mmap_mode="r")

        geom = image_fcs.ScanGeometry.from_scan_settings(scan_settings, laser_freq_hz)
        runtime = image_fcs.photon_runtime(byte_data)
        image = image_fcs.scan_image(
            runtime, geom, offset_samples=OFFSET_SAMPLES, pixel_um=PIXEL_UM
        )
        corr = image_fcs.line_correlations(
            runtime,
            geom,
            image.offset_samples,
            bin_um=CORR_BIN_UM,
            max_lag_um=CORR_MAX_LAG_UM,
            min_line_fraction=MIN_LINE_FRACTION,
        )
        how = "" if OFFSET_SAMPLES is not None else f" (quality {image.offset_quality:.0f})"
        print(
            f"{path.name}: {runtime.size:,} photons, {image.duration_s:.1f} s, "
            f"{runtime.size / max(image.duration_s, 1e-9) * 1e-3:.1f} kHz; "
            f"scan offset {image.offset_samples:+.2f} samples{how}; "
            f"{corr['n_sweeps_used']}/{corr['n_sweeps']} sweeps in the correlation"
        )
        if OFFSET_SAMPLES is None and not image.offset_quality >= 5:
            print("   WARNING: the offset could not be found from this file - set it by hand")
        files.append(
            dict(
                path=path,
                scan_settings=scan_settings,
                geom=geom,
                runtime=runtime,
                image=image,
                corr=corr,
            )
        )
    return files


def combine(files: list):
    """Sum the images and average the correlations (weighted by sweeps used)."""

    counts = sum(f["image"].counts for f in files)
    dwell_s = sum(f["image"].dwell_s for f in files)
    image_khz = counts / dwell_s * 1e-3

    weights = np.array([f["corr"]["n_sweeps_used"] for f in files], dtype=float)
    g = np.array([f["corr"]["g"] for f in files])
    g_err = np.array([f["corr"]["g_err"] for f in files])
    # per-line correlations: average over the files in which the line was used
    stack = np.array([f["corr"]["g_lines"] for f in files])
    n_used = np.isfinite(stack).sum(axis=0)
    g_lines = np.where(
        n_used > 0, np.nan_to_num(stack).sum(axis=0) / np.maximum(n_used, 1), np.nan
    )
    corr = dict(
        lag_um=files[0]["corr"]["lag_um"],
        g=(g * weights[:, None]).sum(axis=0) / weights.sum(),
        g_err=np.sqrt(((g_err * weights[:, None]) ** 2).sum(axis=0)) / weights.sum(),
        g_lines=g_lines,
        line_photons=np.mean([f["corr"]["line_photons"] for f in files], axis=0),
        line_used=np.mean([f["corr"]["line_used"] for f in files], axis=0),
    )
    return image_khz, corr


def plot(files: list, image_khz: np.ndarray, corr: dict):
    """The image, the correlation, and the correlation line by line."""

    geom, image = files[0]["geom"], files[0]["image"]
    line_offsets = geom.line_offsets_um()
    extent = (
        -geom.linear_len_um / 2,
        geom.linear_len_um / 2,
        line_offsets[-1] + geom.line_shift_um / 2,
        line_offsets[0] - geom.line_shift_um / 2,
    )
    slow_label = "z (um)" if geom.slow_axis == "z" else "across the lines (um)"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    shown = ax.imshow(
        image_khz,
        cmap="bone",
        extent=extent,
        aspect="equal",
        vmin=0,
        vmax=np.percentile(image_khz, 99.9),
    )
    fig.colorbar(shown, ax=ax, label="kHz", shrink=0.8)
    ax.set_xlabel(f"along the lines (um), at {geom.angle_deg:.0f} deg")
    ax.set_ylabel(slow_label)
    ax.set_title(
        f"{image_khz.shape[0]} x {image_khz.shape[1]} pixels of {image.pixel_um * 1e3:.0f} nm"
    )

    ax = axes[1]
    lag, g, g_err = corr["lag_um"], corr["g"], corr["g_err"]
    ax.errorbar(lag[1:], g[1:], g_err[1:], fmt=".", ms=4, elinewidth=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("distance along the line (um)")
    ax.set_ylabel("G(dx)")
    ax.set_title("Spatial correlation, all used sweeps")

    ax = axes[2]
    shown = ax.imshow(
        corr["g_lines"][:, 1:],
        aspect="auto",
        cmap="viridis",
        extent=(lag[1], lag[-1], extent[2], extent[3]),
    )
    fig.colorbar(shown, ax=ax, label="G(dx)", shrink=0.8)
    ax.set_xlabel("distance along the line (um)")
    ax.set_ylabel(slow_label)
    ax.set_title("Spatial correlation, line by line (blank: line not used)")

    fig.suptitle(f"{files[0]['path'].parent.parent.name} - {TEMPLATE}")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    files = load_files()
    image_khz, corr = combine(files)
    figure = plot(files, image_khz, corr)
    plt.show()
