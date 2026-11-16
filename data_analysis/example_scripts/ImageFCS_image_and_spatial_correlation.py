"""
ImageFCS measurements (the 'sFCS ImageFCS' tab): the image and the spatial correlation,
in the project's standard structures.

An ImageFCS measurement is saved like any angular scanning-FCS measurement - files
'<template>_angular_<laser>_<time>_<n>.pkl' with '<...>_byte_data.npy' beside each, in the
'solution' folder of the day - and it is processed by the same classes:

    meas = SolutionSFCSMeasurement(...)      the measurement
    meas.data[i].raw                         photons of file i: .pulse_runtime, .coarse,
                                             .fine, .delay_time (after calibrate_tdc),
                                             .line_num
    meas.scan_images_dstack, meas.rois       the scan image of each file (one row per
                                             sweep) and the region used
    meas.tdc_calib                           lifetime histogram (.t_hist, .all_hist_norm)
    cf = meas.cf[CF_NAME]                    the CorrFunc object:
        cf.lag (ms), cf.vt_um                lag, and lag x scan speed = distance
        cf.corrfunc, cf.cf_cr, cf.weights    one row per scan line of each file
        cf.avg_corrfunc, cf.error_corrfunc   weighted average over the lines kept
        cf.avg_cf_cr, cf.error_cf_cr         the same, times the count rate
        cf.normalized, cf.g0                 avg_cf_cr / g0
        cf.j_good, cf.j_bad                  rows kept / rejected by the outlier score
        cf.countrate                         count rate inside the region

Two settings matter for images and differ from a solution measurement:

ROI_SELECTION = "bright lines"
    Whole lines (their linear part) are correlated, and only lines holding at least
    MIN_LINE_FRACTION of the photons of the bright ones are used - the others pass beside
    the object. ("auto", the solution default, thresholds the image for a uniform sample
    and fails on a structured one; "all" uses every line.)

SUBTRACT_MEAN_IMAGE_CORR
    The solution analysis subtracts the correlation of the time-averaged image from every
    line ('subtract_spatial_bg_corr'), which removes whatever does not move and leaves the
    fluctuating part. True: that (for a static object nothing is left). False: the full
    spatial correlation of the lines, structure and envelope of the object included.

The scan shift is found from the sweeps themselves (data_analysis/image_fcs.py); this is
automatic for files written by the ImageFCS-era program.

A properly proportioned image (square pixels, all files summed) comes from image_fcs.
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
from data_analysis.correlation_function import SolutionSFCSMeasurement  # noqa: E402
from utilities.file_utilities import load_file_dict  # noqa: E402

######################
# what to process
######################
DAY_DIR = Path("/Users/oleg/Documents/Experiments/tmp/17_11_2026/solution")
TEMPLATE = "imgfcs_angular_exc_101010_*.pkl"  # one measurement: all its file numbers
MEAS_TYPE = "confocal"  # or "sted"
CF_NAME = "confocal"

ROI_SELECTION = "bright lines"
MIN_LINE_FRACTION = 0.3
SUBTRACT_MEAN_IMAGE_CORR = False
SHOULD_CALIBRATE_TDC = True  # lifetime histogram and per-photon delay times
MAX_DISTANCE_UM = 5.0  # range of the correlation plots


def process() -> SolutionSFCSMeasurement:
    """Load the measurement and correlate it - the standard objects."""

    meas = SolutionSFCSMeasurement(MEAS_TYPE)
    meas.read_fpga_data(
        DAY_DIR / TEMPLATE,
        roi_selection=ROI_SELECTION,
        min_line_fraction=MIN_LINE_FRACTION,
        is_verbose=True,
    )
    if SHOULD_CALIBRATE_TDC:
        try:
            meas.calibrate_tdc()
        except Exception as exc:
            print(f"TDC calibration failed [{exc}] - continuing without delay times.")
    meas.correlate_and_average(cf_name=CF_NAME, subtract_spatial_bg_corr=SUBTRACT_MEAN_IMAGE_CORR)
    return meas


def summed_image():
    """All files of the measurement as one image with square pixels (image_fcs)."""

    images = []
    for path in sorted(DAY_DIR.glob(TEMPLATE)):
        full_data = load_file_dict(path)["full_data"]
        byte_data = np.load(path.with_name(path.stem + "_byte_data.npy"), mmap_mode="r")
        images.append(
            image_fcs.scan_image_from_bytes(
                byte_data, full_data["scan_settings"], full_data["laser_freq_mhz"] * 1e6
            )
        )
        print(
            f"{path.name}: scan offset {images[-1].offset_samples:+.2f} samples "
            f"(quality {images[-1].offset_quality:.0f})"
        )
    counts = sum(image.counts for image in images)
    dwell_s = sum(image.dwell_s for image in images)
    return counts / dwell_s * 1e-3, images[0]


def plot(meas: SolutionSFCSMeasurement, image_khz: np.ndarray, image):
    """The image, the averaged correlation, and the correlation of every line."""

    cf = meas.cf[CF_NAME]
    geom = image.geom
    line_offsets = geom.line_offsets_um()
    extent = (
        -geom.linear_len_um / 2,
        geom.linear_len_um / 2,
        line_offsets[-1] + geom.line_shift_um / 2,
        line_offsets[0] - geom.line_shift_um / 2,
    )

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
    ax.set_ylabel("z (um)" if geom.slow_axis == "z" else "across the lines (um)")
    ax.set_title(
        f"{image_khz.shape[0]} x {image_khz.shape[1]} px of {image.pixel_um * 1e3:.0f} nm"
    )

    in_range = (cf.vt_um > 0) & (cf.vt_um <= MAX_DISTANCE_UM)
    ax = axes[1]
    ax.errorbar(
        cf.vt_um[in_range],
        cf.avg_corrfunc[in_range],
        cf.error_corrfunc[in_range],
        fmt=".",
        ms=4,
        elinewidth=0.6,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("distance = lag x scan speed (um)")
    ax.set_ylabel("G  (cf.avg_corrfunc)")
    ax.set_title(f"Average of {len(cf.j_good)} lines ({len(cf.j_bad)} rejected)")

    ax = axes[2]
    rows = cf.corrfunc[:, in_range]
    shown = ax.pcolormesh(
        cf.vt_um[in_range],
        np.arange(rows.shape[0]),
        rows,
        shading="nearest",
        cmap="viridis",
        vmin=np.percentile(rows, 1),
        vmax=np.percentile(rows, 99),
    )
    fig.colorbar(shown, ax=ax, label="G", shrink=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("distance (um)")
    ax.set_ylabel("row of cf.corrfunc (lines in the region, file after file)")
    ax.set_title("Every line")

    fig.suptitle(f"{DAY_DIR.parent.name} - {TEMPLATE}")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    meas = process()
    cf = meas.cf[CF_NAME]
    image_khz, first_image = summed_image()
    figure = plot(meas, image_khz, first_image)
    plt.show()
