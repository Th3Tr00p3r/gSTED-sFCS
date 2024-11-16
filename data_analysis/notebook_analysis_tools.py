import functools
import os
from contextlib import contextmanager, suppress
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from data_analysis.correlation_function import SolutionSFCSExperiment
from utilities.display import Plotter, default_colors, plot_acfs

try:
    from winsound import Beep as beep  # type: ignore # Windows
except ModuleNotFoundError:

    def beep():
        """Beep to notify that the script has finished"""
        os.system('osascript -e "beep 3"')


@contextmanager
def mpl_backend(backend, verbose=False):
    """Temporarily switch Matplotlib backend. For use in Jupyter notebooks."""

    original_backend = mpl.get_backend()
    if original_backend == backend:
        yield
        return
    if verbose:
        print(
            f"[MPL BACKEND] Switching Matplotlib backend from '{original_backend}' to '{backend}'..."
        )
    plt.close("all")
    mpl.use(backend)
    try:
        yield
    finally:
        plt.pause(0.01)  # Give the event loop time to process
        if backend == "Qt5Agg":
            if verbose:
                print("[MPL BACKEND] Blocking until the window is closed...")
            plt.show(block=True)  # Block until the window is closed
        if verbose:
            print(f"[MPL BACKEND] Switching Matplotlib backend back to '{original_backend}'...")
        plt.close("all")
        mpl.use(original_backend)


class SolutionSFCSExperimentLoader:
    """
    Class to load and process Solution SFCS experiments.
    """

    def __init__(self, data_root: Path, force_processing: bool = False):
        self._data_root = data_root
        self._force_processing = force_processing

    def _build_template_paths(self, data_config: Dict[str, Dict[str, Any]]):
        # build proper paths
        for config in data_config.values():
            config["confocal_path_template"] = (
                self._data_root / config["confocal_date"] / "solution" / config["confocal_template"]
                if config.get("confocal_template")
                else None
            )
            config["sted_path_template"] = (
                self._data_root / config["sted_date"] / "solution" / config["sted_template"]
                if config.get("sted_template")
                else None
            )

    @staticmethod
    def _print_log_files(data_config: Dict[str, Dict[str, Any]]):
        for label, config in data_config.items():
            if config["confocal_path_template"]:
                log_file_path = str(config["confocal_path_template"]).replace("_*.pkl", ".log")
            else:
                # STED only
                log_file_path = str(config["sted_path_template"]).replace("_*.pkl", ".log")
            print(f"{label}:")
            try:
                with open(log_file_path) as f:
                    print(f.read() + "\n\n")
            except FileNotFoundError:
                print("Log file not created yet!")

    def load_experiments(
        self, data_config: Dict[str, Dict[str, Any]], print_log_files: bool = False
    ) -> Dict[str, SolutionSFCSExperiment]:
        # build proper paths (in-place)
        self._build_template_paths(data_config)

        # Print log files (Optional)
        if print_log_files:
            self._print_log_files(data_config)

        # Initialize then load experiments
        exp_dict = {label: SolutionSFCSExperiment(name=label) for label in data_config.keys()}
        for label, exp in exp_dict.items():
            # Determine if processing is forced
            force_processing = self._force_processing or data_config[label].pop(
                "force_processing", False
            )

            # skip already loaded experiments, unless forced
            if not hasattr(exp_dict[label], "confocal") or force_processing:
                with mpl_backend("inline"):
                    exp.load_experiment(
                        **data_config[label],
                        force_processing=force_processing,
                        is_verbose=True,
                    )

                # update processed flag
                if (not exp.confocal.was_processed_data_loaded) and (
                    not exp.sted.was_processed_data_loaded
                ):
                    data_config[label]["was_processed"] = True

                # calibrate TDC - only if STED is loaded and not already synced (which is the case when filtering afterpulsing)
                if exp.sted.is_loaded and not exp.sted.afterpulsing_method == "filter":
                    with mpl_backend("inline"):
                        exp.calibrate_tdc(
                            **data_config[label], force_processing=force_processing, is_verbose=True
                        )

                # save processed data (to avoid re-processing)
                exp.save_processed_measurements(
                    data_root=self._data_root,
                    should_force=data_config[label]["force_save"],
                    # NOTE: processed data is temporarily stored anyway, and kept if not cleared
                    # setting "should_save_data" False saves a lot of time.
                    # After analyzing, each experiment could be saved again (with its
                    # processed data) for faster loading the next time if more analysis
                    # needs to be done
                    should_save_data=False,
                )

        # beep to notify that the script has finished
        beep()

        return exp_dict


class SolutionSFCSExperimentHandler:
    """
    Class to handle Solution SFCS experiments.
    """

    def __init__(self, exp_dict: Optional[Dict[str, SolutionSFCSExperiment]] = None, **kwargs):
        self.exp_dict = exp_dict or {}
        self._positive_filters: Tuple[str, ...] = ("",)
        self._negative_filters: Tuple[str, ...] = ()
        self._data_loader = SolutionSFCSExperimentLoader(**kwargs)
        self._data_config: Dict[str, Dict[str, Any]] = {}

    @property
    def labels(self):
        return list(self.exp_dict.keys())

    @property
    def experiments(self):
        return list(self.exp_dict.values())

    @property
    def filtered_exp_dict(self) -> Dict[str, SolutionSFCSExperiment]:
        fed = {
            label: exp
            for label, exp in self.exp_dict.items()
            if any([str_ in label for str_ in self._positive_filters])
            and not any([str_ in label for str_ in self._negative_filters])
        }
        if not fed:
            print("WARNING: No experiments match the current filter criteria.")
        return fed

    @property
    def positive_filters(self) -> Tuple[str, ...]:
        return self._positive_filters

    @positive_filters.setter
    def positive_filters(self, filters: Optional[List[str]] = None):
        self._positive_filters = tuple(filters) if filters else ("",)
        _ = self.filtered_exp_dict  # trigger warning if no experiments match the current filters

    @property
    def negative_filters(self) -> Tuple[str, ...]:
        return self._negative_filters

    @negative_filters.setter
    def negative_filters(self, filters: Optional[List[str]] = None):
        self._negative_filters = tuple(filters) if filters else ()
        _ = self.filtered_exp_dict  # trigger warning if no experiments match the current filters

    @staticmethod
    def skip_if_all_exp_filtered(func):
        """
        Decorator to run a function only if there are filtered experiments.
        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.filtered_exp_dict:
                return func(self, *args, **kwargs)
            else:
                print("Skipping - no experiments match the current filter criteria.")

        return wrapper

    def load_experiments(self, data_config: Dict[str, Dict[str, Any]], **kwargs):
        """
        Load a set of experiments, and keep the data configuration.
        """
        self._data_config = data_config
        self.exp_dict = self._data_loader.load_experiments(data_config, **kwargs)

    @skip_if_all_exp_filtered
    def plot_afterpulsing_filters(self, backend="inline", **kwargs):
        """
        Plot the afterpulsing filters for a set of experiments.
        """
        with mpl_backend(backend):
            for label, exp in self.filtered_exp_dict.items():
                print(f"{label}:")
                exp.plot_afterpulsing_filters(**kwargs)

    @skip_if_all_exp_filtered
    def plot_correlation_functions(
        self, confocal_only: bool = False, sted_only: bool = False, **kwargs
    ):
        """
        Plot the correlation functions for a set of experiments.
        """
        with Plotter(super_title="All Experiments, All ACFs", figsize=(8, 6)) as ax:
            for label, exp in self.filtered_exp_dict.items():
                with suppress(AttributeError):
                    # # normalized vt_um
                    # exp.plot_correlation_functions(
                    #     parent_ax=ax, sted_only=sted_only
                    # )

                    # normalized linear vt_um
                    exp.plot_correlation_functions(
                        x_scale="log",
                        y_scale="linear",
                        parent_ax=ax,
                        sted_only=sted_only,
                        confocal_only=confocal_only,
                        **kwargs,
                    )

                    # # normalized linear vt_um
                    # exp.confocal.plot_correlation_functions(
                    #     x_scale="log", y_scale="linear", parent_ax=ax, sted_only=sted_only
                    # )
                    #
                    # # normalized vs. log lag
                    # exp.plot_correlation_functions(
                    #     x_field="lag", parent_ax=ax, confocal_only=confocal_only
                    # )
                    #
                    # # avg_cf_cr vs. log lag
                    # exp.plot_correlation_functions(
                    #     x_field="lag", y_field="avg_cf_cr", parent_ax=ax,
                    #     confocal_only=confocal_only
                    # )
                    #
                    # # log normalized vs. vt_um_sq
                    # exp.plot_correlation_functionsrelation_functions(
                    #     x_scale="quadratic", parent_ax=ax, sted_only=sted_only
                    # )

                print(f"\n{label} countrates:\nConfocal - {exp.confocal.avg_cnt_rate_khz:.2f} kHz")
                # print(f"\n{label} line frequency:\nConfocal - {exp.confocal.scan_settings['line_freq_hz']:.2f} Hz")

    @skip_if_all_exp_filtered
    def display_scan_images(self, n_images: int = 3, backend="inline"):
        """
        Display scan images for a set of experiments.
        """
        for label, exp in self.filtered_exp_dict.items():
            print("Confocal scan images:")
            with mpl_backend(backend):
                exp.confocal.display_scan_images(n_images)
            print("STED scan images:")
            try:
                with mpl_backend(backend):
                    exp.sted.display_scan_images(n_images)
            except AttributeError:
                print("STED measurement not loaded!")
            print()

    @skip_if_all_exp_filtered
    def display_lifetime(self, force: bool = False):
        """
        Display lifetime histograms for a set of experiments.
        """
        for label, exp in self.filtered_exp_dict.items():
            # Calculate lifetime parameters if not already done (or forced)
            if exp.lifetime_params is None or force:
                with mpl_backend("Qt5Agg"):
                    exp.get_lifetime_parameters()

                # save (lifetime_params) if calculated
                exp.save_processed_measurements(
                    should_save_data=False,
                    # NOTE: processed files should already exist at this point, so need to force
                    should_force=True,
                )

            # print lifetime parameters
            print(f"Lifetime: {exp.lifetime_params.lifetime_ns:.2f} ns")

    @skip_if_all_exp_filtered
    def plot_countrates(self, backend="inline"):
        """
        Plot the count rates for a set of experiments.
        """
        for label, exp in self.filtered_exp_dict.items():
            with (
                mpl_backend(backend),
                Plotter(
                    super_title=f"'{label}' - Countrate", xlabel="time (s)", ylabel="countrate"
                ) as ax,
            ):
                with suppress(IndexError):
                    cf_conf = list(exp.confocal.cf.values())[0]
                    conf_split_countrate = cf_conf.countrate_list
                    conf_split_time = np.cumsum(cf_conf.split_durations_s)
                    ax.plot(conf_split_time, conf_split_countrate, label="confocal")

                with suppress(IndexError):
                    cf_sted = list(exp.sted.cf.values())[0]
                    sted_split_countrate = cf_sted.countrate_list
                    sted_split_time = np.cumsum(cf_sted.split_durations_s)
                    ax.plot(sted_split_time, sted_split_countrate, label="STED")

                ax.legend()

    @skip_if_all_exp_filtered
    def plot_split_acfs(self, backend="inline"):
        """
        Plot the split ACFs for a set of experiments.
        """
        for label, exp in self.filtered_exp_dict.items():
            # Get the confocal and STED ACFs
            cf_conf = list(exp.confocal.cf.values())[0]
            cf_sted = list(exp.sted.cf.values())[0]

            # Plot the split ACFs along with the mean ACF
            with mpl_backend(backend), Plotter(subplots=(2, 2)) as axes:
                conf_axes, sted_axes = axes[0], axes[1]

                # plot confocal
                with suppress(NameError):
                    plot_acfs(
                        cf_conf.lag,
                        cf_conf.avg_cf_cr,
                        cf_conf.g0,
                        cf_conf.cf_cr,
                        # plot kwargs
                        parent_ax=conf_axes[0],
                        ylabel="Confocal",
                    )
                    conf_axes[0].set_title(f"'{label}' - Split ACFs (All)")
                    plot_acfs(
                        cf_conf.lag,
                        cf_conf.avg_cf_cr,
                        cf_conf.g0,
                        cf_conf.cf_cr,
                        j_good=cf_conf.j_good,
                        # plot kwargs
                        parent_ax=conf_axes[1],
                        super_title=f"'{label}' - Split ACFs (confocal)",
                    )
                    conf_axes[1].set_title(f"'{label}' - Split ACFs (Good)")

                # plot STED
                with suppress(NameError):
                    plot_acfs(
                        cf_sted.lag,
                        cf_sted.avg_cf_cr,
                        cf_sted.g0,
                        cf_sted.cf_cr,
                        # plot kwargs
                        parent_ax=sted_axes[0],
                        ylabel="STED",
                    )
                    plot_acfs(
                        cf_sted.lag,
                        cf_sted.avg_cf_cr,
                        cf_sted.g0,
                        cf_sted.cf_cr,
                        j_good=cf_sted.j_good,
                        # plot kwargs
                        parent_ax=sted_axes[1],
                        super_title=f"'{label}' - Split ACFs (STED)",
                    )

    @skip_if_all_exp_filtered
    def print_fluorescence_properties(self):
        """
        Print the fluorescence properties for a set of experiments.
        """
        for label, exp in self.filtered_exp_dict.items():
            conf_cr = exp.confocal.avg_cnt_rate_khz
            sted_cr = exp.sted.avg_cnt_rate_khz

            print(f"'{label}':")
            print("".join(["-"] * (len(label) + 3)))
            print(
                f"    Confocal Fluorescence-per-Molecule: "
                f"{list(exp.confocal.cf.values())[0].g0 / 1e3:.2f} kHz"
            )
            print(
                f"    Confocal countrate: {conf_cr:.2f} +/- "
                f"{exp.confocal.std_cnt_rate_khz:.2f} kHz"
            )
            print(
                f"    Confocal Molecules in Sampling Volume: "
                f"{conf_cr / (list(exp.confocal.cf.values())[0].g0 / 1e3):.2f}"
            )
            try:
                print(
                    f"\n    STED Fluorescence-per-Molecule: "
                    f"{list(exp.sted.cf.values())[0].g0 / 1e3:.2f} kHz"
                )
                print(
                    f"    STED countrate: {sted_cr:.2f} +/- " f"{exp.sted.std_cnt_rate_khz:.2f} kHz"
                )
                print(
                    f"    STED Molecules in Sampling Volume: "
                    f"{sted_cr / (list(exp.sted.cf.values())[0].g0 / 1e3):.2f}"
                )
            except IndexError:
                print("STED measurement not loaded!")
            print("\n")

    @skip_if_all_exp_filtered
    def re_average_acfs(self, force: bool = False):
        """
        Re-average the ACFs for a set of experiments.
        """
        for label, exp in self.filtered_exp_dict.items():
            if self._data_config[label]["was_processed"] or force:
                print(f"Re-averaging '{label}'...")
                exp.re_average_all(
                    norm_range=(2.7e-3, 3.7e-3),
                    should_use_clustering=True,
                    rejection=2,
                    noise_thresh=0.5,
                    should_plot=True,
                )

                cf_conf = list(exp.confocal.cf.values())[0]
                #         cf_sted = list(exp.sted.cf.values())[0]
                #         for cf in (cf_conf, cf_sted):
                for cf in (cf_conf,):
                    plot_acfs(
                        cf.lag,
                        cf.avg_cf_cr,
                        cf.g0,
                        cf.cf_cr,
                        j_good=cf.j_good,
                        # plot kwargs
                        super_title=f"'{label}' - Split ACFs ({cf.name})",
                    )
            else:
                print(f"{label}: Using existing...")

    @skip_if_all_exp_filtered
    def calculate_hankel_transforms(self, force: bool = False):
        """
        Calculate the Hankel transforms for a set of experiments.
        """
        # TODO: this one needs to be improved in terms of user-friendliness and interactivity -
        #  1. keep the already selected limits on screen
        #  2. keep the previous data (in a different color), as well as it's fit
        #  3. Figure out why the last plot gets stuck after selecting the limit (middle mouse
        #     button or 'Enter')
        for label, exp in self.filtered_exp_dict.items():
            if self._data_config[label]["was_processed"] or force:
                with mpl_backend("Qt5Agg"), Plotter() as ax:
                    # remaining_colors = cycle(default_colors)
                    for cf_name, cf in exp.cf_dict.items():
                        cf.calculate_hankel_transform(
                            ["gaussian"],
                            rmax=200,
                            title_prefix=f"{label}: ",
                            parent_ax=ax,
                        )
                        # set all data points to

                # save processed meas (data not re-saved - should be quick)
                exp.save_processed_measurements(
                    should_save_data=False,
                    # processed files should already exist at this point, so need to force
                    should_force=True,
                )

            else:
                print(f"{label}: Using existing...")
