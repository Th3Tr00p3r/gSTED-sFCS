import functools
import os
from contextlib import contextmanager, suppress
from itertools import cycle, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from data_analysis.correlation_function import SolutionSFCSExperiment
from data_analysis.polymer_physics import (
    dawson_structure_factor_fit,
    debye_structure_factor_fit,
    screened_dawson_structure_factor_fit,
    screened_debye_structure_factor_fit,
)
from utilities.display import Plotter, default_colors, plot_acfs
from utilities.fit_tools import FitError
from utilities.helper import Gate, get_func_attr

try:
    from winsound import Beep as beep  # type: ignore # Windows
except ModuleNotFoundError:

    def beep():
        """Beep to notify that the script has finished"""
        os.system('osascript -e "beep 3"')


@contextmanager
def mpl_backend(backend: Optional[str] = None, verbose=False):
    """Temporarily switch Matplotlib backend. For use in Jupyter notebooks."""
    if backend is None:
        yield
        return

    original_backend = mpl.get_backend()
    if original_backend == backend:
        yield
        return
    if verbose:
        print(
            f"[MPL BACKEND] Switching Matplotlib backend "
            f"from '{original_backend}' to '{backend}'..."
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
                self._data_root
                / config["confocal_date"]
                / "solution"
                / config["confocal_template"]
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
            force_processing = (
                data_config[label].pop("force_processing", False) or self._force_processing
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

                # calibrate TDC - only if STED is loaded and not already synced (which is the
                # case when filtering afterpulsing)
                if exp.sted.is_loaded and not exp.sted.afterpulsing_method == "filter":
                    with mpl_backend("inline"):
                        exp.calibrate_tdc(
                            **data_config[label],
                            force_processing=force_processing,
                            is_verbose=True,
                        )

                # save processed data (to avoid re-processing)
                exp.save_processed_measurements(
                    should_force=data_config[label]["force_save"],
                    # NOTE: processed data is temporarily stored anyway, and kept if not cleared
                    # setting "should_save_data" False saves a lot of time.
                    # After analyzing, each experiment could be saved again (with its
                    # processed data) for faster loading the next time if more analysis
                    # needs to be done
                    should_save_data=False,
                    data_root=self._data_root,
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
        self._data_root = self._data_loader._data_root

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
        self,
        plot_type: str = "norm_vs_vt_um",
        confocal_only: bool = False,
        sted_only: bool = False,
        **kwargs,
    ):
        """
        Plot the correlation functions for a set of experiments.
        """
        with Plotter(super_title="All Experiments, All ACFs", figsize=(8, 6)) as ax:
            for label, exp in self.filtered_exp_dict.items():
                with suppress(AttributeError):

                    # normalized linear vt_um
                    if plot_type == "norm_vs_vt_um":
                        exp.plot_correlation_functions(
                            x_scale="log",
                            y_scale="linear",
                            parent_ax=ax,
                            sted_only=sted_only,
                            confocal_only=confocal_only,
                            **kwargs,
                        )

                    # normalized vs. log lag
                    elif plot_type == "norm_vs_log_lag":
                        exp.plot_correlation_functions(
                            x_field="lag",
                            parent_ax=ax,
                            sted_only=sted_only,
                            confocal_only=confocal_only,
                            **kwargs,
                        )

                    # avg_cf_cr vs. log lag
                    elif plot_type == "avg_cf_cr_vs_log_lag":
                        exp.plot_correlation_functions(
                            x_field="lag",
                            y_field="avg_cf_cr",
                            parent_ax=ax,
                            sted_only=sted_only,
                            confocal_only=confocal_only,
                            **kwargs,
                        )

                    # log normalized vs. vt_um_sq
                    elif plot_type == "log_norm_vs_vt_um_sq":
                        exp.plot_correlation_functions(
                            x_scale="quadratic",
                            parent_ax=ax,
                            confocal_only=confocal_only,
                            sted_only=sted_only,
                            **kwargs,
                        )

                print(f"\n{label} countrates:\nConfocal - {exp.confocal.avg_cnt_rate_khz:.2f} kHz")

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
                    data_root=self._data_root,
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
                    f"    STED countrate: {sted_cr:.2f} +/- "
                    f"{exp.sted.std_cnt_rate_khz:.2f} kHz"
                )
                print(
                    f"    STED Molecules in Sampling Volume: "
                    f"{sted_cr / (list(exp.sted.cf.values())[0].g0 / 1e3):.2f}"
                )
            except IndexError:
                print("STED measurement not loaded!")
            print("\n")

    @skip_if_all_exp_filtered
    def remove_gates(self, meas_type: Optional[str] = None, force: bool = False):
        """
        Remove all gates from a set of experiments.
        """
        for label, exp in self.filtered_exp_dict.items():
            if self._data_config[label]["was_processed"] or force:
                # If measurement type is not specified, remove all gates
                if meas_type is None:
                    exp.remove_gates(should_plot=False, meas_type="confocal")
                    exp.remove_gates(should_plot=False, meas_type="sted")
                # Otherwise, remove gates for the specified measurement type
                else:
                    exp.remove_gates(should_plot=False, meas_type=meas_type)
                try:
                    # save processed meas (data not re-saved - should be quick)
                    exp.save_processed_measurements(
                        should_save_data=False,
                        # processed files should already exist at this point, so need to force
                        should_force=True,
                    )
                except AttributeError:
                    print("\nCode updated? did not save!\n")
        print("Removed all gates from all experiments")

    @skip_if_all_exp_filtered
    def gate(
        self,
        upper_gates: List[float],
        lower_gates: List[float],
        meas_type: str = "sted",
        force: bool = False,
    ):
        """
        Gate a set of experiments.
        """
        gate_list_lt = [Gate(gate_ns) for gate_ns in product(lower_gates, upper_gates)]

        for label, exp in self.filtered_exp_dict.items():
            if self._data_config[label]["was_processed"] or force:
                if exp.lifetime_params is None:
                    with mpl_backend("Qt5Agg"):
                        lt_params = exp.get_lifetime_parameters()
                else:
                    lt_params = exp.lifetime_params
                # NOTE: lt_params.laser_pulse_delay_ns is inaccurate.
                # Using calibrated value from "Laser Propagation Time Calibration.ipynb"
                pulse_delay_ns = 3.70
                lifetime_ns = lt_params.lifetime_ns
                gate_list_ns = [
                    round(gate * lifetime_ns + pulse_delay_ns, 2) for gate in gate_list_lt
                ]
                print(f"{label}:\n{lt_params}")
                exp.add_gates(gate_list_ns, meas_type=meas_type, should_plot=False)
                # save processed meas (data not re-saved - should be quick)
                exp.save_processed_measurements(
                    meas_types=["sted"],  # re-save only STED after gating (confocal unchaged)
                    should_save_data=False,
                    # processed files should already exist at this point, so need to force
                    should_force=True,
                )
            else:
                print(f"{label}: Using pre-processed...")
        print("Gated all experiments")

    @skip_if_all_exp_filtered
    def re_average_acfs(
        self,
        norm_range: Tuple[float, float] = (2.7e-3, 3.7e-3),
        should_use_clustering: bool = True,
        rejection: int = 2,
        noise_thresh: float = 0.5,
        force: bool = False,
    ):
        """
        Re-average the ACFs for a set of experiments.
        """
        with mpl_backend("inline"):
            for label, exp in self.filtered_exp_dict.items():
                if self._data_config[label]["was_processed"] or force:
                    print(f"Re-averaging '{label}'...")
                    exp.re_average_all(
                        norm_range=norm_range,
                        should_use_clustering=should_use_clustering,
                        rejection=rejection,
                        noise_thresh=noise_thresh,
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
    def calculate_hankel_transforms(self, force: bool = False, save_data: bool = False, **kwargs):
        """
        Calculate the Hankel transforms for a set of experiments.
        """
        with mpl_backend("Qt5Agg"):
            for label, exp in self.filtered_exp_dict.items():
                x_lim = (0.01, 1.0)
                with Plotter(**kwargs) as ax:
                    for (cf_name, cf), color in zip(exp.cf_dict.items(), cycle(default_colors)):
                        if not getattr(cf, "hankel_transforms", False) or force:
                            cf.calculate_hankel_transform(
                                ["gaussian"],
                                rmax=200,
                                title_prefix=f"{label}: ",
                                parent_ax=ax,
                                x_lim=x_lim,
                            )
                            # remove the last two plots (data and noise estimate), and plot
                            # the interpolation instead
                            [line.remove() for line in ax.lines[-2:]]
                            cf.hankel_transforms["gaussian"].plot_interpolation(
                                ax=ax, color=color, label_prefix=cf_name
                            )

                            # save processed meas (data not re-saved - should be quick)
                            if save_data:
                                exp.save_processed_measurements(
                                    should_save_data=False,
                                    # processed files should already exist at this point,
                                    # so need to force
                                    should_force=True,
                                    data_root=self._data_root,
                                )
                            else:
                                print(f"{label}: Warning - data not saved! (save_data=False)")

                        else:
                            print(f"{label}: Using existing...")

                        # Get the last x_lim for the next CorrFunc
                        extrap_point = cf.hankel_transforms["gaussian"].IE.x_lims.upper
                        new_max_x = max(extrap_point * 1.2, x_lim[1])
                        x_lim = (0.01, new_max_x)

    @skip_if_all_exp_filtered
    def plot_hankel_transforms(self, backend="inline", **kwargs):
        """
        Plot the Hankel transforms for a set of experiments.
        """
        with mpl_backend(backend):
            for label, exp in self.filtered_exp_dict.items():
                exp.plot_hankel_transforms(**kwargs)

    @skip_if_all_exp_filtered
    def calculate_structure_factors(
        self, exp_label2cal_label: Dict[str, str], force: bool = False
    ):
        """
        Calculate the structure factors for a set of experiments.
        """

        with mpl_backend("Qt5Agg"):
            for exp, cal_exp in [
                (self.exp_dict[exp_label], self.exp_dict[cal_label])
                for exp_label, cal_label in exp_label2cal_label.items()
                if exp_label in self.filtered_exp_dict
            ]:
                exp.calculate_structure_factors(cal_exp, rmax=200, should_force=force)

                # save processed meas (data not re-saved - should be quick)
                exp.save_processed_measurements(
                    should_save_data=False,
                    # processed files should already exist at this point, so need to force
                    should_force=True,
                    data_root=self._data_root,
                    verbose=False,
                )

    @skip_if_all_exp_filtered
    def plot_structure_factors(self, backend: Optional[str] = None, **kwargs):
        """
        Plot the structure factors for a set of experiments, each with a unique color.
        """
        with mpl_backend(backend):
            with Plotter(super_title="Structure Factors", figsize=(8, 6)) as ax:
                # Create a color cycle to assign a unique color to each experiment
                color_cycle = cycle(default_colors)

                for exp_label, exp in self.filtered_exp_dict.items():
                    if hasattr(exp, "cal_exp"):
                        # Get the next color from the cycle
                        color = next(color_cycle)
                        # Pass the color to the plot_structure_factors method
                        exp.plot_structure_factors(parent_ax=ax, color=color, **kwargs)

    @skip_if_all_exp_filtered
    def fit_confocal_structure_factors(self, interp_type: str = "gaussian", **kwargs):
        """
        Fit the structure factors for a set of experiments.
        """
        # Re-order the experiments to fit the dilute experiments first
        dilute_first_exp_labels = list(self.filtered_exp_dict.keys())
        dilute_first_exp_labels.sort(key=lambda x: " D " in x, reverse=True)

        # Fit the structure factors for each experiment, with the dilute experiments first
        for exp_label, exp in [
            (label, self.filtered_exp_dict[label]) for label in dilute_first_exp_labels
        ]:
            # Get the appropriate fit function for the experiment
            try:
                fit_func = self._get_structure_factor_fit_func(exp_label)
            except ValueError as e:
                print(f"Can't select fit function for '{exp_label}': {e}, skipping...")
                continue
            print(f"Using fit function: {get_func_attr(fit_func, '__name__')} for '{exp_label}'")
            # Fit the structure factors for the confocal measurements
            try:
                exp.confocal.cf["confocal"].structure_factors[interp_type].fit(fit_func, **kwargs)
            except FitError as e:
                print(f"Error fitting '{exp_label}': {e}, skipping...")

    def _get_structure_factor_fit_func(self, label: str):
        """
        Returns the appropriate structure factor fit function for the given label.
        The logic is as follows:
        * If the label contains " L " (linearized), return a "debye" fit function
        * If the label contains " OC " (nicked), return a "dawson" fit function
        * If the label contains " D " (dilute), or "PL " (partially labeled),
            return a "regular" fit function
        * If the label contains " SD " (semi-dilute), return a "screened" fit function - in this
            case, the Rg_dilute and B_dilute values must be provided by the fit for the dilute
            version of the experiment
        """

        def get_dilute_params(semidilute_label: str):
            """
            Get the dilute parameters from the label.
            """
            # Get the dilute label
            dilute_label = semidilute_label.replace(" SD ", " D ")
            # Get the fit parameters for the dilute label
            dilute_fit_params = (
                self.filtered_exp_dict[dilute_label]
                .confocal.cf["confocal"]
                .structure_factors["gaussian"]
                .fit_params
            )
            # Get the Rg_dilute and B_dilute values from the fit parameters
            return dilute_fit_params.beta["Rg"], dilute_fit_params.beta["B"]

        if " L " in label:
            if " D " in label or "PL " in label:
                return debye_structure_factor_fit
            elif " SD " in label:
                Rg_dilute, B_dilute = get_dilute_params(label)
                # return the screened fit function with the dilute values
                return functools.partial(
                    screened_debye_structure_factor_fit, Rg_dilute=Rg_dilute, B_dilute=B_dilute
                )
            else:
                raise ValueError(f"Label {label} contains ' L ' but not ' D ', 'PL ' or ' SD '!")
        elif " OC " in label:
            if " D " in label or "PL " in label:
                return dawson_structure_factor_fit
            elif " SD " in label:
                Rg_dilute, B_dilute = get_dilute_params(label)
                return functools.partial(
                    screened_dawson_structure_factor_fit, Rg_dilute=Rg_dilute, B_dilute=B_dilute
                )
            else:
                raise ValueError(f"Label {label} contains ' OC ' but not ' D ', 'PL ' or ' SD '!")
        else:
            raise ValueError(f"Label {label} does not contain ' L ' or ' OC '!")

    @skip_if_all_exp_filtered
    def print_confocal_structure_factor_fitted_parameters(self):
        """
        Print the fitted parameters for the structure factors of a set of experiments.
        """
        for exp_label, exp in self.filtered_exp_dict.items():
            if hasattr(exp, "cal_exp"):
                structure_factor = exp.confocal.cf["confocal"].structure_factors["gaussian"]
                if structure_factor.fit_params is not None:
                    print(f"{exp_label}:")
                    structure_factor.fit_params.print_fitted_params()
                    print()
