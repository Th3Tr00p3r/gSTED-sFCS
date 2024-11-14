from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt

from data_analysis.correlation_function import SolutionSFCSExperiment
from utilities.display import Plotter


@contextmanager
def mpl_backend_switcher(backend, verbose=False):
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
        plt.pause(0.001)  # Give the event loop time to process
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
                with mpl_backend_switcher("inline"):
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
                    with mpl_backend_switcher("inline"):
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

        return exp_dict


class SolutionSFCSExperimentHandler:
    """
    Class to handle Solution SFCS experiments.
    """

    def __init__(self, exp_dict):
        self.exp_dict = exp_dict
        self._positive_filters: Tuple[str, ...] = ("",)
        self._negative_filters: Tuple[str, ...] = ()

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

    def plot_afterpulsing_filters(self, backend="inline", **kwargs):
        """
        Plot the afterpulsing filters for a set of experiments.
        """
        # Skip if no experiments match the current filters
        if not self.filtered_exp_dict:
            return

        with mpl_backend_switcher(backend):
            for label, exp in self.filtered_exp_dict.items():
                print(f"{label}:")
                exp.plot_afterpulsing_filters(**kwargs)

    def plot_correlation_functions(
        self, confocal_only: bool = False, sted_only: bool = False, **kwargs
    ):
        """
        Plot the correlation functions for a set of experiments.
        """
        # Skip if no experiments match the current filters
        if not self.filtered_exp_dict:
            return

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

                with suppress(IndexError):
                    print(f"STED - {exp.sted.avg_cnt_rate_khz:.2f} kHz")
