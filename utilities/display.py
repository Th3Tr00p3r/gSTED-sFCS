"""Plotting and image-showing utilities"""

from collections import namedtuple
from contextlib import suppress
from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import ListedColormap, to_rgb
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen

import utilities.helper as helper

plt.rcParams.update({"figure.max_open_warning": 300})
default_colors = tuple(plt.rcParams["axes.prop_cycle"].by_key()["color"])


class GuiDisplay:
    """Doc."""

    GuiDisplayOptions = namedtuple(
        "GuiDisplayOptions",
        "clear fix_aspect show_axis scroll_zoom cursor",
        defaults=(True, False, None, False, False),
    )

    def __init__(self, layout, gui_parent=None):
        self.plotter = None
        self.figure = plt.figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        if gui_parent is not None:
            self.toolbar = NavigationToolbar(self.canvas, gui_parent)
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        """Doc."""

        self.figure.clear()
        self.canvas.draw_idle()

    def entitle_and_label(self, x_label: str = None, y_label: str = None, title: str = None):
        """Doc"""

        options = self.GuiDisplayOptions(clear=False, show_axis=True)
        with Plotter(gui_display=self, gui_options=options, super_title=title) as ax:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

    def plot(self, x, y, *args, should_clear=False, **kwargs):
        """Wrapper for matplotlib.pyplot.plot."""

        options = self.GuiDisplayOptions() if should_clear else self.GuiDisplayOptions(clear=False)
        with Plotter(gui_display=self, gui_options=options) as ax:
            ax.plot(x, y, *args, **kwargs)

    def display_patterns(self, xy_pairs_list: List[tuple], *args, labels=None, scroll_zoom=False):
        """ "Doc."""

        with Plotter(
            gui_display=self,
            gui_options=self.GuiDisplayOptions(show_axis=False, scroll_zoom=scroll_zoom),
        ) as ax:
            try:
                for (x, y), label in zip(xy_pairs_list, labels):
                    with suppress(ValueError):
                        # ValueError: x/y is NaN (no AO_int in legacy measurements?)
                        ax.plot(x, y, lw=0.4, label=label)
            except TypeError:  # xy_pairs_list is a really 'x', 'y' is in args
                ax.plot(xy_pairs_list, *args, "k", lw=0.3)
            else:
                ax.legend()
            ax.set_aspect("equal")
            ax.invert_xaxis()  # NOTE: this is to match the scan image orientation # TESTESTEST

    def display_image(self, image: np.ndarray, reuse_plotter=True, imshow_kwargs=dict(), **kwargs):
        """Doc."""

        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be of type numpy.ndarray")

        options = self.GuiDisplayOptions(
            fix_aspect=True,
            show_axis=False,
            scroll_zoom=kwargs.get("scroll_zoom", True),
            cursor=kwargs.get("cursor", False),
        )

        # optionally reuse plotter instance
        if not reuse_plotter or self.plotter is None:
            self.plotter = Plotter(gui_display=self, gui_options=options)

        with self.plotter as ax:
            ax.imshow(image, interpolation="none", **imshow_kwargs)

    def plot_acfs(self, *args, **kwargs):
        """Doc."""

        kwargs["gui_display"] = self
        plot_acfs(*args, **kwargs)

    def add_patch(self, patch, annotation=None, should_clear=True):
        """Add patches to existing Axes."""

        if should_clear:
            expandable_artists = [
                artist
                for artist in self.axes[0].get_children()
                if artist.get_label() == "expandable"
            ]
            for artist in expandable_artists:
                artist.remove()

        with suppress(AttributeError):
            patch.set_label("expandable")
            # plot patch
            self.axes[0].add_artist(patch)
            # annotate
            if annotation:
                self.axes[0].annotate(
                    annotation,
                    patch._center,
                    color="w",
                    weight="bold",
                    fontsize=11,
                    ha="center",
                    va="center",
                )
            self.canvas.draw_idle()


class Plotter:
    """A generalized, hierarchical plotting tool, designed to work as a context manager."""

    AX_SIZE = (3.75, 2.5)

    def __init__(self, **kwargs):
        self.parent_figure = kwargs.get(
            "parent_figure"
        )  # TODO: not currently used. Can be used for implementing 'subfigures'
        self.parent_ax = kwargs.get("parent_ax")
        self.gui_display: GuiDisplay = kwargs.get("gui_display")
        self.gui_options = kwargs.get("gui_options", GuiDisplay.GuiDisplayOptions())
        self.subplots = kwargs.get("subplots", (1, 1))
        self.figsize = kwargs.get("figsize")
        self.super_title = kwargs.get("super_title")
        self.xlabel = kwargs.get("xlabel")
        self.ylabel = kwargs.get("ylabel")
        self.should_force_aspect = kwargs.get("should_force_aspect", False)
        self.fontsize = kwargs.get("fontsize", 14)
        self.xlim: Tuple[float, float] = kwargs.get("xlim")
        self.ylim: Tuple[float, float] = kwargs.get("ylim")
        self.x_scale: str = kwargs.get("x_scale")
        self.y_scale: str = kwargs.get("y_scale")
        self.sharex: bool = kwargs.get("sharex", False)
        self.sharey: bool = kwargs.get("sharey", False)
        self.should_autoscale = kwargs.get("should_autoscale", False)
        self.selection_limits: helper.Limits = kwargs.get("selection_limits")
        self.selection_type: str = kwargs.get("selection_type", "both")
        self.should_close_after_selection = kwargs.get("should_close_after_selection", False)
        self.subplot_kw = kwargs.get("subplot_kw", {})  # dict(projection='3d')
        self.should_show = kwargs.get("should_show", False)  # show on exit

    def __enter__(self):
        """Prepare the 'axes' object to use in context manager"""

        # dealing with GUI plotting
        if self.gui_display is not None:
            try:
                self.gui_display.last_pos = self.gui_display.axes[0].cursor.pos
            except AttributeError:
                self.gui_display.last_pos = (0, 0)
            except IndexError:
                raise RuntimeError("Attempting to set last position of a multi-axes figure.")

            self.fig = self.gui_display.figure
            if self.gui_options.clear:  # clear and create new axes
                self.gui_display.figure.clear()
                self.axes = self.gui_display.figure.subplots(
                    *self.subplots, sharex=self.sharex, sharey=self.sharey
                )
                if not hasattr(self.axes, "size"):  # if self.axes is not an ndarray
                    self.axes = np.array([self.axes])
            else:  # use exising axes
                self.axes = np.array(self.gui_display.axes)
            self.gui_display.axes = self.axes

        # dealing with a figure object
        elif self.parent_ax is None:
            if self.parent_figure is None:  # creating a new figure
                if self.figsize is None:  # auto-determine size
                    n_rows, n_cols = self.subplots
                    ax_width, ax_height = self.AX_SIZE
                    self.figsize = (n_cols * ax_width, n_rows * ax_height)
                self.fig = plt.figure(figsize=self.figsize, constrained_layout=True)
                self.axes = self.fig.subplots(
                    *self.subplots,
                    sharex=self.sharex,
                    sharey=self.sharey,
                    subplot_kw=self.subplot_kw,
                )
                if not hasattr(self.axes, "size"):  # if self.axes is not an ndarray
                    self.axes = np.array([self.axes])
            else:  # using given figure
                self.fig = self.parent_figure
                self.axes = np.array(self.parent_figure.get_axes())

        # dealing with a axes object
        else:
            if not hasattr(self.parent_ax, "size"):  # if parent_ax is not an ndarray
                self.axes = np.array([self.parent_ax])
            else:
                self.axes = self.parent_ax
            try:  # 1D array of axes
                self.fig = self.axes[0].figure
            except AttributeError:  # 2D array of axes
                self.fig = self.axes[0][0].figure

        if self.axes.size == 1:
            return self.axes[0]  # return a single Axes object
        else:
            return self.axes  # return a Numpy ndarray of Axes objects

    def __exit__(self, *exc):
        """
        Set axes attributes.
        Set figure attirbutes and show it, if Plotter is at top of hierarchy.
        """

        if self.gui_display is not None:  # dealing with GUI plotting
            for ax in self.axes.flatten().tolist():
                if self.gui_options.fix_aspect:
                    self.should_force_aspect = True
                if self.gui_options.scroll_zoom:
                    ax.org_lims = (helper.Limits(ax.get_xlim()), helper.Limits(ax.get_ylim()))
                    ax.org_dims = tuple(lim.interval() for lim in ax.org_lims)
                    ax.zoom_func = zoom_factory(ax)
                if self.gui_options.cursor:
                    ax.cursor = cursor_factory(ax, self.gui_display.last_pos)
                with suppress(TypeError):  # TODO: test this (is type error correct?)
                    ax.axis(self.gui_options.show_axis)

        for ax in self.axes.flatten().tolist():
            # set ax attributes
            self._set_axis_attributes(ax)
            # manual selection
            if self.selection_limits is not None:
                self.fig.suptitle(self.super_title, fontsize=self.fontsize)
                self.fig.show()
                while True:
                    selected_points_list = self.fig.ginput(n=-1, timeout=-1)
                    x_coords = [x for (x, y) in selected_points_list]
                    stop_condition = (
                        len(x_coords) >= 2 if self.selection_type == "both" else x_coords
                    )
                    if stop_condition:
                        break
                if self.selection_type == "upper":  # select upper bound
                    self.selection_limits(self.selection_limits.lower, min(x_coords))
                elif self.selection_type == "lower":  # select lower bound
                    self.selection_limits(max(x_coords), self.selection_limits.upper)
                elif self.selection_type == "both":  # select both bounds
                    self.selection_limits(min(x_coords), max(x_coords))
                else:
                    raise ValueError("'selection_type' must be of 'upper', 'lower' or 'both'.")
                if self.should_close_after_selection:
                    plt.close(self.fig)
                return

        # show on exit (for testing)
        if self.should_show:
            self.fig.show()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if self.parent_ax is None:  # set figure attributes, and show it (dealing with figure)
            if self.gui_display is not None:
                self.gui_display.figure.suptitle(self.super_title, fontsize=self.fontsize)
                self.gui_display.canvas.draw_idle()
            else:
                self.fig.suptitle(self.super_title, fontsize=self.fontsize)
                self.fig.canvas.draw_idle()

    def _quadratic_xscale_backwards(self, x):
        """Doc"""

        if (x < 0).any():
            new_x = np.empty(x.shape)
            new_x[x < 0] = 0
            new_x[x >= 0] = x[x >= 0] ** (1 / 2)
            return new_x
        else:
            return x ** (1 / 2)

    def _set_axis_attributes(self, ax):
        """Doc."""

        if self.should_force_aspect:
            force_aspect(ax, aspect=1)
        if self.should_autoscale:
            ax.autoscale()
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        if self.x_scale is not None:
            if self.x_scale == "quadratic":
                ax.set_xscale(
                    "function", functions=(lambda x: x**2, self._quadratic_xscale_backwards)
                )
            else:
                ax.set_xscale(self.x_scale)
        if self.y_scale is not None:
            ax.set_yscale(self.y_scale)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        [text.set_fontsize(self.fontsize) for text in [ax.title, ax.xaxis.label, ax.yaxis.label]]


class NavigationToolbar(NavigationToolbar2QT):
    """
    Only display the buttons we need.
    https://stackoverflow.com/questions/12695678/how-to-modify-the-navigation-toolbar-easily-in-a-matplotlib-figure-window
    """

    toolitems = [
        t for t in NavigationToolbar2QT.toolitems if t[0] in {"Home", "Pan", "Zoom", "Save"}
    ]


def cursor_factory(ax, init_cursor_pos):
    """Doc."""

    class Cursor:
        """
        A crosshair cursor, adapted from:
        https://matplotlib.org/stable/gallery/misc/cursor_demo.html
        """

        def __init__(self, ax):
            self.ax = ax
            self.pos = init_cursor_pos
            x, y = init_cursor_pos
            self.horizontal_line = ax.axhline(y=y, color="y", lw=1, ls="--")
            self.vertical_line = ax.axvline(x=x, color="y", lw=1, ls="--")
            # text location in axes coordinates
            self.text = ax.text(0.72, 0.9, "", transform=ax.transAxes)

        def _set_crosshair_visible(self, visible):
            need_redraw = self.horizontal_line.get_visible() != visible
            self.horizontal_line.set_visible(visible)
            self.vertical_line.set_visible(visible)
            self.text.set_visible(visible)
            return need_redraw

        def _on_mouse_press(self, event):
            if not event.inaxes:
                need_redraw = self._set_crosshair_visible(False)
                if need_redraw:
                    self.ax.figure.canvas.draw_idle()
            else:
                self._set_crosshair_visible(True)
                self.move_to_pos((event.xdata, event.ydata))

        def move_to_pos(self, pos: Tuple[float, float]):
            """Doc."""

            x, y = pos
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.ax.figure.canvas.draw_idle()
            self.pos = pos

    cursor = Cursor(ax)
    ax.figure.canvas.mpl_connect("button_press_event", cursor._on_mouse_press)
    return cursor


def zoom_factory(ax, base_scale=1.5):
    """
    Enable zoom in/out by scrolling:
    https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
    """

    def zoom_fun(event):
        # fixes homebutton operation
        with suppress(AttributeError):
            # AttributeError - when toolbar is not defined
            ax.figure.canvas.toolbar.push_current()
        # get the current x and y limits
        cur_xlim = helper.Limits(ax.get_xlim())
        cur_ylim = helper.Limits(ax.get_ylim())
        # set the range
        cur_xrange = cur_xlim.interval() * 0.5
        cur_yrange = cur_ylim.interval() * 0.5
        xmouse = event.xdata  # get event x location
        ymouse = event.ydata  # get event y location
        cur_x_center = cur_xlim.center()
        cur_y_center = cur_ylim.center()
        try:
            xdata = cur_x_center + 0.25 * (xmouse - cur_x_center)
            ydata = cur_y_center + 0.25 * (ymouse - cur_y_center)
        except TypeError:
            # scrolling outside of image results in None
            return

        if event.button == "up":  # deal with zoom in
            scale_factor = 1 / base_scale
        else:  # deal with zoom out
            scale_factor = base_scale

        # set new limits
        new_xlim = helper.Limits(
            xdata - cur_xrange * scale_factor, xdata + cur_xrange * scale_factor
        )
        new_ylim = helper.Limits(
            ydata + cur_yrange * scale_factor, ydata - cur_yrange * scale_factor
        )
        # limit zoom out
        org_width, org_height = ax.org_dims
        if (new_xlim.interval() > org_width * 1.1) or (-new_ylim.interval() > org_height * 1.1):
            new_xlim, new_ylim = ax.org_lims
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw_idle()  # force re-draw the next time the GUI refreshes

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect("scroll_event", zoom_fun)

    # return the function
    return zoom_fun


def auto_brightness_and_contrast(image: np.ndarray, percent_factor=300) -> np.ndarray:
    """
    See:
    https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    """

    yen_threshold = threshold_yen(image) * percent_factor / 100
    return rescale_intensity(image, (0, yen_threshold), (0, image.max()))


def force_aspect(ax, aspect=1) -> None:
    """
    See accepted answer here:
    https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    """

    with suppress(ValueError):
        img, *_ = ax.get_images()
        extent = img.get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def get_gradient_colormap(
    n_lines,
    color_list: Optional[List[str | Tuple[int | float, int | float, int | float]]] = None,
    **kwargs,
) -> np.ndarray:
    """Create a multicolor gradient colormap."""
    color_list = color_list or ["magenta", "lime", "cyan"]

    # get a list of RGB colors from the list of colors provided (i.e. (1.0, 0.0, 1.0) for magenta)
    rgb_color_list = [to_rgb(color) if isinstance(color, str) else color for color in color_list]
    n_colors = len(rgb_color_list)

    # split gradient switch points evenly across the number of lines, and build the gradients
    cmap = np.ones((n_lines, 4))
    grad_switch_idxs = [int(n_lines / (n_colors - 1)) * i for i in range(n_colors)]
    for i in range(n_colors - 1):
        r1, g1, b1 = rgb_color_list[i]
        r2, g2, b2 = rgb_color_list[i + 1]

        gradient_slice = slice(grad_switch_idxs[i], grad_switch_idxs[i + 1])
        slice_len = grad_switch_idxs[i + 1] - grad_switch_idxs[i]

        cmap[gradient_slice, 0] = np.linspace(r1, r2, slice_len)
        cmap[gradient_slice, 1] = np.linspace(g1, g2, slice_len)
        cmap[gradient_slice, 2] = np.linspace(b1, b2, slice_len)

    return cmap


def plot_acfs(
    x: np.ndarray,
    avg_cf_cr: np.ndarray = None,
    g0: float = None,
    cf_cr: np.ndarray = None,
    j_good: list = [],
    n_lines=1000,  # was 14
    batch_sizes=None,
    **kwargs,
):
    """Doc."""

    if g0 is None:
        if avg_cf_cr is not None:
            g0 = avg_cf_cr[(x > 2e-3) & (x < 3e-3)].mean()
        elif cf_cr is not None:
            g0 = cf_cr.mean(axis=0)[(x > 2e-3) & (x < 3e-3)].mean()
        else:
            raise ValueError("Either 'avg_cf_cr' or 'cf_cr' must be supplied!")

    kwargs["x_scale"] = "log"
    kwargs["xlim"] = (1e-3, 1e1)
    if 0 < g0 < np.inf:
        kwargs["ylim"] = (-g0 / 2, g0 * 2)
    else:
        kwargs["ylim"] = (-1e4, 1e4)

    with Plotter(**kwargs) as ax:
        if cf_cr is not None:
            if j_good:
                cf_cr = cf_cr[j_good]
            cf_cr = (
                helper.batch_mean_rows(cf_cr, batch_sizes=batch_sizes)
                if batch_sizes
                else helper.batch_mean_rows(cf_cr, n_lines)
            )
            cmap = get_gradient_colormap(cf_cr.shape[0], **kwargs)
            ax.set_prop_cycle(color=cmap)
            ax.plot(x, cf_cr.T, lw=0.4)
            # Add a colorbar (colors only)
            sm = plt.cm.ScalarMappable(cmap=ListedColormap(cmap))
            sm.set_array([])
            ax.get_figure().colorbar(sm, ax=ax, orientation="vertical", format="")
        if avg_cf_cr is not None:
            ax.set_prop_cycle(color="k")
            ax.plot(x, avg_cf_cr, lw=1.4, label="avg_cf_cr")
        # add horizontal line at y=0
        ax.axhline(y=0, color="k", lw=0.3, ls="--")
        ax.legend()


def display_scatter(arr2d, labels, title="", cmap="viridis", **kwargs):
    """Expects `arr2d` to be a 2D array-like with samples along the 0th dimension"""

    if arr2d.ndim != 2:
        raise ValueError("Must supply a 2D array! (sample rows, feature columns)")

    if arr2d.shape[1] not in [2, 3]:
        raise ValueError("Can only display 2D or 3D data!")

    # prepare 3d projection if needed
    if arr2d.shape[1] == 3:
        kwargs["subplot_kw"] = dict(projection="3d")

    with Plotter(**kwargs) as ax:
        dim_slices = [col for col in arr2d.T]
        ax.scatter(*dim_slices, c=labels)
        ax.set_title(title)


def display_dim_reduction(arr2d, labels, name: str, **kwargs):
    """Expects a 2D matrix with 2/3 components (columns)"""
    components = range(1, arr2d.shape[1] + 1)
    axes = "xyz"
    axis_labels_dict = {
        f"{ax_str}label": f"Principle Component {component_num}"
        for ax_str, component_num in zip(axes, components)
    }
    display_scatter(arr2d, labels, **axis_labels_dict, title=f"{name} Visualization", **kwargs)


def move_labels_to_end_of_legend(axes: Axes | np.ndarray, labels_to_move: List[str]):
    """
    Move labels to the end of the legend.
    """
    # Skip if no labels to move
    if not labels_to_move:
        return

    # Convert single axes to array
    if isinstance(axes, Axes):
        axes = np.array([axes])

    for ax in axes:
        # Get the current legend handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Find the indexes of the labels to move
        idxs_to_move = [
            idx
            for idx, label in enumerate(labels)
            if any([ltm in label for ltm in labels_to_move])
        ]

        # If no matching labels are found, continue to the next axes
        if not idxs_to_move:
            continue

        # Sort indices in descending order to avoid shifting issues when popping
        idxs_to_move_sorted = sorted(idxs_to_move, reverse=True)

        moved_labels: List[str] = []
        moved_handles: List = []

        for idx in idxs_to_move_sorted:
            moved_labels.insert(0, labels.pop(idx))
            moved_handles.insert(0, handles.pop(idx))

        # Append the moved labels and handles at the end
        labels.extend(moved_labels)
        handles.extend(moved_handles)

        # Update the legend with the new order
        ax.legend(handles, labels)
