"""Plotting and image-showing utilities"""

from collections import namedtuple
from contextlib import suppress
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import to_rgb
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen

from utilities.helper import Limits

plt.rcParams.update({"figure.max_open_warning": 100})


class GuiDisplay:
    """Doc."""

    GuiDisplayOptions = namedtuple(
        "GuiDisplayOptions",
        "clear fix_aspect show_axis scroll_zoom cursor",
        defaults=(True, False, None, False, False),
    )

    def __init__(self, layout, gui_parent=None):
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

    def plot(self, x, y, *args, **kwargs):
        """Wrapper for matplotlib.pyplot.plot."""

        options = self.GuiDisplayOptions(clear=False)
        with Plotter(gui_display=self, gui_options=options) as ax:
            ax.plot(x, y, *args, **kwargs)

    def display_patterns(self, xy_pairs_list: List[tuple], *args, labels=None, scroll_zoom=False):
        """ "Doc."""

        with Plotter(
            gui_display=self,
            gui_options=self.GuiDisplayOptions(show_axis=False, scroll_zoom=scroll_zoom),
        ) as ax:
            try:
                for x, y in xy_pairs_list:
                    ax.plot(x, y, lw=0.4)
                    ax.legend(labels, fontsize=6)
            except TypeError:  # xy_pairs_list is a really 'x', 'y' is in args
                ax.plot(xy_pairs_list, *args, "k", lw=0.3)
            ax.set_aspect("equal")

    def display_image(self, image: np.ndarray, imshow_kwargs=dict(), **kwargs):
        """Doc."""

        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be of type numpy.ndarray")

        options = self.GuiDisplayOptions(
            fix_aspect=True,
            show_axis=False,
            scroll_zoom=kwargs.get("scroll_zoom", True),
            cursor=kwargs.get("cursor", False),
        )
        with Plotter(gui_display=self, gui_options=options) as ax:
            ax.imshow(image, interpolation="none", **imshow_kwargs)

    def plot_acfs(
        self,
        x: Tuple[np.ndarray, str],
        avg_cf_cr: np.ndarray,
        g0: float,
        cf_cr: np.ndarray = None,
        **kwargs,
    ):
        """Doc."""

        x_arr, x_type = x

        if x_type == "lag":
            kwargs["x_scale"] = "log"
            kwargs["xlim"] = (1e-3, 1e1)
            if 0 < g0 < np.inf:
                kwargs["ylim"] = (-g0 / 2, g0 * 2)
            else:
                kwargs["ylim"] = (-1e4, 1e4)

        if x_type == "disp":
            kwargs["y_scale"] = "log"
            x_arr = x_arr ** 2
            kwargs["xlim"] = (0, 0.6)
            kwargs["ylim"] = (g0 * 1.5e-3, g0 * 1.1)

        with Plotter(gui_display=self, **kwargs) as ax:
            if cf_cr is not None:
                cmap = get_gradient_colormap(cf_cr.shape[0])
                ax.set_prop_cycle(color=cmap)
                ax.plot(x_arr, cf_cr.T, lw=0.4)
            ax.set_prop_cycle(color="k")
            ax.plot(x_arr, avg_cf_cr, lw=1.4)


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
        self.should_force_aspect = kwargs.get("should_force_aspect", False)
        self.fontsize = kwargs.get("fontsize", 14)
        self.xlim: Tuple[float, float] = kwargs.get("xlim")
        self.ylim: Tuple[float, float] = kwargs.get("ylim")
        self.x_scale: str = kwargs.get("x_scale")
        self.y_scale: str = kwargs.get("y_scale")
        self.should_autoscale = kwargs.get("should_autoscale", False)
        self.selection_limits: Limits = kwargs.get("selection_limits")

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
                self.axes = self.gui_display.figure.subplots(*self.subplots)
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
                self.axes = self.fig.subplots(*self.subplots)
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
                    ax.org_lims = (Limits(ax.get_xlim()), Limits(ax.get_ylim()))
                    ax.org_dims = tuple(lim.interval() for lim in ax.org_lims)
                    ax.zoom_func = zoom_factory(ax)
                if self.gui_options.cursor:
                    ax.cursor = cursor_factory(ax, self.gui_display.last_pos)
                with suppress(TypeError):  # TODO: test this (is type error correct?)
                    ax.axis(self.gui_options.show_axis)

        for ax in self.axes.flatten().tolist():  # set ax attributes
            if self.should_force_aspect:
                force_aspect(ax, aspect=1)
            if self.should_autoscale:
                ax.autoscale()
            if self.xlim is not None:
                ax.set_xlim(self.xlim)
            if self.ylim is not None:
                ax.set_ylim(self.ylim)
            if self.x_scale is not None:
                ax.set_xscale(self.x_scale)
            if self.y_scale is not None:
                ax.set_yscale(self.y_scale)
            [
                text.set_fontsize(self.fontsize)
                for text in [ax.title, ax.xaxis.label, ax.yaxis.label]
            ]
            if self.selection_limits is not None:  # manual selection
                self.fig.suptitle(self.super_title, fontsize=self.fontsize)
                self.fig.show()
                selected_points_list = self.fig.ginput(n=-1, timeout=-1)
                x_coords = [x for (x, y) in selected_points_list]
                if len(x_coords) < 2:
                    raise RuntimeError("Must select at least 2 points!")
                self.selection_limits(min(x_coords), max(x_coords))
                return

        if self.parent_ax is None:  # set figure attributes, and show it (dealing with figure)
            if self.gui_display is not None:
                self.gui_display.figure.suptitle(self.super_title, fontsize=self.fontsize)
                self.gui_display.canvas.draw_idle()
            else:
                self.fig.suptitle(self.super_title, fontsize=self.fontsize)
                self.fig.show()
                self.fig.canvas.draw_idle()


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

        def move_to_pos(self, pos):
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
        cur_xlim = Limits(ax.get_xlim())
        cur_ylim = Limits(ax.get_ylim())
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
        new_xlim = Limits(xdata - cur_xrange * scale_factor, xdata + cur_xrange * scale_factor)
        new_ylim = Limits(ydata + cur_yrange * scale_factor, ydata - cur_yrange * scale_factor)
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

    img, *_ = ax.get_images()
    extent = img.get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def get_gradient_colormap(n_lines, color_list=["magenta", "lime", "cyan"]) -> np.ndarray:
    """Create a multicolor gradient colormap"""

    # get a list of RGB colors from the list of colors provided (i.e. (1.0, 0.0, 1.0) for "magenta")
    rgb_color_list = [to_rgb(color) for color in color_list]
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
