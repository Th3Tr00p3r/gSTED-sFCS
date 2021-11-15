"""Plotting and image-showing utilities"""

from contextlib import contextmanager
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen

from utilities.helper import LimitRange


class GuiDisplay:
    """Doc."""

    def __init__(self, layout, parent=None):
        self.figure = plt.figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        if parent is not None:
            self.toolbar = NavigationToolbar(self.canvas, parent)
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        """Doc."""

        self.figure.clear()
        self.canvas.draw_idle()

    def entitle_and_label(self, x_label: str = "", y_label: str = "", title: str = ""):
        """Doc"""

        with self._show_internal_ax(clear=False) as ax:
            if title:
                ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

    def plot(self, x, y, *args, **kwargs):
        """Wrapper for matplotlib.pyplot.plot."""

        try:
            self.ax.plot(x, y, *args, **kwargs)
        except AttributeError:
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(x, y, *args, **kwargs)
        finally:
            self.canvas.draw_idle()

    def display_pattern(self, x, y):
        """Doc."""

        with self._show_internal_ax(show_axis=False) as ax:
            ax.plot(x, y, "k", lw=0.3)

    def display_image(self, image: np.ndarray, cursor=False, *args, **kwargs):
        """Doc."""

        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be of type numpy.ndarray")

        with self._show_internal_ax(
            fix_aspect=True,
            show_axis=False,
            scroll_zoom=True,
            cursor=cursor,
        ) as ax:
            ax.imshow(image, *args, **kwargs)

    def plot_acfs(
        self, x: Tuple[np.ndarray, str], avg_cf_cr: np.ndarray, g0: float, cf_cr: np.ndarray = None
    ):
        """Doc."""

        x_arr, x_type = x

        with self._show_internal_ax() as ax:
            if x_type == "lag":
                ax.set_xscale("log")
                ax.set_xlim(1e-4, 1e1)
                if g0 > 0:
                    try:
                        ax.set_ylim(-g0 / 2, g0 * 2)
                    except ValueError:
                        # g0 is not a finite number
                        ax.set_ylim(-1e4, 1e4)
                else:
                    ax.set_ylim(-1e4, 1e4)
            if x_type == "disp":
                ax.set_yscale("log")
                x_arr = x_arr ** 2
                ax.set_xlim(0, 0.6)
                ax.set_ylim(g0 * 1.5e-3, g0 * 1.1)

            if cf_cr is not None:
                for row_acf in cf_cr:
                    ax.plot(x_arr, row_acf)
            ax.plot(x_arr, avg_cf_cr, "k")

    @contextmanager
    def _show_internal_ax(
        self,
        clear=True,
        fix_aspect=False,
        show_axis=None,
        scroll_zoom=False,
        cursor=False,
    ):
        """Doc."""

        try:
            last_pos = self.ax.cursor.pos
        except AttributeError:
            last_pos = (0, 0)

        if clear:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
        else:
            ax = self.ax

        try:
            yield ax
        finally:
            if fix_aspect:
                force_aspect(ax, aspect=1)
            if scroll_zoom:
                ax.org_lims = (LimitRange(ax.get_xlim()), LimitRange(ax.get_ylim()))
                ax.org_dims = tuple(lim.interval() for lim in ax.org_lims)
                ax.zoom_func = zoom_factory(ax)
            if cursor:
                ax.cursor = cursor_factory(ax, last_pos)
            if show_axis is not None:
                ax.axis(show_axis)
            self.ax = ax
            self.canvas.draw_idle()


class Plotter:
    """A generalized, hierarchical plotting tool, designed to work as a context manager."""

    AX_SIZE = (3.75, 2.5)

    def __init__(self, **kwargs):
        self.parent_figure = kwargs.get(
            "parent_figure"
        )  # TODO: not currently used. Can possibly be used for implementing subfigures
        self.parent_ax = kwargs.get("parent_ax")
        self.subplots = kwargs.get("subplots", (1, 1))
        self.figsize = kwargs.get("figsize")
        self.super_title = kwargs.get("super_title")
        self.should_force_aspect = kwargs.get("should_force_aspect", False)
        self.fontsize = kwargs.get("fontsize", 14)
        self.xlim: Tuple[float, float] = kwargs.get("xlim")
        self.ylim: Tuple[float, float] = kwargs.get("ylim")
        self.should_autoscale = kwargs.get("should_autoscale", False)

    def __enter__(self):
        """Prepare the 'axes' object to use in context manager"""

        # dealing with a figure object
        if self.parent_ax is None:
            if self.parent_figure is None:  # creating a new figure
                if self.figsize is None:  # auto-determine size
                    n_rows, n_cols = self.subplots
                    ax_width, ax_height = self.AX_SIZE
                    figsize = (n_cols * ax_width, n_rows * ax_height)
                self.fig = plt.figure(figsize=figsize, constrained_layout=True)
                self.axes = self.fig.subplots(*self.subplots)
                if not hasattr(self.axes, "size"):  # if self.axes is not an ndarray
                    self.axes = np.array([self.axes])
            else:  # using given figure
                self.axes = np.array(self.parent_figure.get_axes())

        # dealing with a axes object
        else:
            if not hasattr(self.parent_ax, "size"):  # if parent_ax is not an ndarray
                self.axes = np.array([self.parent_ax])
            else:
                self.axes = self.parent_ax

        if self.axes.size == 1:
            return self.axes[0]  # return a single Axes object
        else:
            return self.axes  # return a Numpy ndarray of Axes objects

    def __exit__(self, *exc):
        """
        Set axes attributes.
        Set figure attirbutes and show it, if Plotter is at top of hierarchy.
        """

        for ax in self.axes.flatten().tolist():  # set ax attributes
            if self.should_force_aspect:
                force_aspect(ax, aspect=1)
            if self.should_autoscale:
                ax.autoscale()
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            [
                text.set_fontsize(self.fontsize)
                for text in [ax.title, ax.xaxis.label, ax.yaxis.label]
            ]

        if self.parent_ax is None:  # set figure attributes, and show it (dealing with figure)
            self.fig.suptitle(self.super_title, fontsize=self.fontsize)
            self.fig.show()


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
        ax.figure.canvas.toolbar.push_current()
        # get the current x and y limits
        cur_xlim = LimitRange(ax.get_xlim())
        cur_ylim = LimitRange(ax.get_ylim())
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
        new_xlim = LimitRange(xdata - cur_xrange * scale_factor, xdata + cur_xrange * scale_factor)
        new_ylim = LimitRange(ydata + cur_yrange * scale_factor, ydata - cur_yrange * scale_factor)
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
