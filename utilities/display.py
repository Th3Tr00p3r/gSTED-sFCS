"""Plotting and image-showing utilities"""

from contextlib import contextmanager, suppress
from typing import Tuple

import numpy as np
import pyqtgraph as pg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class AnalysisDisplay:
    """Doc."""

    def __init__(self, layout, parent=None):
        self.figure = plt.figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        if parent:
            self.toolbar = NavigationToolbar(self.canvas, parent)
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        """Doc."""

        self.figure.clear()
        self.canvas.draw()

    def entitle_and_label(self, x_label: str = "", y_label: str = "", title: str = ""):
        """Doc"""

        if title:
            self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.canvas.draw()

    def plot(self, x, y, **kwargs):
        """Wrapper for matplotlib.pyplot.plot."""

        try:
            self.ax.plot(x, y, **kwargs)
        except AttributeError:
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(x, y, **kwargs)
        finally:
            self.canvas.draw()

    def display_pattern(self, x, y):
        """Doc."""

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(x, y, color="black", lw=0.3)
        self.ax.axis(False)
        self.canvas.draw()

    def display_image(self, image: np.ndarray, axis=True, cursor=False):
        """Doc."""

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(image)
        self.ax.axis(axis)
        force_aspect(self.ax, aspect=1)
        self.ax.org_edges = (
            self.ax.get_xlim()[1] - self.ax.get_xlim()[0],
            self.ax.get_ylim()[0] - self.ax.get_ylim()[1],
        )
        self.zoom_func = zoom_factory(self.ax)
        if cursor:
            self.cursor = cursor_factory(self.ax)
        self.canvas.draw()

    def plot_acfs(
        self, x: (np.ndarray, str), average_cf_cr: np.ndarray, g0: float, cf_cr: np.ndarray = None
    ):
        """Doc."""

        x, x_type = x

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        if x_type == "lag":
            self.ax.set_xscale("log")
            self.ax.set_xlim(1e-4, 1e1)
            try:
                self.ax.set_ylim(-g0 / 2, g0 * 2)
            except ValueError:
                # g0 is not a finite number
                self.ax.set_ylim(-1e4, 1e4)
        if x_type == "disp":
            self.ax.set_yscale("log")
            x = x ** 2
            self.ax.set_xlim(0, 0.6)
            self.ax.set_ylim(g0 * 1.5e-3, g0 * 1.1)

        if cf_cr is not None:
            for row_acf in cf_cr:
                self.ax.plot(x, row_acf)
        self.ax.plot(x, average_cf_cr, "black")
        self.canvas.draw()


class ImageScanDisplay:
    """Doc."""

    def __init__(self, layout):
        glw = pg.GraphicsLayoutWidget()
        self.vb = glw.addViewBox()
        self.hist = pg.HistogramLUTItem()
        glw.addItem(self.hist)
        layout.addWidget(glw)

    def replace_image(self, image: np.ndarray, limit_zoomout=True, crosshair=True):
        """Doc."""

        self.vb.clear()
        image_item = pg.ImageItem(image)
        self.vb.addItem(image_item)
        self.hist.setImageItem(image_item)

        if limit_zoomout:
            self.vb.setLimits(
                xMin=0,
                xMax=image.shape[0],
                minXRange=0,
                maxXRange=image.shape[0],
                yMin=0,
                yMax=image.shape[1],
                minYRange=0,
                maxYRange=image.shape[1],
            )
        if crosshair:
            self.vLine = pg.InfiniteLine(angle=90, movable=True)
            self.hLine = pg.InfiniteLine(angle=0, movable=True)
            self.vb.addItem(self.vLine)
            self.vb.addItem(self.hLine)
            with suppress(AttributeError):  # AttributeError -  first image since init
                self.move_crosshair(self.last_roi)  # keep crosshair at last position
            self.vb.scene().sigMouseClicked.connect(self.mouseClicked)

        self.vb.autoRange()

    def move_crosshair(self, loc: Tuple[float, float]):
        """Doc."""

        self.vLine.setPos(loc[0])
        self.hLine.setPos(loc[1])
        self.last_roi = loc

    def mouseClicked(self, evt):
        """Doc."""
        # TODO: selected position is not accurate for some reason.
        # TODO: also note the location and the value at that point on the GUI (check out pyqtgraph examples)

        with suppress(AttributeError):  # AttributeError - outside image
            pos = evt.pos()
            mousePoint = self.vb.mapSceneToView(pos)
            self.move_crosshair(loc=(mousePoint.x(), mousePoint.y()))


def cursor_factory(ax):
    """Doc."""

    class Cursor:
        """
        A cross hair cursor.
        """

        def __init__(self, ax):
            self.ax = ax
            self.horizontal_line = ax.axhline(color="k", lw=0.8, ls="--")
            self.vertical_line = ax.axvline(color="k", lw=0.8, ls="--")
            # text location in axes coordinates
            self.text = ax.text(0.72, 0.9, "", transform=ax.transAxes)

        def set_cross_hair_visible(self, visible):
            need_redraw = self.horizontal_line.get_visible() != visible
            self.horizontal_line.set_visible(visible)
            self.vertical_line.set_visible(visible)
            self.text.set_visible(visible)
            return need_redraw

        def on_mouse_release(self, event):
            if not event.inaxes:
                need_redraw = self.set_cross_hair_visible(False)
                if need_redraw:
                    self.ax.figure.canvas.draw()
            else:
                self.set_cross_hair_visible(True)
                x, y = event.xdata, event.ydata
                # update the line positions
                self.horizontal_line.set_ydata(y)
                self.vertical_line.set_xdata(x)
                #                self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
                self.ax.figure.canvas.draw()

    cursor = Cursor(ax)
    ax.figure.canvas.mpl_connect("button_release_event", cursor.on_mouse_release)
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
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        # set the range
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * 0.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * 0.5
        xmouse = event.xdata  # get event x location
        ymouse = event.ydata  # get event y location
        cur_xcentre = (cur_xlim[1] + cur_xlim[0]) * 0.5
        cur_ycentre = (cur_ylim[1] + cur_ylim[0]) * 0.5
        xdata = cur_xcentre + 0.25 * (xmouse - cur_xcentre)
        ydata = cur_ycentre + 0.25 * (ymouse - cur_ycentre)
        if event.button == "up":
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == "down":
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        #        org_lims
        new_xlim = [xdata - cur_xrange * scale_factor, xdata + cur_xrange * scale_factor]
        new_ylim = [ydata - cur_yrange * scale_factor, ydata + cur_yrange * scale_factor]
        # limit zoom out
        org_width, org_height = ax.org_edges
        if ((new_xlim[1] - new_xlim[0]) > org_width * 1.1) or (
            (new_ylim[0] - new_ylim[1]) > org_height * 1.1
        ):
            return
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw_idle()  # force re-draw the next time the GUI refreshes

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect("scroll_event", zoom_fun)

    # return the function
    return zoom_fun


@contextmanager
def ax_show(should_force_aspect=False):
    """
    Creates a Matplotlib figure, and yields a single ax object
    which is to be manipulated, then shows the figure.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        yield ax
    finally:
        if should_force_aspect:
            force_aspect(ax, aspect=1)
        fig.show()


def force_aspect(ax, aspect=1) -> None:
    """
    See accepted answer here:
    https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    """

    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
