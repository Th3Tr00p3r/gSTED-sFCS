"""Plotting and image-showing utilities"""

from contextlib import suppress
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

    def display_image(self, image: np.ndarray, axes="on"):
        """Doc."""

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(image)
        self.ax.axis(axes)
        force_aspect(self.ax, aspect=1)
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


def force_aspect(ax, aspect=1) -> None:
    """
    See accepted answer here:
    https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    """

    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
