"""Plotting and image-showing utilities"""

from contextlib import contextmanager

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class Display:
    """Doc."""

    def __init__(self, layout=None, parent=None):
        if layout is not None:
            self.figure = plt.figure(tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            if parent is not None:
                self.toolbar = NavigationToolbar(self.canvas, parent)
                layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)

    def clear(self):
        """Doc."""

        self.figure.clear()
        self.canvas.draw()

    def entitle_and_label(self, x_label: str = "", y_label: str = "", title: str = ""):
        """Doc"""

        with self._show_internal_ax() as ax:
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
            self.canvas.draw()

    def display_pattern(self, x, y):
        """Doc."""

        with self._show_internal_ax(clear=True, show_axis=False) as ax:
            ax.plot(x, y, "k", lw=0.3)

    def display_image(self, image: np.ndarray, axis=True, cursor=False, *args, **kwargs):
        """Doc."""

        with self._show_internal_ax(
            clear=True,
            fix_aspect=True,
            show_axis=False,
            scroll_zoom=True,
            cursor=cursor,
        ) as ax:
            ax.imshow(image, *args, **kwargs)

    def plot_acfs(
        self, x: (np.ndarray, str), average_cf_cr: np.ndarray, g0: float, cf_cr: np.ndarray = None
    ):
        """Doc."""

        x, x_type = x

        with self._show_internal_ax(clear=True) as ax:
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
                x = x ** 2
                ax.set_xlim(0, 0.6)
                ax.set_ylim(g0 * 1.5e-3, g0 * 1.1)

            if cf_cr is not None:
                for row_acf in cf_cr:
                    ax.plot(x, row_acf)
            ax.plot(x, average_cf_cr, "k")

    @contextmanager
    def _show_internal_ax(
        self,
        clear=False,
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
                ax.org_lims = (ax.get_xlim(), ax.get_ylim())
                ax.org_dims = tuple(abs(lim[1] - lim[0]) for lim in ax.org_lims)
                ax.zoom_func = zoom_factory(ax)
            if cursor:
                ax.cursor = cursor_factory(ax, last_pos)
            if show_axis is not None:
                ax.axis(show_axis)
            self.ax = ax
            self.canvas.draw()

    @contextmanager
    def show_external_ax(self):
        """
        Creates a Matplotlib figure, and yields a single ax object
        which is to be manipulated, then shows the figure.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            yield ax
        finally:
            force_aspect(ax, aspect=1)
            fig.show()


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

        def _on_mouse_release(self, event):
            if not event.inaxes:
                need_redraw = self._set_crosshair_visible(False)
                if need_redraw:
                    self.ax.figure.canvas.draw()
            else:
                self._set_crosshair_visible(True)
                self.move_to_pos((event.xdata, event.ydata))

        def move_to_pos(self, pos):
            """Doc."""

            x, y = pos
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            #                self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()
            self.pos = pos

    cursor = Cursor(ax)
    ax.figure.canvas.mpl_connect("button_release_event", cursor._on_mouse_release)
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
        org_width, org_height = ax.org_dims
        if ((new_xlim[1] - new_xlim[0]) > org_width * 1.1) or (
            (new_ylim[0] - new_ylim[1]) > org_height * 1.1
        ):
            new_xlim, new_ylim = ax.org_lims
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw_idle()  # force re-draw the next time the GUI refreshes

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect("scroll_event", zoom_fun)

    # return the function
    return zoom_fun


def auto_brightness_and_contrast(image: np.ndarray, clip_hist_percent=0) -> np.ndarray:
    """
    See:
    https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    """

    def convert_scale(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        new_img = image * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)

    if clip_hist_percent == 0:
        return image

    # Estimate the best number of bins to use - choose 'n_bins' such that at least half have a single count in them
    for n_bins in reversed(range(10, 256)):
        bin_counts, _ = np.histogram(image, bins=n_bins)
        if (bin_counts > 0).sum() >= (bin_counts.size / 2):
            break

    # Calculate histogram
    # hist definition to match 'hist = cv2.calcHist([gray],[0],None,[256],[0,256])',
    # derived from https://stackoverflow.com/questions/25013732/comparing-rgb-histograms-plt-hist-np-histogram-and-cv2-comparehist
    hist = bin_counts.ravel().astype(float)
    hist_size = n_bins

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    #    print(f"alpha: {alpha:.1f}\nbeta: {beta:.1f}\n") # TESTESTEST
    return convert_scale(image, alpha=alpha, beta=beta)


def force_aspect(ax, aspect=1) -> None:
    """
    See accepted answer here:
    https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    """

    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)