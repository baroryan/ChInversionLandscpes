import numpy as np
import matplotlib.pyplot as plt


class PolygonPicker:
    """Left-click add, right-click delete nearest, space/enter to finish."""

    def __init__(self, fig, ax, color='red'):
        self.fig, self.ax = fig, ax
        self.coords = []
        self.markers, = ax.plot([], [], 'x', color=color)
        self.outline, = ax.plot([], [], '-', color=color, lw=1)
        self.cids = [fig.canvas.mpl_connect('button_press_event', self._onclick),
                     fig.canvas.mpl_connect('key_press_event', self._onkey)]

    @property
    def xy(self):
        return np.array(self.coords)

    def _redraw(self):
        if self.coords:
            xy = self.xy
            closed = np.vstack([xy, xy[:1]])
            self.markers.set_data(xy[:, 0], xy[:, 1])
            self.outline.set_data(closed[:, 0], closed[:, 1])
        else:
            self.markers.set_data([], [])
            self.outline.set_data([], [])
        self.fig.canvas.draw_idle()

    def _onclick(self, event):
        if event.inaxes is not self.ax:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            self.coords.append((x, y))
        elif event.button == 3 and self.coords:
            d = [(x - cx)**2 + (y - cy)**2 for cx, cy in self.coords]
            self.coords.pop(int(np.argmin(d)))
        self._redraw()

    def _onkey(self, event):
        if event.key in (' ', 'enter'):
            for cid in self.cids:
                self.fig.canvas.mpl_disconnect(cid)