#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:38:29 2026

@author: bar
"""
import numpy as np
import matplotlib.pyplot as plt


def draw_polygon(fig=None, ax=None, close_fig=True, color='red'):
    """Draw a polygon on an existing axis and return its vertices.

    Left-click: add vertex. Right-click: delete nearest vertex.
    Space/Enter: finish.

    Returns
    -------
    np.ndarray of shape (N, 2) with the vertex coordinates (data units).
    """
    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    plt.ion()
    coords = []
    markers, = ax.plot([], [], 'x', color=color)
    outline, = ax.plot([], [], '-', color=color, lw=1)
    old_title = ax.get_title()
    ax.set_title('left click add, right click delete, space to finish')
    finished = False

    def redraw():
        if coords:
            xy = np.array(coords)
            closed = np.vstack([xy, xy[:1]])
            markers.set_data(xy[:, 0], xy[:, 1])
            outline.set_data(closed[:, 0], closed[:, 1])
        else:
            markers.set_data([], [])
            outline.set_data([], [])
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            coords.append((x, y))
        elif event.button == 3 and coords:
            d = [(x - cx) ** 2 + (y - cy) ** 2 for cx, cy in coords]
            coords.pop(int(np.argmin(d)))
        redraw()

    def onkey(event):
        nonlocal finished
        if event.key in (' ', 'enter'):
            finished = True

    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show(block=False)

    while not finished and plt.fignum_exists(fig.number):
        plt.pause(0.1)

    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    if close_fig:
        plt.close(fig)
    else:
        ax.set_title(old_title)

    return np.array(coords)