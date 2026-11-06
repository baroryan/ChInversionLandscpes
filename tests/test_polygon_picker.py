"""PolygonPicker driven by real matplotlib mouse/key events (no display needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402

import basinPickerFromAx as polysel  # noqa: E402


@pytest.fixture
def figure():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.canvas.draw()
    yield fig, ax
    plt.close(fig)


def _click(fig, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    fig.canvas.callbacks.process("button_press_event",
                                 MouseEvent("button_press_event", fig.canvas, px, py, button=button))


def _key(fig, key):
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key))


def test_left_clicks_add_vertices(figure):
    fig, ax = figure
    picker = polysel.PolygonPicker(fig, ax)
    verts = [(10, 10), (80, 15), (70, 90)]
    for v in verts:
        _click(fig, ax, *v)
    assert np.allclose(picker.xy, verts)


def test_right_click_removes_nearest(figure):
    fig, ax = figure
    picker = polysel.PolygonPicker(fig, ax)
    for v in [(10, 10), (80, 15), (70, 90)]:
        _click(fig, ax, *v)
    _click(fig, ax, 78, 17, button=3)
    assert np.allclose(picker.xy, [(10, 10), (70, 90)])


def test_space_stops_picking(figure):
    fig, ax = figure
    picker = polysel.PolygonPicker(fig, ax)
    _click(fig, ax, 10, 10)
    _key(fig, " ")
    _click(fig, ax, 50, 50)
    assert np.allclose(picker.xy, [(10, 10)])


def test_clicks_outside_axes_are_ignored(figure):
    fig, ax = figure
    picker = polysel.PolygonPicker(fig, ax)
    fig.canvas.callbacks.process("button_press_event",
                                 MouseEvent("button_press_event", fig.canvas, 1, 1, button=1))
    assert picker.xy.shape == (0,)
