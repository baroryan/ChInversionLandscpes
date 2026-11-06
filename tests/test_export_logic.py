"""Fast tests of basin selection/export logic on a synthetic DEM (no dagger needed)."""
import warnings

import numpy as np
import pytest
from matplotlib.path import Path as MplPath


def _basins_in(mask, D):
    return np.unique(D.basinID.ravel()[mask])


def test_flat_and_array_coordinates_agree(synthetic_dem):
    D = synthetic_dem
    xa, ya = D.GetArrayCorr()
    xf, yf = D.GetFlatCorr()
    assert np.array_equal(xa.ravel(), xf)
    assert np.array_equal(ya.ravel(), yf)


def test_export_by_local_ids(synthetic_dem, tmp_path):
    D = synthetic_dem
    local_ids = [0, 2, 5]
    expected = np.sort(D.LocalBasinIDToBasinID(local_ids))

    out = tmp_path / "local.npz"
    D.ExportByBasins(str(out), basinIDs=local_ids)
    saved = np.load(out)

    assert np.array_equal(_basins_in(saved["pixelMask"], D), expected)
    assert np.array_equal(_basins_in(saved["riverMask"], D), expected)


def test_export_by_global_ids(synthetic_dem, tmp_path):
    D = synthetic_dem
    global_ids = np.sort(D.basinMapping["basinID"].to_numpy()[[1, 3, 4]])

    out = tmp_path / "global.npz"
    D.ExportByBasins(str(out), basinIDs=global_ids, basinIDType="global")
    saved = np.load(out)

    assert np.array_equal(_basins_in(saved["pixelMask"], D), global_ids)
    assert np.array_equal(_basins_in(saved["riverMask"], D), global_ids)


def test_global_ids_are_not_accepted_as_local(synthetic_dem, tmp_path):
    D = synthetic_dem
    global_ids = D.basinMapping["basinID"].to_numpy()[:3]  # large numbers, not valid local IDs
    with pytest.raises(ValueError):
        D.ExportByBasins(str(tmp_path / "x.npz"), basinIDs=global_ids)


def test_bad_basin_id_type_raises(synthetic_dem, tmp_path):
    with pytest.raises(ValueError):
        synthetic_dem.ExportByBasins(str(tmp_path / "x.npz"), basinIDs=[0], basinIDType="localish")


def test_export_basin_by_polygon(synthetic_dem, tmp_path):
    D = synthetic_dem
    xf, yf = D.GetFlatCorr()
    x0, x1 = np.quantile(xf, [0.2, 0.75])
    y0, y1 = np.quantile(yf, [0.15, 0.8])
    poly = np.array([[x0, y0], [x1, y0 + 50], [x1 - 100, y1], [x0 + 50, y1]])

    inside = MplPath(poly).contains_points(np.column_stack((xf, yf)))
    expected = np.intersect1d(_basins_in(inside, D), D.riverData.basinID.unique())
    assert len(expected) > 1  # sanity: polygon covers several basins

    out = tmp_path / "poly.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # 'missing basinIDs' warning would fail
        ids, mask = D.ExportBasinByPolygon(str(out), polygons=poly)
    saved = np.load(out)

    assert np.array_equal(np.sort(ids), expected)
    assert np.array_equal(_basins_in(saved["pixelMask"], D), expected)
    assert np.array_equal(_basins_in(saved["riverMask"], D), expected)
    assert np.array_equal(mask, saved["pixelMask"])


def test_export_basin_by_several_polygons(synthetic_dem, tmp_path):
    D = synthetic_dem
    xf, yf = D.GetFlatCorr()
    xm, ym = xf.min(), yf.min()
    p1 = np.array([[xm + 10, ym + 10], [xm + 200, ym + 10], [xm + 200, ym + 200]])
    p2 = np.array([[xf.max() - 200, yf.max() - 200], [xf.max() - 10, yf.max() - 200], [xf.max() - 10, yf.max() - 10]])

    ids, _ = D.ExportBasinByPolygon(str(tmp_path / "two.npz"), polygons=[p1, p2])
    flat = D.basinID.ravel()
    pts = np.column_stack((xf, yf))
    expected = np.intersect1d(
        np.unique(np.concatenate([flat[MplPath(p).contains_points(pts)] for p in (p1, p2)])),
        D.riverData.basinID.unique(),
    )
    assert np.array_equal(np.sort(ids), expected)


@pytest.mark.parametrize("all_perimeters", [True, False])
def test_plot_all_runs(synthetic_dem, all_perimeters):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    synthetic_dem.PlotAll(ax=ax, numOfBasinsPrint=3, plotAllPerimeters=all_perimeters)
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["0", "1", "2"]
    plt.close(fig)
