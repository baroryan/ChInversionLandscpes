"""End-to-end check on the shipped Santa Ynez example. Needs the full environment (dagger, scabbard).

Skipped automatically when dagger is not installed.
"""
import warnings

import numpy as np
import pytest

from conftest import REAL_DAGGER, REPO_ROOT

pytestmark = pytest.mark.skipif(not REAL_DAGGER, reason="dagger/scabbard not installed")

DEM_PATH = REPO_ROOT / "examples" / "DEMs" / "santa_ynez.tif"
SHIPPED_MODEL = REPO_ROOT / "examples" / "frdModels" / "sy_centralBasins.npz"
# settings and basin selection from examples/chi_space_inversion.ipynb
LOCAL_IDS = [18, 19, 14, 17, 23, 15, 29, 20, 24, 27, 35, 32, 9, 12, 13, 7, 8, 39, 11, 31, 16, 22, 21]


@pytest.fixture(scope="module")
def dem():
    import loadAndExportDem as load_dem
    return load_dem.loadDEMDiet(str(DEM_PATH), Z0=30.0, A0=1e6)


def test_dem_has_rivers(dem):
    assert len(dem.riverData) > 0
    assert len(dem.basinMapping) > max(LOCAL_IDS)


def test_export_by_basins_and_load(dem, tmp_path):
    import chifrd

    out = tmp_path / "central.npz"
    dem.ExportByBasins(str(out), basinIDs=LOCAL_IDS)
    f = chifrd.LoadFowardModel(str(out), kx=1, ky=1)
    assert f is not None


def test_export_basin_by_polygon_and_load(dem, tmp_path):
    import chifrd

    selected = dem.LocalBasinIDToBasinID(LOCAL_IDS)
    r = dem.riverData[dem.riverData.basinID.isin(selected)]
    poly = np.array([[r.X.min(), r.Y.min()], [r.X.max(), r.Y.min()],
                     [r.X.max(), r.Y.max()], [r.X.min(), r.Y.max()]])

    out = tmp_path / "poly.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ids, mask = dem.ExportBasinByPolygon(str(out), polygons=poly)

    assert set(selected) <= set(ids)
    saved = np.load(out)
    flat = dem.basinID.ravel()
    assert np.array_equal(np.unique(flat[saved["pixelMask"]]), np.sort(ids))
    assert np.array_equal(mask, saved["pixelMask"])

    f = chifrd.LoadFowardModel(str(out), kx=1, ky=1)
    assert f is not None


def test_shipped_forward_model_loads():
    import chifrd

    f = chifrd.LoadFowardModel(str(SHIPPED_MODEL), kx=1, ky=1)
    assert f is not None


# Verified run (Bar, 2026-11-06): examples/frdModels/sy_centralBasins.npz, kx=1, ky=1,
# chifrd.InvertLandScape(f_2d, std_m_n=0.05). Values are the printed "total misfit" after
# iterations 1..23, i.e. 100 * misfit / initial misfit; the run stopped at i = 24.
VERIFIED_MISFIT_PERCENT = [
    42.65, 38.76, 35.03, 27.28, 25.58, 22.50, 20.07, 18.07, 16.41, 15.06, 13.89, 12.93,
    12.14, 11.49, 10.96, 10.54, 10.19, 9.91, 9.69, 9.51, 9.36, 9.25, 9.15,
]
# printed values are rounded to 2 decimals; allow a little extra for platform differences
MISFIT_TOLERANCE = 0.05


def test_santa_ynez_inversion_matches_verified_run():
    import chifrd

    f_2d = chifrd.LoadFowardModel(str(SHIPPED_MODEL), kx=1, ky=1)
    inversion = chifrd.InvertLandScape(f_2d, std_m_n=0.05)

    misfit = np.asarray(inversion.misfitForStep)
    misfit_percent = 100 * misfit[1:] / misfit[0]

    assert len(inversion.step) == len(VERIFIED_MISFIT_PERCENT) + 1, (
        f"stopped after {len(inversion.step) - 1} iterations, verified run took {len(VERIFIED_MISFIT_PERCENT)}"
    )
    np.testing.assert_allclose(misfit_percent, VERIFIED_MISFIT_PERCENT, atol=MISFIT_TOLERANCE)
