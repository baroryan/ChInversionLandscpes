"""Shared test setup.

- Puts the repository root on sys.path.
- If dagger / scabbard / numba are not installed (the fast CI job), replaces them with
  small stand-ins so the pure-Python logic can still be tested. The end-to-end tests
  check `REAL_DAGGER` and skip themselves when the stand-ins are in use.
- Provides `synthetic_dem`: a loadDEMDiet object built without dagger, with large
  dagger-style global basin IDs so local/global ID mix-ups are caught.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__stub__ = True
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


try:
    import dagger  # noqa: F401
    import scabbard  # noqa: F401
    REAL_DAGGER = True
except ImportError:
    REAL_DAGGER = False
    _stub("dagger")
    _stub("scabbard")

try:
    import numba  # noqa: F401
except ImportError:
    _identity = lambda *a, **k: (lambda f: f)  # noqa: E731
    _stub("numba", jit=_identity, njit=_identity, prange=range)


import loadAndExportDem as load_dem  # noqa: E402


def _build_synthetic_dem(ny=60, nx=90, dx=30.0, n_rows=4, n_cols=6, n_without_rivers=3, seed=0):
    """Grid of rectangular basins; the last `n_without_rivers` basins have no river nodes."""
    rng = np.random.default_rng(seed)
    X = np.arange(nx) * dx + 5.0e5
    Y = np.arange(ny) * dx + 4.0e6

    n_basins = n_rows * n_cols
    global_ids = rng.choice(np.arange(1000, 50000), size=n_basins, replace=False)
    basinID = np.zeros((ny, nx), dtype=np.int64)
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            basinID[i * ny // n_rows:(i + 1) * ny // n_rows, j * nx // n_cols:(j + 1) * nx // n_cols] = global_ids[k]
            k += 1

    D = load_dem.loadDEMDiet.__new__(load_dem.loadDEMDiet)
    N = nx * ny
    D.ftype = np.float64
    D.demFilename = "synthetic"
    D.Z0 = 0.0
    D.A0 = 1.0
    D.dem = types.SimpleNamespace(X=X, Y=Y, nx=nx, ny=ny, extent=lambda: (X[0], X[-1], Y[0], Y[-1]))
    D.shape = (ny, nx)
    D.A = np.ones(N)
    D.dX = np.ones(N)
    D.Z = rng.random(N) * 1000
    D.recs = np.arange(N)
    D.stack = np.arange(N)
    D.basinID = basinID

    xf, yf = D.GetFlatCorr()
    flat = basinID.ravel()
    nodes = np.concatenate([
        rng.choice(np.flatnonzero(flat == g), size=int(rng.integers(3, 30)), replace=False)
        for g in global_ids[:n_basins - n_without_rivers]
    ])
    rivers = pd.DataFrame({"nodes": nodes, "basinID": flat[nodes], "X": xf[nodes], "Y": yf[nodes]})
    D.riverData, D.basinMapping = D.AddLocalBasinID(rivers)
    D.continentalDivide = pd.DataFrame({"X": xf[::5], "Y": yf[::5], "basinID": flat[::5]})
    return D


@pytest.fixture
def synthetic_dem():
    return _build_synthetic_dem()
