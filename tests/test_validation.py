"""Physical validation on small grids (CPU) and on the GPU (marked)."""

import numpy as np
import pytest

from pylbm.validation import ghia, karman, poiseuille, shear_wave, taylor_green


@pytest.mark.parametrize("tau", [0.6, 1.0, 1.4])
def test_shear_wave_viscosity_cpu(tau):
    r = shear_wave.run_shear_wave(tau, n=32, backend="numpy", variant="roll")
    assert r["rel_error"] < 0.02


def test_poiseuille_bgk_second_order_and_trt_exact():
    c = poiseuille.convergence((8, 16), collision="bgk", backend="numpy", variant="roll")
    assert 1.9 < c["order"] < 2.1
    r = poiseuille.run_poiseuille(12, collision="trt", backend="numpy", variant="roll")
    assert r["L2"] < 1e-9


def test_tgv3d_viscosity_cpu():
    r = taylor_green.run_taylor_green(n=16, plane="yz", backend="numpy", variant="roll")
    assert r["rel_error"] < 0.03


def test_ghia_points_and_exclusion():
    ref = ghia.reference(400)
    assert len(ref["y"]) == len(ref["u"]) == 17
    assert ("v", "400", 0.9063) in ghia.EXCLUDE


@pytest.mark.gpu
@pytest.mark.parametrize("re,n,limit", [(100, 128, 0.006), (1000, 256, 0.01)])
def test_ghia_gpu(re, n, limit):
    r = ghia.run_cavity(re, n)
    assert r.converged
    assert r.errors["Linf"] < limit


@pytest.mark.gpu
def test_shear_wave_gpu():
    r = shear_wave.run_shear_wave(0.8, n=128)
    assert r["rel_error"] < 1e-3


@pytest.mark.gpu
def test_tgv3d_gpu():
    r = taylor_green.run_taylor_green(n=48, plane="xz")
    assert r["rel_error"] < 5e-3


@pytest.mark.gpu
def test_karman_strouhal_gpu():
    r = karman.run_karman(d=20, t_transient=40.0, periods=12.0)
    st_lo, st_hi = karman.REFERENCE["St"]
    assert 0.95 * st_lo < r["St"] < 1.05 * st_hi
    assert np.isfinite(r["cd_max"]) and 3.0 < r["cd_max"] < 3.6
