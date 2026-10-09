import numpy as np
import pytest

from pylbm.core import Ops, collide
from pylbm.lattice import D2Q9, D3Q19, MRT_M_D2Q9, MRT_NORM_D2Q9, equilibrium
from pylbm.problem import Problem


@pytest.mark.parametrize("lat", [D2Q9, D3Q19])
def test_isotropy(lat):
    c, w = lat.c.astype(float), lat.w
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose((w[:, None] * c).sum(0), 0.0)
    second = np.einsum("i,ia,ib->ab", w, c, c)
    assert np.allclose(second, np.eye(lat.d) / 3.0)
    assert all((c[lat.opp] == -c).all(axis=1))
    assert lat.ct == tuple(tuple(r) for r in lat.c.tolist())


def test_mrt_basis_orthogonal():
    gram = MRT_M_D2Q9 @ MRT_M_D2Q9.T
    assert np.allclose(gram, np.diag(MRT_NORM_D2Q9))


@pytest.mark.parametrize("lat", [D2Q9, D3Q19])
def test_equilibrium_moments(lat):
    rng = np.random.default_rng(0)
    rho = 1 + 0.01 * rng.standard_normal((4, 5))
    u = 0.05 * rng.standard_normal((lat.d, 4, 5))
    feq = equilibrium(lat, rho, u)
    assert np.allclose(feq.sum(0), rho)
    j = np.einsum("ia,i...->a...", lat.c.astype(float), feq)
    assert np.allclose(j, rho * u)


@pytest.mark.parametrize("force", [None, (1e-4, -2e-4)])
def test_mrt_with_equal_rates_is_bgk(force):
    """MRT reduces to BGK when every non-conserved rate equals 1/tau."""
    tau = 0.8
    rng = np.random.default_rng(1)
    f = D2Q9.w[:, None, None] * (1 + 0.05 * rng.standard_normal((9, 6, 7)))
    pb = Problem((7, 6), tau=tau, force=force)
    pm = Problem((7, 6), tau=tau, force=force, collision="mrt", s_e=1 / tau, s_eps=1 / tau)
    # force the energy-flux rate to 1/tau as well
    pm.mrt_rate_tuple = lambda: (0.0,) + (1 / tau,) * 2 + (0.0, 1 / tau, 0.0, 1 / tau) + (
        1 / tau,) * 2
    ops = Ops(np)
    a = collide(list(f), pb, ops)
    b = collide(list(f), pm, ops)
    assert np.allclose(a, b, atol=1e-14)


def test_collision_conserves_mass_and_momentum():
    rng = np.random.default_rng(2)
    f = D2Q9.w[:, None, None] * (1 + 0.05 * rng.standard_normal((9, 6, 7)))
    for coll in ("bgk", "trt", "mrt"):
        p = Problem((7, 6), tau=0.7, collision=coll)
        g = collide(list(f), p, Ops(np))
        assert np.allclose(g.sum(0), f.sum(0))
        j0 = np.einsum("ia,iyx->ayx", D2Q9.c.astype(float), f)
        j1 = np.einsum("ia,iyx->ayx", D2Q9.c.astype(float), g)
        assert np.allclose(j0, j1)
