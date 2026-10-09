"""NumPy reference: conservation, symmetry, and the CPU Ghia check run in CI."""

import numpy as np

from pylbm.backends.numpy_backend import NumpySolver
from pylbm.problem import Problem
from pylbm.validation.ghia import run_cavity


def test_cavity_conserves_mass():
    p = Problem.cavity((24, 20), re=50, u_lid=0.1)
    s = NumpySolver(p)
    m0 = s.populations().sum()
    s.step(300)
    assert abs(s.populations().sum() - m0) < 1e-10 * m0


def test_closed_box_at_rest_stays_at_rest():
    p = Problem((10, 8), tau=0.9)
    s = NumpySolver(p)
    s.step(50)
    assert np.allclose(s.populations(), p.initial_populations(), atol=1e-15)


def test_channel_is_symmetric():
    p = Problem((6, 17), tau=0.9, periodic=(True, False), force=(1e-5, 0.0))
    s = NumpySolver(p)
    s.step(400)
    _, u = s.macroscopic()
    assert np.allclose(u[0], u[0][::-1], atol=1e-14)
    assert np.abs(u[1]).max() < 1e-14


def test_ghia_re100_cpu():
    """Coarse 48x48 cavity in float64: within a few % of Ghia et al. (1982)."""
    r = run_cavity(100, 48, backend="numpy", variant="roll", tol=1e-6)
    assert r.converged
    assert r.errors["Linf"] < 0.03
