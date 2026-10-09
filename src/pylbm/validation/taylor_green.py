"""3D check (D3Q19): decay of Taylor-Green vortices in a periodic box.

A 2D Taylor-Green vortex u = U (-cos kx sin ky, sin kx cos ky) is an exact Navier-Stokes
solution whose kinetic energy decays as exp(-4 nu k^2 t). Embedding it in each of the three
coordinate planes of a periodic n^3 box exercises all 18 moving D3Q19 directions.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from . import make_solver

PLANES = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}


def tgv_problem(n: int, tau: float, amplitude: float, plane: str) -> Problem:
    a, b = PLANES[plane]
    k = 2 * np.pi / n
    grids = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")  # (z, y, x)
    coord = {0: grids[2], 1: grids[1], 2: grids[0]}  # spatial axis -> coordinate array
    u = np.zeros((3, n, n, n))
    u[a] = -amplitude * np.cos(k * coord[a]) * np.sin(k * coord[b])
    u[b] = amplitude * np.sin(k * coord[a]) * np.cos(k * coord[b])
    p0 = -0.25 * amplitude**2 * (np.cos(2 * k * coord[a]) + np.cos(2 * k * coord[b]))
    return Problem((n, n, n), tau=tau, periodic=(True, True, True), init_u=u,
                   init_rho=1.0 + 3.0 * p0)


def run_taylor_green(n: int = 32, tau: float = 0.8, amplitude: float = 0.02,
                     plane: str = "xy", samples: int = 10, backend: str | None = None,
                     variant: str | None = None) -> dict:
    """Fit the energy decay rate and return the measured viscosity."""
    p = tgv_problem(n, tau, amplitude, plane)
    k = 2 * np.pi / n
    steps = int(1.0 / (4 * p.nu * k * k))  # one e-fold of the energy
    s = make_solver(p, backend, variant)
    chunk = max(s.step_multiple, steps // samples)
    chunk -= chunk % s.step_multiple
    ts, es = [], []
    for _ in range(samples + 1):
        rho, u = s.macroscopic()
        ts.append(s.steps_done)
        es.append(float(0.5 * (rho * (u**2).sum(axis=0)).mean()))
        s.step(chunk)
    slope = np.polyfit(np.array(ts, float), np.log(es), 1)[0]
    nu_meas = -slope / (4 * k * k)
    return {"n": n, "tau": tau, "plane": plane, "backend": s.name, "nu_theory": p.nu,
            "nu_measured": float(nu_meas), "rel_error": float(abs(nu_meas - p.nu) / p.nu)}
