"""Shear-wave decay: measured vs. theoretical kinematic viscosity nu = (tau - 1/2)/3."""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from . import make_solver


def run_shear_wave(tau: float, n: int = 64, amplitude: float = 0.01, steps: int | None = None,
                   samples: int = 20, collision: str = "bgk", backend: str | None = None,
                   variant: str | None = None) -> dict:
    """Decay of u_x = A sin(2 pi y / n) in a periodic n x n box.

    The amplitude a(t) (projection of the x-averaged u_x on the sine mode) decays as
    exp(-nu k^2 t); a least-squares fit of log a(t) gives the measured viscosity.
    """
    k = 2 * np.pi / n
    y = np.arange(n)
    u0 = np.zeros((2, n, n))
    u0[0] = amplitude * np.sin(k * y)[:, None]
    p = Problem((n, n), tau=tau, collision=collision, periodic=(True, True), init_u=u0)
    nu_th = p.nu
    if steps is None:  # decay to ~exp(-2)
        steps = int(2.0 / (nu_th * k * k))
    s = make_solver(p, backend, variant)
    chunk = max(s.step_multiple, steps // samples)
    chunk -= chunk % s.step_multiple
    ts, amps = [], []
    mode = np.sin(k * y)
    for _ in range(samples + 1):
        _, u = s.macroscopic()
        prof = u[0].mean(axis=1)
        ts.append(s.steps_done)
        amps.append(2.0 / n * float(prof @ mode))
        s.step(chunk)
    ts_a, amps_a = np.array(ts, float), np.array(amps)
    slope = np.polyfit(ts_a, np.log(np.abs(amps_a)), 1)[0]
    nu_meas = -slope / (k * k)
    return {"tau": tau, "n": n, "nu_theory": nu_th, "nu_measured": float(nu_meas),
            "rel_error": float(abs(nu_meas - nu_th) / nu_th), "backend": s.name,
            "t": ts_a.tolist(), "amplitude": amps_a.tolist()}
