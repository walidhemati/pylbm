"""Body-force driven Poiseuille channel vs. the analytic parabola, with grid convergence."""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from . import make_solver


def analytic(h: int, force: float, nu: float) -> np.ndarray:
    """u_x at the node centres y_j = j + 1/2 for walls at y = 0 and y = h (rho = 1)."""
    y = np.arange(h) + 0.5
    return force / (2.0 * nu) * y * (h - y)


def run_poiseuille(h: int, tau: float = 0.8, u_max: float = 0.02, collision: str = "bgk",
                   nx: int = 4, backend: str | None = None, variant: str | None = None,
                   tol: float = 1e-9, max_steps: int = 2_000_000) -> dict:
    """Run to steady state; return relative L2 / Linf errors against the parabola."""
    nu = (tau - 0.5) / 3.0
    force = 8.0 * nu * u_max / h**2
    p = Problem((nx, h), tau=tau, collision=collision, periodic=(True, False),
                force=(force, 0.0))
    s = make_solver(p, backend, variant)
    ref = analytic(h, force, nu)
    check = max(s.step_multiple * 50, int(0.5 * h * h / nu))
    check += check % s.step_multiple
    old = np.zeros(h)
    best, stale = np.inf, 0
    while s.steps_done < max_steps:
        s.step(check)
        _, u = s.macroscopic()
        prof = u[0].mean(axis=1)
        change = float(np.abs(prof - old).max()) / u_max
        old = prof
        if change < tol:
            break
        if change < 0.5 * best:  # stop at the round-off floor
            best, stale = change, 0
        else:
            stale += 1
        if stale >= 5 and best < 1e-4:
            break
    err = prof - ref
    return {"h": h, "tau": tau, "collision": collision, "backend": s.name,
            "steps": s.steps_done,
            "L2": float(np.sqrt(np.sum(err**2) / np.sum(ref**2))),
            "Linf": float(np.abs(err).max() / ref.max()),
            "y": ((np.arange(h) + 0.5) / h).tolist(), "u": (prof / u_max).tolist()}


def convergence(hs: tuple[int, ...] = (8, 16, 32, 64), **kw: object) -> dict:
    """Errors for several channel widths plus the fitted order of convergence."""
    runs = [run_poiseuille(h, **kw) for h in hs]  # type: ignore[arg-type]
    errs = np.array([r["L2"] for r in runs])
    order = float(-np.polyfit(np.log(hs), np.log(errs), 1)[0]) if (errs > 0).all() else None
    return {"runs": runs, "order": order}
