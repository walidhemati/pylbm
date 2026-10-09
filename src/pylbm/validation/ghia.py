"""Lid-driven cavity vs. Ghia, Ghia & Shin (1982), Tables I and II."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..problem import Problem
from . import make_solver

_DATA = Path(__file__).parent / "data" / "ghia1982.json"

#: Table II, Re=400, x=0.9063 is a widely reported misprint (-0.23827 between -0.22847 and
#: -0.44993); it is excluded from the error norms (see data/ghia1982.json "notes").
EXCLUDE = {("v", "400", 0.9063)}


def ghia_table() -> dict:
    """The tabulated Ghia et al. (1982) centreline velocities."""
    return json.loads(_DATA.read_text())


def reference(re: int) -> dict[str, np.ndarray]:
    """Arrays y, u(y) on x=1/2 and x, v(x) on y=1/2 for ``re`` in {100, 400, 1000}."""
    t = ghia_table()
    key = str(int(re))
    return {
        "y": np.array(t["u_vertical"]["y"]),
        "u": np.array(t["u_vertical"][key]),
        "x": np.array(t["v_horizontal"]["x"]),
        "v": np.array(t["v_horizontal"][key]),
    }


def centerlines(ux: np.ndarray, uy: np.ndarray, u_lid: float) -> dict[str, np.ndarray]:
    """u along the vertical centreline and v along the horizontal one (normalised).

    Fields are (ny, nx); node centres sit at (i + 1/2)/n because the walls are half-way.
    The profiles are extended with the wall values so they can be interpolated anywhere.
    """
    ny, nx = ux.shape

    def mid(a: np.ndarray, axis: int) -> np.ndarray:
        n = a.shape[axis]
        if n % 2:
            return np.take(a, n // 2, axis=axis)
        return 0.5 * (np.take(a, n // 2 - 1, axis=axis) + np.take(a, n // 2, axis=axis))

    yc = (np.arange(ny) + 0.5) / ny
    xc = (np.arange(nx) + 0.5) / nx
    u = mid(ux, 1) / u_lid
    v = mid(uy, 0) / u_lid
    return {
        "y": np.concatenate([[0.0], yc, [1.0]]),
        "u": np.concatenate([[0.0], u, [1.0]]),
        "x": np.concatenate([[0.0], xc, [1.0]]),
        "v": np.concatenate([[0.0], v, [0.0]]),
    }


def errors(prof: dict[str, np.ndarray], re: int) -> dict[str, float]:
    """L2 (RMS) and Linf errors at the interior Ghia points, in units of the lid speed."""
    ref = reference(re)
    out: dict[str, float] = {}
    for comp, coord in (("u", "y"), ("v", "x")):
        pts, vals = ref[coord], ref[comp]
        keep = (pts > 0) & (pts < 1)
        keep &= np.array([(comp, str(re), float(p)) not in EXCLUDE for p in pts])
        sim = np.interp(pts[keep], prof[coord], prof[comp])
        d = sim - vals[keep]
        out[f"{comp}_L2"] = float(np.sqrt(np.mean(d**2)))
        out[f"{comp}_Linf"] = float(np.max(np.abs(d)))
    out["L2"] = float(np.sqrt(0.5 * (out["u_L2"] ** 2 + out["v_L2"] ** 2)))
    out["Linf"] = max(out["u_Linf"], out["v_Linf"])
    return out


@dataclass
class CavityResult:
    re: int
    n: int
    backend: str
    collision: str
    steps: int
    converged: bool | str
    errors: dict[str, float]
    profile: dict[str, list[float]] = field(repr=False)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def run_cavity(re: int, n: int = 128, u_lid: float = 0.1, collision: str = "bgk",
               backend: str | None = None, variant: str | None = None,
               tol: float = 1e-5, max_steps: int = 1_000_000, check_every: int | None = None,
               ) -> CavityResult:
    """Run the cavity to steady state and compare with Ghia et al. (1982).

    Convergence: the RMS change of u between two checks ``check_every`` steps apart, divided
    by the lid speed, drops below ``tol`` (the fp32 round-off floor is ~2e-6).
    """
    p = Problem.cavity(n, re=re, u_lid=u_lid, collision=collision)
    s = make_solver(p, backend, variant)
    check = check_every or max(1000, 4 * n)
    check -= check % s.step_multiple
    _, u_old = s.macroscopic()
    converged: bool | str = False
    best, stale = np.inf, 0
    while s.steps_done < max_steps:
        s.step(check)
        _, u = s.macroscopic()
        if not np.isfinite(u).all():
            break
        change = float(np.sqrt(np.mean((u - u_old) ** 2))) / u_lid
        u_old = u
        if change < tol:
            converged = True
            break
        # 16-bit storage has a higher round-off floor: stop once the change stagnates
        if change < 0.9 * best:
            best, stale = change, 0
        else:
            stale += 1
        if stale >= 10 and best < 50 * tol:
            converged = "plateau"
            break
    prof = centerlines(u_old[0], u_old[1], u_lid)
    return CavityResult(re, n, s.name, collision, s.steps_done, converged, errors(prof, re),
                        {k: v.tolist() for k, v in prof.items()})
