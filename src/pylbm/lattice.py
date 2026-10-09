"""Velocity sets (D2Q9, D3Q19) and moment matrices.

Conventions used everywhere in pylbm:

* Arrays are stored structure-of-arrays, ``f[i, y, x]`` in 2D and ``f[i, z, y, x]`` in 3D,
  with x the fastest (column) index.
* x points right (increasing column), y points up (increasing row), z points out of the
  (x, y) plane. Row 0 is the bottom wall, so plot with ``origin="lower"``.
* The moving lid is the top wall (y = ny - 1/2 in 2D, z = nz - 1/2 in 3D) and moves in +x.
* Walls sit half-way between the last fluid node and the (virtual) solid node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CS2 = 1.0 / 3.0  # lattice speed of sound squared


@dataclass(frozen=True)
class Lattice:
    """A DdQq velocity set."""

    name: str
    d: int
    c: np.ndarray  # (q, d) integer velocities
    w: np.ndarray  # (q,) weights
    opp: np.ndarray  # (q,) index of the opposite velocity
    # plain-Python copies (tuples) for code traced by torch.compile
    ct: tuple = ()
    wt: tuple = ()
    oppt: tuple = ()

    @property
    def q(self) -> int:
        return len(self.w)


def _make(name: str, c: list[list[int]], w: list[float]) -> Lattice:
    ca = np.array(c, dtype=np.int64)
    opp = np.array([int(np.where((ca == -ci).all(axis=1))[0][0]) for ci in ca])
    wa = np.array(w, dtype=np.float64)
    return Lattice(name, ca.shape[1], ca, wa, opp, tuple(tuple(int(v) for v in r) for r in ca),
                   tuple(float(v) for v in wa), tuple(int(v) for v in opp))


#: D2Q9, ordered 0 rest, 1 E, 2 N, 3 W, 4 S, 5 NE, 6 NW, 7 SW, 8 SE (Krueger et al. 2017).
D2Q9 = _make(
    "D2Q9",
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4,
)

#: D3Q19, opposite pairs adjacent: (1,2), (3,4), ..., (17,18).
D3Q19 = _make(
    "D3Q19",
    [
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
    ],
    [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12,
)

LATTICES = {"D2Q9": D2Q9, "D3Q19": D3Q19}

#: Lallemand & Luo (2000) orthogonal moment basis for D2Q9 in the velocity order above:
#: rows are (rho, e, eps, jx, qx, jy, qy, pxx, pxy).
MRT_M_D2Q9 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=np.float64,
)
MRT_NORM_D2Q9 = (MRT_M_D2Q9**2).sum(axis=1)  # M is orthogonal: M^-1 = M^T diag(1/norm)


def magic_rate(s_nu: float, magic: float = 3.0 / 16.0) -> float:
    """Odd-moment relaxation rate giving the TRT "magic" parameter Lambda.

    Lambda = (1/s_nu - 1/2)(1/s_q - 1/2). Lambda = 3/16 places the half-way bounce-back wall
    exactly mid-link for Poiseuille flow (Ginzburg et al. 2008).
    """
    tau_q = 0.5 + magic / (1.0 / s_nu - 0.5)
    return 1.0 / tau_q


def mrt_rates(tau: float, s_e: float = 1.64, s_eps: float = 1.54) -> np.ndarray:
    """Diagonal MRT relaxation rates for the D2Q9 basis above.

    Shear rates are 1/tau, the energy-flux rates s_q follow the magic relation (Lambda=3/16),
    and the bulk/energy rates are free parameters (defaults 1.64 and 1.54). Conserved moments
    get rate 0 (they are unchanged by collision; the Guo force term still acts on them).
    """
    s_nu = 1.0 / tau
    s_q = magic_rate(s_nu)
    return np.array([0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, s_nu, s_nu])


def equilibrium(lat: Lattice, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Second-order equilibrium f_i^eq(rho, u); ``u`` has shape (d, ...)."""
    usq = sum(u[a] ** 2 for a in range(lat.d))
    out = np.empty((lat.q,) + rho.shape, dtype=np.result_type(rho, u))
    for i in range(lat.q):
        cu = sum(lat.c[i, a] * u[a] for a in range(lat.d))
        out[i] = lat.w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return out
