"""Array-level LBM kernels (stream + collide) written once for NumPy and PyTorch.

This is the readable, unfused formulation of the algorithm. With NumPy arrays in float64
it is the correctness oracle (:class:`pylbm.reference.NumpySolver`); with torch tensors it
forms the "torch-eager" and "torch-compile" rungs of the optimization ladder.

Algorithm per time step (pull scheme, populations stored post-collision):

1. stream:  f_i(x) = f*_i(x - c_i)                      (periodic wrap by ``roll``)
2. walls:   f_i(x) = f*_opp(i)(x) + 6 w_i rho_w c_i.u_w  if x - c_i is behind a wall
3. collide: f*_i = f_i + Omega_i(f) + forcing           (BGK, TRT or MRT; Guo forcing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lattice import MRT_M_D2Q9, MRT_NORM_D2Q9
from .problem import Problem


class Ops:
    """Tiny adapter so the same code runs on numpy arrays and torch tensors."""

    def __init__(self, xp: Any) -> None:
        self.xp = xp
        self.is_torch = xp.__name__ == "torch"

    def roll(self, a: Any, shifts: tuple[int, ...], axes: tuple[int, ...]) -> Any:
        if self.is_torch:
            return self.xp.roll(a, shifts=shifts, dims=axes)
        return np.roll(a, shifts, axis=axes)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return self.xp.where(cond, a, b)

    def stack(self, arrs: list[Any]) -> Any:
        return self.xp.stack(arrs)


@dataclass
class Geometry:
    """Pre-computed boundary data for :func:`stream` (arrays live on the target device)."""

    problem: Problem
    solid_src: list[Any] | None  # per direction: bool array, source node is solid
    inlet_corr: list[Any] | None  # per direction: correction on the inlet face (or None)


def make_geometry(p: Problem, to_array: Any) -> Geometry:
    """Build :class:`Geometry`; ``to_array`` converts numpy arrays to the target array type."""
    lat = p.lattice
    solid_src = None
    if p.solid is not None:
        solid = np.asarray(p.solid, dtype=bool)
        axes = tuple(range(p.dim))
        solid_src = []
        for i in range(lat.q):
            shifts = tuple(int(lat.c[i, a]) for a in reversed(range(p.dim)))
            solid_src.append(to_array(np.roll(solid, shifts, axis=axes)))
    inlet_corr = None
    if p.inlet is not None:
        if p.dim != 2:
            raise ValueError("inlet only implemented in 2D")
        uin = np.asarray(p.inlet, dtype=np.float64)
        inlet_corr = []
        for i in range(lat.q):
            if lat.c[i, 0] == 1:
                inlet_corr.append(to_array(6.0 * lat.w[i] * lat.c[i, 0] * uin))
            else:
                inlet_corr.append(None)
    return Geometry(p, solid_src, inlet_corr)


def _face(dim: int, axis: int, side: int) -> tuple:
    """Index tuple selecting the low (side=0) or high (side=-1) face of spatial ``axis``."""
    idx: list[Any] = [slice(None)] * dim
    idx[dim - 1 - axis] = side
    return tuple(idx)


def stream(f: Any, geo: Geometry, ops: Ops) -> list[Any]:
    """Pull streaming with half-way bounce-back; returns a list of q arrays."""
    p = geo.problem
    lat = p.lattice
    d = p.dim
    axes = tuple(range(d))
    out = []
    for i in range(lat.q):
        shifts = tuple(int(lat.c[i, a]) for a in reversed(range(d)))
        if any(shifts):
            out.append(ops.roll(f[i], shifts, axes))
        else:
            out.append(f[i].clone() if ops.is_torch else f[i].copy())
    # solid obstacles (stationary)
    if geo.solid_src is not None:
        for i in range(lat.q):
            if lat.c[i].any():
                out[i] = ops.where(geo.solid_src[i], f[int(lat.opp[i])], out[i])
    # box walls; later axes override earlier ones at edges/corners (lid precedence)
    for a in range(d):
        if p.periodic[a]:
            continue
        top = a == d - 1
        for i in range(lat.q):
            ca = lat.c[i, a]
            if ca == 1:  # source is behind the low wall
                fc = _face(d, a, 0)
                val = f[int(lat.opp[i])][fc]
                if a == 0 and geo.inlet_corr is not None:
                    val = val + geo.inlet_corr[i]
                out[i][fc] = val
            elif ca == -1:  # source is behind the high wall
                fc = _face(d, a, -1)
                val = f[int(lat.opp[i])][fc]
                if top and p.u_lid != 0.0 and lat.c[i, 0] != 0:
                    val = val + 6.0 * float(lat.w[i]) * int(lat.c[i, 0]) * p.u_lid
                out[i][fc] = val
    if p.outlet:
        for i in range(lat.q):
            if lat.c[i, 0] == -1:
                out[i][..., -1] = out[i][..., -2]
    return out


def moments(f: list[Any] | Any, p: Problem) -> tuple[Any, list[Any]]:
    """Density and velocity u = (sum c f + F/2) / rho (Guo) from populations."""
    lat = p.lattice
    rho = f[0]
    for i in range(1, lat.q):
        rho = rho + f[i]
    u = []
    for a in range(p.dim):
        j = None
        for i in range(lat.q):
            ci = lat.c[i, a]
            if ci == 0:
                continue
            term = f[i] if ci == 1 else -f[i]
            j = term if j is None else j + term
        if p.force[a] != 0.0:
            j = j + 0.5 * p.force[a]
        u.append(j / rho)
    return rho, u


def _equilibria(rho: Any, u: list[Any], p: Problem) -> tuple[list[Any], list[Any]]:
    """Return (feq_i, c_i.u) lists."""
    lat = p.lattice
    usq = u[0] * u[0]
    for a in range(1, p.dim):
        usq = usq + u[a] * u[a]
    feq, cus = [], []
    for i in range(lat.q):
        cu = 0.0
        for a in range(p.dim):
            if lat.c[i, a] != 0:
                cu = cu + int(lat.c[i, a]) * u[a]
        cus.append(cu)
        feq.append(float(lat.w[i]) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq))
    return feq, cus


def _guo_source(u: list[Any], cus: list[Any], p: Problem) -> list[Any]:
    """Guo et al. (2002) source S_i = w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)] (no prefactor)."""
    lat = p.lattice
    uf = 0.0
    for a in range(p.dim):
        if p.force[a] != 0.0:
            uf = uf + u[a] * p.force[a]
    out = []
    for i in range(lat.q):
        cf = float(sum(lat.c[i, a] * p.force[a] for a in range(p.dim)))
        out.append(float(lat.w[i]) * (3.0 * (cf - uf) + 9.0 * cus[i] * cf))
    return out


def collide(f: list[Any], p: Problem, ops: Ops) -> Any:
    """Collision with optional Guo forcing; returns the stacked post-collision array."""
    lat = p.lattice
    rho, u = moments(f, p)
    feq, cus = _equilibria(rho, u, p)
    src = _guo_source(u, cus, p) if p.has_force else None
    out: list[Any] = []
    if p.collision == "bgk":
        om = 1.0 / p.tau
        for i in range(lat.q):
            v = f[i] - om * (f[i] - feq[i])
            if src is not None:
                v = v + (1.0 - 0.5 * om) * src[i]
            out.append(v)
    elif p.collision == "trt":
        sp, sm = p.trt_rates()
        for i in range(lat.q):
            o = int(lat.opp[i])
            fp, fm = 0.5 * (f[i] + f[o]), 0.5 * (f[i] - f[o])
            ep, em = 0.5 * (feq[i] + feq[o]), 0.5 * (feq[i] - feq[o])
            v = f[i] - sp * (fp - ep) - sm * (fm - em)
            if src is not None:
                gp, gm = 0.5 * (src[i] + src[o]), 0.5 * (src[i] - src[o])
                v = v + (1.0 - 0.5 * sp) * gp + (1.0 - 0.5 * sm) * gm
            out.append(v)
    else:  # mrt, D2Q9: m-space update  dm_k = -s_k M(f - feq) + (1 - s_k/2) M S
        s = [float(v) for v in p.mrt_rates()]
        M, nrm = MRT_M_D2Q9.tolist(), MRT_NORM_D2Q9.tolist()
        dm = []
        for k in range(lat.q):
            acc = 0.0
            for i in range(lat.q):
                if M[k][i] != 0.0:
                    term = -s[k] * M[k][i] * (f[i] - feq[i])
                    if src is not None:
                        term = term + (1.0 - 0.5 * s[k]) * M[k][i] * src[i]
                    acc = acc + term
            dm.append(acc)
        for i in range(lat.q):
            v = f[i]
            for k in range(lat.q):
                if M[k][i] != 0.0:
                    v = v + (M[k][i] / nrm[k]) * dm[k]
            out.append(v)
    return ops.stack(out)


def step(f: Any, geo: Geometry, ops: Ops) -> Any:
    """One full time step on the post-collision state ``f``."""
    return collide(stream(f, geo, ops), geo.problem, ops)


def macroscopic(f: Any, p: Problem) -> tuple[Any, list[Any]]:
    """(rho, [u_x, u_y(, u_z)]) from the stored post-collision populations.

    For a post-collision state the momentum already contains the full force F, so the
    physical velocity is (sum c f* - F/2) / rho.
    """
    lat = p.lattice
    fl = [f[i] for i in range(lat.q)]
    rho, u = moments(fl, p)
    if p.has_force:
        u = [(u[a] * rho - p.force[a]) / rho for a in range(p.dim)]
    return rho, u
