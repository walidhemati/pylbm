"""Karman vortex street: Schaefer & Turek (1996) benchmark 2D-2 (Re = 100, periodic).

Channel 2.2 x 0.41, cylinder of diameter D = 0.1 centred at (0.2, 0.2), parabolic inflow
with U_max = 1.5 (mean velocity U = 1), Re = U D / nu = 100. Reference intervals (Table 4 of
the benchmark paper): c_D,max in [3.22, 3.24], c_L,max in [0.99, 1.01], St in [0.295, 0.305].

The cylinder is a staircase of solid nodes with half-way bounce-back, the inflow uses the
moving-wall (Ladd) bounce-back with the parabolic profile, and the outflow copies the
unknown populations from the neighbouring column. Forces use the momentum-exchange method.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem

REFERENCE = {"St": (0.2950, 0.3050), "cd_max": (3.2200, 3.2400), "cl_max": (0.9900, 1.0100)}


def schafer_turek_problem(d: int = 20, u_mean: float = 0.05, collision: str = "trt") -> Problem:
    """Lattice version of benchmark 2D-2 with ``d`` nodes per cylinder diameter."""
    dx = 0.1 / d
    nx, ny = round(2.2 / dx), round(0.41 / dx)
    h = ny  # walls at y = 0 and y = ny (half-way)
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(xc, yc)
    solid = (X - 0.2) ** 2 + (Y - 0.2) ** 2 <= 0.05**2
    u_max = 1.5 * u_mean
    yy = np.arange(ny) + 0.5
    inlet = 4.0 * u_max * yy * (h - yy) / h**2
    nu = u_mean * d / 100.0
    init_u = np.zeros((2, ny, nx))
    init_u[0] = inlet[:, None]
    init_u[0][solid] = 0.0
    # small transverse kick to trigger shedding quickly (the periodic state does not depend
    # on it)
    init_u[1] = 0.05 * u_mean * np.exp(-((X - 0.3) ** 2 + (Y - 0.2) ** 2) / 0.05**2)
    return Problem((nx, ny), tau=3.0 * nu + 0.5, collision=collision, periodic=(False, False),
                   solid=solid, inlet=inlet, outlet=True, init_u=init_u,
                   meta={"d": d, "u_mean": u_mean, "dx": dx})


def _force_masks(p: Problem) -> tuple[list[np.ndarray], np.ndarray]:
    """Per direction: fluid nodes whose neighbour x + c_i is solid."""
    lat = p.lattice
    fluid = ~p.solid
    masks = []
    for i in range(lat.q):
        nb_solid = np.roll(p.solid, (-int(lat.c[i, 1]), -int(lat.c[i, 0])), axis=(0, 1))
        masks.append(fluid & nb_solid)
    return masks, lat.c.astype(np.float64)


def run_karman(d: int = 20, u_mean: float = 0.05, periods: float = 20.0,
               t_transient: float = 60.0, collision: str = "trt", backend: str | None = None,
               variant: str | None = None, sample_every: int = 10) -> dict:
    """Run benchmark 2D-2 and measure St, max c_D, max c_L.

    ``t_transient`` and ``periods`` are in convective units D / U; the statistics use the
    last ``periods`` convective units... after the transient.
    """
    import torch

    from ..backends import get

    p = schafer_turek_problem(d, u_mean, collision)
    prefer = [(backend, variant or "gather")] if backend else [
        ("torch-compile", "gather"), ("torch-eager", "gather"), ("numpy", "roll")]
    s = None
    for b, v in prefer:
        cls = get(b, v)
        if cls.available()[0] and cls.supports(p)[0]:
            s = cls(p)
            break
    if s is None:
        raise RuntimeError("no backend for the Karman case")
    masks, c = _force_masks(p)
    t_conv = d / u_mean  # steps per convective unit
    n_trans = int(t_transient * t_conv)
    n_meas = int(periods * t_conv)
    is_torch = hasattr(s, "f") and isinstance(getattr(s, "f", None), torch.Tensor)
    if is_torch:
        mk = torch.stack([torch.as_tensor(m, device=s.f.device) for m in masks]).to(s.f.dtype)
        ct = torch.as_tensor(c, device=s.f.device, dtype=s.f.dtype)

        def force() -> np.ndarray:
            per_dir = (s.f * mk).sum(dim=(1, 2))  # sum_x f*_i(x) over boundary links
            return (2.0 * (per_dir[:, None] * ct).sum(0)).double().cpu().numpy()
    else:
        mk_np = np.stack(masks).astype(float)

        def force() -> np.ndarray:
            per_dir = (s.populations() * mk_np).sum(axis=(1, 2))
            return 2.0 * (per_dir[:, None] * c).sum(0)

    with torch.no_grad():
        s.step(n_trans)
        ts, fs = [], []
        done = 0
        while done < n_meas:
            s.step(sample_every)
            done += sample_every
            ts.append(s.steps_done)
            fs.append(force())
    fs_a = np.array(fs)
    scale = 2.0 / (1.0 * u_mean**2 * d)
    cd, cl = fs_a[:, 0] * scale, fs_a[:, 1] * scale
    # Strouhal number from the dominant lift frequency (zero-padded FFT + parabolic peak)
    sig = cl - cl.mean()
    nfft = 16 * len(sig)
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), nfft))
    k = int(np.argmax(spec[1:]) + 1)
    if 1 <= k < len(spec) - 1:
        a, b, g = spec[k - 1], spec[k], spec[k + 1]
        k = k + 0.5 * (a - g) / (a - 2 * b + g)
    freq = k / (nfft * sample_every)  # per time step
    st = freq * d / u_mean
    rho, u = s.macroscopic()
    ux, uy = u[0], u[1]
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    vort[p.solid] = np.nan
    return {"d": d, "u_mean": u_mean, "tau": p.tau, "collision": collision,
            "backend": s.name, "shape": list(p.shape), "steps": s.steps_done,
            "St": float(st), "cd_max": float(cd.max()), "cl_max": float(cl.max()),
            "cd_mean": float(cd.mean()), "reference": REFERENCE,
            "t_conv": (np.array(ts) / t_conv).tolist()[::5], "cl": cl.tolist()[::5],
            "cd": cd.tolist()[::5], "_vorticity": vort * d / u_mean, "_solid": p.solid}
