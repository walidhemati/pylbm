"""Run every validation case and write ``results/validation.json`` (+ Karman snapshot)."""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import backends as B
from ..problem import Problem

ROOT = Path(__file__).resolve().parents[3]


def equivalence(steps: int = 200, log: Any = print) -> list[dict]:
    """Max |f - f_ref| of every available backend vs. the float64 NumPy reference."""
    cases = {
        "cavity-2d-bgk": Problem((48, 40), tau=0.7, u_lid=0.1),
        "cavity-2d-mrt": Problem((48, 40), tau=0.7, u_lid=0.1, collision="mrt"),
        "channel-2d-trt-force": Problem((48, 40), tau=0.7, collision="trt",
                                        periodic=(True, False), force=(1e-5, 2e-6)),
        "cavity-3d-bgk": Problem((20, 16, 12), tau=0.7, u_lid=0.1),
    }
    out = []
    for name, p in cases.items():
        ref = B.make("numpy", "roll", p)
        ref.step(steps)
        fr = ref.populations()
        for b, v in B.entries():
            if b == "numpy":
                continue
            cls = B.get(b, v)
            if not cls.available()[0] or not cls.supports(p)[0] or not cls.equivalent:
                continue
            s = cls(p)
            s.step(steps)
            err = float(np.abs(s.populations() - fr).max())
            out.append({"case": name, "backend": b, "variant": v, "steps": steps,
                        "max_abs_diff": err, "storage": getattr(s, "storage", "f32")})
            log(f"  equivalence {name:22s} {b}/{v:14s} max|df| = {err:.2e}")
            del s
    return out


def run_all(quick: bool = False, out: str | Path | None = None, log: Any = print,
            cases: list[str] | None = None) -> dict:
    """Run the validation suite. ``quick`` uses small grids (minutes on a CPU)."""
    from . import ghia, karman, poiseuille, shear_wave, taylor_green

    gpu = any(b != "numpy" for b, _ in B.available())
    want = set(cases or ["equivalence", "ghia", "shear", "poiseuille", "karman", "tgv"])
    res: dict[str, Any] = {
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "quick" if quick else "full", "gpu": gpu,
    }
    t0 = time.perf_counter()
    if "equivalence" in want:
        log("[validate] backend equivalence vs NumPy reference")
        res["equivalence"] = equivalence(log=log)

    if "ghia" in want:
        log("[validate] lid-driven cavity vs Ghia et al. (1982)")
        if quick or not gpu:
            runs = [(100, 64, "bgk", "numpy", "roll")]
        else:
            runs = [(100, 128, "bgk", None, None), (400, 256, "bgk", None, None),
                    (1000, 256, "bgk", None, None), (1000, 256, "trt", None, None),
                    (1000, 256, "mrt", None, None),
                    (100, 64, "bgk", None, None), (100, 256, "bgk", None, None),
                    (100, 128, "bgk", "triton", "fused-fp16"),
                    (100, 128, "bgk", "triton", "fused-bf16"),
                    (1000, 256, "bgk", "triton", "fused-aa"),
                    (1000, 256, "bgk", "triton", "fused-fp16"),
                    (1000, 256, "bgk", "triton", "fused-bf16"),
                    (1000, 256, "bgk", "cupy-cuda-c", "fused-aa-fp16"),
                    (100, 128, "bgk", "torch-compile", "original")]
        res["ghia"] = []
        for re, n, coll, b, v in runs:
            try:
                r = ghia.run_cavity(re, n, collision=coll, backend=b, variant=v)
            except Exception as e:
                log(f"  ghia Re={re} {b}/{v} failed: {e}")
                continue
            res["ghia"].append(r.as_dict())
            log(f"  Re={re:5d} n={n:4d} {coll} {r.backend:28s} steps={r.steps:7d} "
                f"conv={r.converged} L2={r.errors['L2']:.4f} Linf={r.errors['Linf']:.4f}")

    if "shear" in want:
        log("[validate] shear-wave decay viscosity")
        taus = [0.55, 0.8, 1.0, 1.5] if quick or not gpu else \
            [0.52, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 1.95]
        n = 32 if quick or not gpu else 128
        res["shear_wave"] = []
        for tau in taus:
            r = shear_wave.run_shear_wave(tau, n=n)
            r.pop("t"), r.pop("amplitude")
            res["shear_wave"].append(r)
            log(f"  tau={tau:.2f} nu={r['nu_theory']:.5f} measured={r['nu_measured']:.5f} "
                f"rel.err={r['rel_error']:.2e} ({r['backend']})")

    if "poiseuille" in want:
        log("[validate] Poiseuille channel (float64 reference)")
        hs = (8, 16, 32) if quick else (8, 16, 32, 64)
        res["poiseuille"] = {}
        for coll in ("bgk", "trt"):
            c = poiseuille.convergence(hs, collision=coll, backend="numpy", variant="roll")
            res["poiseuille"][coll] = c
            errs = ", ".join(f"{r['L2']:.2e}" for r in c["runs"])
            log(f"  {coll}: L2 = [{errs}] order={c['order']}")
        if gpu and not quick:
            r = poiseuille.run_poiseuille(32, collision="bgk")
            res["poiseuille"]["bgk_fp32_h32"] = r
            log(f"  fp32 ({r['backend']}) h=32: L2 = {r['L2']:.2e}")

    if "karman" in want and gpu:
        log("[validate] Karman street, Schaefer-Turek 2D-2")
        d = 20 if quick else 40
        r = karman.run_karman(d=d)
        vort, solid = r.pop("_vorticity"), r.pop("_solid")
        np.savez_compressed(ROOT / "results" / "karman_vorticity.npz",
                            vorticity=np.nan_to_num(vort).astype(np.float16), solid=solid,
                            d=d)
        res["karman"] = r
        log(f"  D={d}: St={r['St']:.4f} cd_max={r['cd_max']:.3f} cl_max={r['cl_max']:.3f}")

    if "tgv" in want:
        log("[validate] D3Q19 Taylor-Green decay")
        res["taylor_green_3d"] = []
        cfgs = [("numpy", "roll", 16)] if quick or not gpu else \
            [("triton", "fused", 64), ("cupy-cuda-c", "fused", 64), ("numpy", "roll", 24)]
        for b, v, n in cfgs:
            for plane in ("xy", "yz", "xz"):
                try:
                    r = taylor_green.run_taylor_green(n=n, plane=plane, backend=b, variant=v)
                except Exception as e:
                    log(f"  tgv {b}/{v} failed: {e}")
                    continue
                res["taylor_green_3d"].append(r)
                log(f"  {plane} n={n} {r['backend']}: rel.err nu = {r['rel_error']:.2e}")
    res["wall_time_s"] = time.perf_counter() - t0
    if out is None:
        out = ROOT / "results" / ("validation_quick.json" if quick else "validation.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    log(f"[validate] wrote {out} in {res['wall_time_s']:.0f} s")
    return res
