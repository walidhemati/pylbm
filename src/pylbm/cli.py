"""Command line interface: ``pylbm run | validate | bench | backends | kernels``."""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np


def _cmd_backends(_: argparse.Namespace) -> None:
    from . import backends as B

    for b, v, ok, why in B.status():
        cls = None
        try:
            cls = B.get(b, v)
        except Exception:
            pass
        desc = cls.description if cls else ""
        print(f"{'ok ' if ok else '-- '} {b + '/' + v:28s} {desc if ok else why}")


def _cmd_run(a: argparse.Namespace) -> None:
    from .problem import Problem
    from .validation import make_solver

    if a.case == "cavity":
        p = Problem.cavity(a.n, re=a.re, u_lid=a.u_lid, collision=a.collision)
    elif a.case == "cavity3d":
        p = Problem.cavity(a.n, re=a.re, u_lid=a.u_lid, collision=a.collision, dim=3)
    elif a.case == "karman":
        from .validation.karman import schafer_turek_problem

        p = schafer_turek_problem(d=max(10, a.n // 10), collision=a.collision)
    else:
        raise SystemExit(f"unknown case {a.case}")
    s = make_solver(p, a.backend, a.variant)
    steps = a.steps - a.steps % s.step_multiple
    print(f"{s.name}: {p.lattice.name} {'x'.join(map(str, p.shape))}, tau={p.tau:.4f}, "
          f"{steps} steps")
    s.step(s.step_multiple)
    s.synchronize()
    t0 = time.perf_counter()
    s.step(steps)
    s.synchronize()
    dt = time.perf_counter() - t0
    print(f"{p.n_nodes * steps / dt / 1e6:.1f} MLUPS ({dt:.2f} s, not a benchmark: use "
          f"bench/run.py)")
    if a.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rho, u = s.macroscopic()
        speed = np.sqrt((u**2).sum(axis=0))
        if p.dim == 3:
            speed = speed[:, p.shape[1] // 2, :]  # (z, x) slice through the middle
        fig, ax = plt.subplots(figsize=(5, 5 * speed.shape[0] / speed.shape[1]))
        ax.imshow(speed, origin="lower", cmap="turbo")
        ax.set_axis_off()
        fig.savefig(a.out, dpi=150, bbox_inches="tight")
        print(f"wrote {a.out}")


def _cmd_validate(a: argparse.Namespace) -> None:
    from .validation.runner import run_all

    run_all(quick=a.quick, out=a.out, cases=a.cases)


def _cmd_bench(a: argparse.Namespace) -> None:
    from .benchmark import run

    plan = "full" if a.full else "dev" if a.dev else "quick"
    run(plan, out=a.out, only=a.only)


def _cmd_kernels(a: argparse.Namespace) -> None:
    from .codegen import KernelSpec, source

    d = 3 if a.lattice == "D3Q19" else 2
    spec = KernelSpec(lattice=a.lattice, periodic=(False,) * d, lid=True, force=a.force,
                      collision=a.collision, storage=a.storage, pattern=a.pattern)
    if a.target == "cuda":
        from .backends.cupy_backend import kernel_source
        from .problem import Problem

        p = Problem((8,) * d, collision=a.collision, u_lid=0.1)
        print(kernel_source(p, a.storage, a.pattern, 128))
    else:
        print(source(spec, a.target))


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore", module="cupy")
    ap = argparse.ArgumentParser(prog="pylbm", description="Roofline-optimized LBM in Python")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("backends", help="list backends and whether they run here")
    sp.set_defaults(func=_cmd_backends)

    sp = sub.add_parser("run", help="run a case and report throughput")
    sp.add_argument("case", choices=["cavity", "cavity3d", "karman"])
    sp.add_argument("--n", type=int, default=256)
    sp.add_argument("--re", type=float, default=100.0)
    sp.add_argument("--u-lid", type=float, default=0.1)
    sp.add_argument("--collision", default="bgk", choices=["bgk", "trt", "mrt"])
    sp.add_argument("--backend", default=None)
    sp.add_argument("--variant", default=None)
    sp.add_argument("--steps", type=int, default=10000)
    sp.add_argument("--out", default=None, help="write a speed image (png)")
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("validate", help="run the validation suite")
    sp.add_argument("--quick", action="store_true", help="small grids (CPU friendly)")
    sp.add_argument("--cases", nargs="*", default=None,
                    help="subset of: equivalence ghia shear poiseuille karman tgv")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=_cmd_validate)

    sp = sub.add_parser("bench", help="run the benchmark (same engine as bench/run.py)")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--dev", action="store_true")
    g.add_argument("--full", action="store_true")
    sp.add_argument("--out", default=None)
    sp.add_argument("--only", nargs="*", default=None)
    sp.set_defaults(func=_cmd_bench)

    sp = sub.add_parser("kernels", help="print a generated / CUDA C kernel")
    sp.add_argument("--target", default="triton", choices=["triton", "warp", "numba", "cuda"])
    sp.add_argument("--lattice", default="D2Q9", choices=["D2Q9", "D3Q19"])
    sp.add_argument("--collision", default="bgk", choices=["bgk", "trt", "mrt"])
    sp.add_argument("--storage", default="f32", choices=["f32", "f16", "bf16"])
    sp.add_argument("--pattern", default="ab", choices=["ab", "aa-odd", "aa-even"])
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_kernels)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
