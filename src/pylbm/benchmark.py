"""Benchmark engine behind ``bench/run.py`` and ``pylbm bench``.

Timing rules (shared Python-HPC contract): warm-up including JIT/compile before timing,
device synchronisation around every timed region, >= 5 repeats (median, min, max), each
timed region >= 0.5 s. Effective bandwidth uses the minimal traffic model: every lattice
update reads and writes each population once, i.e. 2 q s bytes per node for storage size s
(72 B for D2Q9/fp32, 36 B for 16-bit storage, 152 B for D3Q19/fp32).
"""

from __future__ import annotations

import datetime as _dt
import gc
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import backends as B
from .problem import Problem

SPEC_BANDWIDTH_GBS = 448.0  # RTX 3070 Laptop GPU, 256-bit GDDR6 @ 14 Gbps

LADDER_2D = [
    ("numpy", "roll"),
    ("torch-eager", "roll"),
    ("torch-eager", "gather"),
    ("torch-compile", "original"),
    ("torch-compile", "roll"),
    ("torch-compile", "gather"),
    ("numba", "fused"),
    ("warp", "fused"),
    ("triton", "fused"),
    ("cupy-cuda-c", "fused"),
    ("triton", "fused-aa"),
    ("cupy-cuda-c", "fused-aa"),
    ("triton", "fused-fp16"),
    ("triton", "fused-bf16"),
    ("cupy-cuda-c", "fused-fp16"),
    ("triton", "fused-aa-fp16"),
    ("cupy-cuda-c", "fused-aa-fp16"),
]
LADDER_3D = [
    ("numpy", "roll"),
    ("torch-compile", "roll"),
    ("numba", "fused"),
    ("warp", "fused"),
    ("triton", "fused"),
    ("cupy-cuda-c", "fused"),
    ("triton", "fused-aa"),
    ("cupy-cuda-c", "fused-aa"),
    ("triton", "fused-fp16"),
    ("cupy-cuda-c", "fused-fp16"),
    ("cupy-cuda-c", "fused-aa-fp16"),
]
SLOW = {("numpy", "roll"), ("torch-eager", "roll"), ("torch-eager", "gather")}
COMPILE = {("torch-compile", "original"), ("torch-compile", "roll"),
           ("torch-compile", "gather")}


@dataclass
class Plan:
    """What to measure."""

    name: str
    sizes_2d: list[int]
    sizes_3d: list[int]
    slow_2d: list[int]  # sizes for numpy / torch-eager
    compile_2d: list[int]  # sizes for torch.compile variants (each size recompiles)
    slow_3d: list[int]
    compile_3d: list[int]
    collision_size: int  # 2D size for the BGK/TRT/MRT comparison
    repeats: int
    min_time: float
    copy_mib: int


PLANS = {
    "quick": Plan("quick", [1024], [64], [128], [256], [24], [32], 512, 3, 0.15, 256),
    "dev": Plan("dev", [256, 512, 1024, 2048], [64, 128], [256, 512], [512, 2048], [32],
                [64], 2048, 5, 0.5, 256),
    "full": Plan("full", [256, 512, 1024, 2048, 4096], [64, 128, 256], [256, 512],
                 [256, 1024, 4096], [32, 64], [64, 128], 2048, 5, 0.5, 1024),
}


# ----------------------------------------------------------------------------- helpers
def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": platform.python_version(),
                           "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    for mod in ("torch", "triton", "warp", "cupy", "numba", "numpy"):
        try:
            import importlib
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    smi = gpu_state()
    if smi:
        out["driver"] = smi.get("driver_version")
    return out


def gpu_state() -> dict[str, str]:
    """Read-only nvidia-smi snapshot (clocks, temperature, power)."""
    q = "driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals, strict=False))
    except Exception:
        return {}


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _free_bytes() -> int:
    try:
        import torch

        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return 0


def bench_problem(dim: int, n: int, collision: str = "bgk") -> Problem:
    """Benchmark case: lid-driven cavity (all walls, moving lid), Re = 1000-ish."""
    return Problem.cavity(n, re=1000.0, u_lid=0.05, collision=collision, dim=dim)


def time_solver(solver: Any, min_time: float, repeats: int) -> tuple[list[float], int]:
    """Return per-repeat wall times and the number of steps per repeat."""
    m = solver.step_multiple
    steps = 2 * m
    solver.step(steps)  # warm-up incl. JIT / compile
    solver.synchronize()
    while True:  # calibrate
        t0 = time.perf_counter()
        solver.step(steps)
        solver.synchronize()
        dt = time.perf_counter() - t0
        if dt >= min_time:
            break
        grow = max(2.0, min(10.0, 1.2 * min_time / max(dt, 1e-6)))
        steps = int(np.ceil(steps * grow / m) * m)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        solver.step(steps)
        solver.synchronize()
        times.append(time.perf_counter() - t0)
    return times, steps


def measure(backend: str, variant: str, dim: int, n: int, plan: Plan, copy_bw: float,
            collision: str = "bgk", log: Any = print) -> dict | None:
    """Benchmark one (backend, variant, size); returns a run record or None if skipped."""
    cls = B.get(backend, variant)
    ok, why = cls.available()
    if not ok:
        log(f"  skip {backend}/{variant}: {why}")
        return None
    p = bench_problem(dim, n, collision)
    ok, why = cls.supports(p)
    if not ok:
        log(f"  skip {backend}/{variant} {n}^{dim}: {why}")
        return None
    q = p.lattice.q
    need = 3 * q * p.n_nodes * 4 + (q * p.n_nodes * 8 if backend == "torch-eager" else 0)
    if backend not in ("numpy",) and _free_bytes() and need > 0.9 * _free_bytes():
        log(f"  skip {backend}/{variant} {n}^{dim}: needs ~{need / 2**30:.1f} GiB")
        return None
    _free_gpu()
    state0 = gpu_state()
    t_setup = time.perf_counter()
    s = cls(p)
    times, steps = time_solver(s, plan.min_time, plan.repeats)
    setup = time.perf_counter() - t_setup - sum(times)
    mlups = [p.n_nodes * steps / t / 1e6 for t in times]
    med = float(np.median(mlups))
    bpu = s.bytes_per_node
    eff = med * 1e6 * bpu / 1e9
    rec = {
        "backend": backend, "variant": variant, "lattice": p.lattice.name,
        "collision": collision, "size": list(p.shape),
        "dtype": "float32", "storage": getattr(s, "storage", "f32" if backend != "numpy"
                                               else "f64"),
        "metric": "MLUPS", "value": med, "repeats": len(times), "median": med,
        "min": float(min(mlups)), "max": float(max(mlups)), "steps_per_repeat": steps,
        "bytes_per_unit": bpu, "effective_GBs": eff,
        "pct_copy_bw": 100.0 * eff / copy_bw, "pct_spec_bw": 100.0 * eff / SPEC_BANDWIDTH_GBS,
        "flops_per_unit": None, "pct_peak_flops": None,
        "memory_MiB": s.memory_bytes() / 2**20,
        "notes": "; ".join(filter(None, [
            cls.description,
            "float64 storage" if backend == "numpy" else "",
            "legacy boundaries (not equivalent)" if not cls.equivalent else "",
            f"setup+warm-up {setup:.1f}s",
            f"sm clock {state0.get('clocks.sm', '?')}, T {state0.get('temperature.gpu', '?')}C",
        ])),
    }
    if backend == "numpy":
        rec["dtype"] = "float64"
    del s
    _free_gpu()
    log(f"  {backend:13s} {variant:14s} {collision} {'x'.join(map(str, p.shape)):>14s} "
        f"{med:9.1f} MLUPS {eff:6.1f} GB/s {rec['pct_copy_bw']:5.1f}% copy")
    return rec


def run(plan_name: str = "quick", out: str | Path | None = None, only: list[str] | None = None,
        log: Any = print) -> dict:
    """Run a benchmark plan and write the results JSON."""
    plan = PLANS[plan_name]
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "bench"))
    try:
        from roofline import copy_bandwidth  # type: ignore[import-not-found]
    except Exception:  # installed without the repo: inline fallback
        copy_bandwidth = None
    t_start = time.perf_counter()
    if copy_bandwidth is not None:
        ceiling = copy_bandwidth(plan.copy_mib)
        copy_bw = float(ceiling["copy_bandwidth_GBs"])
    else:
        copy_bw = 361.0
    log(f"[pylbm bench:{plan_name}] measured copy bandwidth {copy_bw:.1f} GB/s")
    runs: list[dict] = []

    def keep(b: str, v: str) -> bool:
        return not only or b in only or f"{b}/{v}" in only

    def sizes_for(entry: tuple[str, str], dim: int) -> list[int]:
        if dim == 2:
            if entry in SLOW:
                return plan.slow_2d
            if entry in COMPILE:
                return plan.compile_2d
            return plan.sizes_2d
        if entry in SLOW:
            return plan.slow_3d
        if entry in COMPILE:
            return plan.compile_3d
        return plan.sizes_3d

    for dim, ladder in ((2, LADDER_2D), (3, LADDER_3D)):
        log(f"-- D{dim}Q{9 if dim == 2 else 19} cavity")
        for entry in ladder:
            if not keep(*entry):
                continue
            for n in sizes_for(entry, dim):
                try:
                    rec = measure(*entry, dim, n, plan, copy_bw, log=log)
                except Exception as e:  # never let one backend kill the run
                    log(f"  FAIL {entry} {n}: {type(e).__name__}: {e}")
                    _free_gpu()
                    continue
                if rec:
                    runs.append(rec)
    log("-- collision operators (D2Q9)")
    for entry in (("triton", "fused"), ("cupy-cuda-c", "fused"), ("triton", "fused-fp16")):
        if not keep(*entry):
            continue
        for coll in ("trt", "mrt"):
            try:
                rec = measure(*entry, 2, plan.collision_size, plan, copy_bw, collision=coll,
                              log=log)
            except Exception as e:
                log(f"  FAIL {entry} {coll}: {e}")
                continue
            if rec:
                runs.append(rec)
    result = {
        "machine": _versions(),
        "ceiling": {"copy_bandwidth_GBs": copy_bw, "spec_bandwidth_GBs": SPEC_BANDWIDTH_GBS,
                    "fp32_tflops": None},
        "plan": plan_name,
        "provisional": plan_name != "full",
        "gpu_state_end": gpu_state(),
        "wall_time_s": time.perf_counter() - t_start,
        "runs": runs,
    }
    if out is None:
        out = root / "results" / ("rtx3070-laptop.json" if plan_name == "full"
                                  else f"{plan_name}.json")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    log(f"[pylbm bench] wrote {out} ({len(runs)} runs, {result['wall_time_s']:.0f} s)")
    return result
