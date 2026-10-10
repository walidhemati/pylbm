"""Lettuce timing worker for ``bench/counterparts.py`` (runs inside Lettuce's own venv).

    ~/.venvs/cp-lettuce/bin/python bench/lettuce_worker.py '<json config>'

The config holds ``runs`` (a list of ``{"dim", "n", "native"}``), ``min_time`` and
``repeats``. The case mirrors Lettuce's own ``lettuce benchmark`` command: a periodic
Taylor--Green vortex (Re = 1, Ma = 0.05), BGK collision, PRE_STREAMING, single precision.
Every timed region is bracketed by ``torch.cuda.synchronize()``. One line
``CP_JSON {...}`` is printed per run. This file imports Lettuce and is never imported by pylbm.
"""

from __future__ import annotations

import json
import sys
import time


def _one(lt: object, torch: object, dim: int, n: int, native: bool, min_time: float,
         repeats: int) -> dict:
    ctx = lt.Context(torch.device("cuda:0"), torch.float32, use_native=native)
    stencil = lt.D2Q9 if dim == 2 else lt.D3Q19
    flow = lt.TaylorGreenVortex(ctx, resolution=[n] * dim, reynolds_number=1,
                                mach_number=0.05, stencil=stencil)
    coll = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
    sim = lt.Simulation(flow, coll, [], streaming_strategy=lt.StreamingStrategy.PRE_STREAMING)
    if native and sim._collide_and_stream.__module__.startswith("lettuce._simulation"):
        raise RuntimeError("native CUDA kernel requested but not resolved")
    torch.cuda.reset_peak_memory_stats()
    t_setup = time.perf_counter()
    steps = 10
    sim(steps)  # warm-up (and native module load)
    torch.cuda.synchronize()
    while True:  # calibrate: one timed region >= min_time
        t0 = time.perf_counter()
        sim(steps)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if dt >= min_time:
            break
        steps = int(steps * max(2.0, min(10.0, 1.2 * min_time / max(dt, 1e-6))))
    setup = time.perf_counter() - t_setup
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sim(steps)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    finite = bool(torch.isfinite(flow.f).all().item())
    return {"dim": dim, "n": n, "native": native, "nodes": n**dim, "steps": steps,
            "times": times, "setup_s": setup, "finite": finite,
            "peak_MiB": torch.cuda.max_memory_allocated() / 2**20}


def main() -> None:
    cfg = json.loads(sys.argv[1])
    import lettuce as lt
    import torch

    head = {"lettuce": lt.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    print("CP_JSON " + json.dumps({"header": head}), flush=True)
    for r in cfg["runs"]:
        try:
            out = _one(lt, torch, r["dim"], r["n"], r["native"], cfg["min_time"],
                       cfg["repeats"])
        except Exception as e:  # report (e.g. out of memory) and continue
            out = {**r, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
        torch.cuda.empty_cache()
        print("CP_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
