"""Measure the memory-bandwidth ceiling of the current GPU (device-to-device copy).

Bandwidth counts read + write bytes, matching the coordinator's 361 GB/s reference
measurement (torch ``copy_`` of a 1 GiB buffer on an RTX 3070 Laptop GPU).

    python bench/roofline.py            # 1 GiB buffers (needs ~2 GiB free)
    python bench/roofline.py --mib 256  # smaller, for a shared GPU
"""

from __future__ import annotations

import argparse
import json
import statistics

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _copy(src, dst, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(dst + off, tl.load(src + off, mask=m), mask=m)

except Exception:  # pragma: no cover
    triton = None


def copy_bandwidth(mib: int = 1024, repeats: int = 20) -> dict:
    """Median device-to-device copy bandwidth in GB/s (read + write)."""
    import torch

    n = mib * 2**20 // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda").fill_(1.0)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        b.copy_(a)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) / 1e3)
    bw = [2 * n * 4 / t / 1e9 for t in times]
    del a, b
    torch.cuda.empty_cache()
    return {"copy_bandwidth_GBs": statistics.median(bw), "max_GBs": max(bw), "mib": mib}


def triton_copy_bandwidth(mib: int = 256, repeats: int = 20) -> float:
    """Copy bandwidth of a plain Triton kernel with the LBM kernels' launch shape."""
    import torch

    n = mib * 2**20 // 4
    a = torch.ones(n, device="cuda")
    b = torch.empty_like(a)
    grid = (triton.cdiv(n, 1024),)
    for _ in range(3):
        _copy[grid](a, b, n, BLOCK=1024)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _copy[grid](a, b, n, BLOCK=1024)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) / 1e3)
    return statistics.median(2 * n * 4 / t / 1e9 for t in times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mib", type=int, default=1024)
    a = ap.parse_args()
    out = copy_bandwidth(a.mib)
    out["triton_copy_GBs"] = triton_copy_bandwidth(min(a.mib, 256))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
