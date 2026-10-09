"""Backend registry: every rung of the optimization ladder behind one interface.

Backends are imported lazily so that a missing package (or a missing GPU) only disables
the affected entries; it never breaks the CLI.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from .base import Solver, Unsupported

# (backend, variant) -> "module:Class"
_ENTRIES: dict[tuple[str, str], str] = {
    ("numpy", "roll"): "numpy_backend:NumpySolver",
    ("torch-eager", "roll"): "torch_backend:TorchEagerRoll",
    ("torch-eager", "gather"): "torch_backend:TorchEagerGather",
    ("torch-compile", "original"): "torch_backend:TorchLegacyOriginal",
    ("torch-compile", "roll"): "torch_backend:TorchCompileRoll",
    ("torch-compile", "gather"): "torch_backend:TorchCompileGather",
    ("triton", "fused"): "triton_backend:TritonFused",
    ("triton", "fused-aa"): "triton_backend:TritonFusedAA",
    ("triton", "fused-fp16"): "triton_backend:TritonFusedFP16",
    ("triton", "fused-bf16"): "triton_backend:TritonFusedBF16",
    ("triton", "fused-aa-fp16"): "triton_backend:TritonFusedAAFP16",
    ("warp", "fused"): "warp_backend:WarpFused",
    ("numba", "fused"): "numba_backend:NumbaFused",
    ("cupy-cuda-c", "fused"): "cupy_backend:CupyFused",
    ("cupy-cuda-c", "fused-aa"): "cupy_backend:CupyFusedAA",
    ("cupy-cuda-c", "fused-fp16"): "cupy_backend:CupyFusedFP16",
    ("cupy-cuda-c", "fused-aa-fp16"): "cupy_backend:CupyFusedAAFP16",
}

#: plotting colours shared by every repo of the suite
COLORS = {
    "numpy": "#7f7f7f",
    "torch-eager": "#9467bd",
    "torch-compile": "#8c564b",
    "triton": "#1f77b4",
    "warp": "#2ca02c",
    "numba": "#ff7f0e",
    "cupy-cuda-c": "#d62728",
}


def entries() -> list[tuple[str, str]]:
    """All registered (backend, variant) pairs, in ladder order."""
    return list(_ENTRIES)


def get(backend: str, variant: str) -> type[Solver]:
    """Import and return a solver class (raises ImportError if unavailable)."""
    try:
        target = _ENTRIES[(backend, variant)]
    except KeyError as e:
        raise KeyError(f"unknown backend/variant {backend}/{variant}") from e
    mod_name, cls_name = target.split(":")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = importlib.import_module(f"{__name__}.{mod_name}")
    return getattr(mod, cls_name)


def status() -> list[tuple[str, str, bool, str]]:
    """(backend, variant, available, reason) for every registered entry."""
    out = []
    for b, v in _ENTRIES:
        try:
            cls = get(b, v)
            ok, why = cls.available()
        except Exception as e:  # import failure etc.
            ok, why = False, f"{type(e).__name__}: {e}"
        out.append((b, v, ok, why))
    return out


def available() -> list[tuple[str, str]]:
    """Entries that can run on this machine."""
    return [(b, v) for b, v, ok, _ in status() if ok]


def make(backend: str, variant: str, problem: Any, **kw: Any) -> Solver:
    """Instantiate a solver for ``problem``."""
    cls = get(backend, variant)
    ok, why = cls.available()
    if not ok:
        raise Unsupported(f"{backend}/{variant} unavailable: {why}")
    return cls(problem, **kw)


__all__ = ["COLORS", "Solver", "Unsupported", "available", "entries", "get", "make", "status"]
