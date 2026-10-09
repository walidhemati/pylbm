"""Physical validation cases (Ghia cavity, shear wave, Poiseuille, Karman street, 3D TGV)."""

from __future__ import annotations

from ..backends import Solver, get
from ..problem import Problem

#: preference order when a case does not ask for a specific backend
GPU_PREFERENCE = [
    ("triton", "fused"),
    ("cupy-cuda-c", "fused"),
    ("warp", "fused"),
    ("numba", "fused"),
    ("torch-compile", "gather"),
    ("torch-eager", "gather"),
    ("numpy", "roll"),
]


def make_solver(problem: Problem, backend: str | None = None, variant: str | None = None,
                prefer: list[tuple[str, str]] | None = None) -> Solver:
    """Instantiate ``backend/variant`` or the first available backend that supports it."""
    if backend is not None:
        cls = get(backend, variant or "fused")
        ok, why = cls.available()
        if not ok:
            raise RuntimeError(f"{backend}/{variant} unavailable: {why}")
        return cls(problem)
    for b, v in prefer or GPU_PREFERENCE:
        try:
            cls = get(b, v)
        except Exception:
            continue
        if cls.available()[0] and cls.supports(problem)[0]:
            return cls(problem)
    raise RuntimeError("no backend supports this problem")
