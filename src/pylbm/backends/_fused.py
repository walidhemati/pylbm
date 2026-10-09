"""Shared plumbing for the fused single-kernel GPU backends.

Population buffers are torch CUDA tensors (flat, SoA, ``q * N`` values); the kernels of
other toolkits (Warp, Numba, CuPy) see the same memory zero-copy through the CUDA array
interface.  Two-buffer variants ping-pong between ``A`` and ``B``; AA-pattern variants keep
a single buffer and alternate the odd/even kernels.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..problem import Problem
from .base import Solver, check_box

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_TORCH_DTYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}


def cuda_ok() -> tuple[bool, str]:
    if torch is None:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    return True, ""


class FusedSolver(Solver):
    """Base class: subclasses implement :meth:`_compile` and :meth:`_launch`."""

    storage: ClassVar[str] = "f32"
    aa: ClassVar[bool] = False
    lattices: ClassVar[tuple[str, ...]] = ("D2Q9", "D3Q19")
    collisions: ClassVar[tuple[str, ...]] = ("bgk", "trt", "mrt")

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return cuda_ok()

    @classmethod
    def supports(cls, problem: Problem) -> tuple[bool, str]:
        ok, why = check_box(problem)
        if not ok:
            return ok, why
        if problem.lattice.name not in cls.lattices:
            return False, f"{problem.lattice.name} not implemented"
        if problem.collision not in cls.collisions:
            return False, f"{problem.collision} not implemented"
        q = problem.lattice.q
        if q * problem.n_nodes >= 2**31:
            return False, "int32 indexing limit"
        return True, ""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.step_multiple = 2 if self.aa else 1
        self.itemsize = 2 if self.storage in ("f16", "bf16") else 4
        self.q = problem.lattice.q
        self.N = problem.n_nodes
        self.tdtype = getattr(torch, _TORCH_DTYPES[self.storage])
        self.w = problem.lattice.w
        self.opp = problem.lattice.opp
        self.A = torch.empty(self.q * self.N, device="cuda", dtype=self.tdtype)
        self.B = None if self.aa else torch.empty_like(self.A)
        self._compile()
        self.set_populations(problem.initial_populations())

    # ------------------------------------------------------------------ state
    def _encode(self, f: np.ndarray) -> Any:
        f = np.asarray(f, dtype=np.float64)
        if self.storage != "f32":
            f = f - self.w.reshape((-1,) + (1,) * (f.ndim - 1))
        if self.aa:
            f = f[self.opp]  # swapped layout: slot opp(i) holds population i
        t = torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32)
        return t.to(self.tdtype).reshape(-1).cuda()

    def set_populations(self, f: np.ndarray) -> None:
        self.A.copy_(self._encode(f))

    def populations(self) -> np.ndarray:
        self.synchronize()
        g = self.A.to(torch.float64).cpu().numpy().reshape(
            (self.q,) + self.problem.array_shape)
        if self.aa:
            g = g[self.opp]
        if self.storage != "f32":
            g = g + self.w.reshape((-1,) + (1,) * (g.ndim - 1))
        return g

    def memory_bytes(self) -> int:
        return (1 if self.aa else 2) * self.q * self.N * self.itemsize

    # ------------------------------------------------------------------ stepping
    def _compile(self) -> None:
        raise NotImplementedError

    def _launch(self, which: str, src: Any, dst: Any) -> None:
        """Launch kernel ``which`` ('ab', 'aa-odd', 'aa-even') reading src, writing dst."""
        raise NotImplementedError

    def _advance(self, n: int) -> None:
        if self.aa:
            for _ in range(n // 2):
                self._launch("aa-odd", self.A, self.A)
                self._launch("aa-even", self.A, self.A)
        else:
            for _ in range(n):
                self._launch("ab", self.A, self.B)
                self.A, self.B = self.B, self.A

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def scalar_args(self) -> tuple:
        """(nx, ny, nz, N, ulid, fx, fy, fz, r0, r1, r2, r3) for the kernels."""
        from ..codegen import runtime_rates

        p = self.problem
        shp = tuple(p.shape) + (1,) * (3 - p.dim)
        force = tuple(p.force) + (0.0,) * (3 - p.dim)
        return (*shp, self.N, float(p.u_lid), *force, *runtime_rates(p))
