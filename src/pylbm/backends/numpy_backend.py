"""Pure NumPy reference solver (float64) -- the correctness oracle."""

from __future__ import annotations

import numpy as np

from ..core import Ops, make_geometry, step
from ..problem import Problem
from .base import Solver


class NumpySolver(Solver):
    """Unfused roll-based pull scheme in float64; supports every feature."""

    backend = "numpy"
    variant = "roll"
    description = "NumPy float64 reference (np.roll streaming)"
    itemsize = 8

    def __init__(self, problem: Problem, dtype: type = np.float64) -> None:
        super().__init__(problem)
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        self.ops = Ops(np)
        self.geo = make_geometry(problem, lambda a: np.asarray(a))
        self.f = problem.initial_populations().astype(dtype)

    def set_populations(self, f: np.ndarray) -> None:
        self.f = np.array(f, dtype=self.dtype)

    def populations(self) -> np.ndarray:
        return self.f.astype(np.float64)

    def _advance(self, n: int) -> None:
        for _ in range(n):
            self.f = step(self.f, self.geo, self.ops)
