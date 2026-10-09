"""Common solver interface implemented by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..core import macroscopic
from ..problem import Problem


class Unsupported(Exception):
    """Raised when a backend/variant cannot run a given problem."""


class Solver(ABC):
    """A time stepper for one :class:`~pylbm.problem.Problem`.

    The *canonical state* exchanged through :meth:`populations` / :meth:`set_populations`
    is the post-collision population array in SoA layout ``(q, ny, nx)`` (or ``(q, nz, ny,
    nx)``) as float64, regardless of the backend's internal layout, precision or streaming
    pattern.
    """

    backend: ClassVar[str] = ""
    variant: ClassVar[str] = ""
    description: ClassVar[str] = ""
    #: storage bytes per population value (minimal traffic model: 2 q itemsize per node)
    itemsize: ClassVar[int] = 4
    #: in-place schemes (AA pattern) advance in pairs of steps
    step_multiple: ClassVar[int] = 1
    #: does the minimal traffic model hold (False for e.g. the legacy code)?
    equivalent: ClassVar[bool] = True

    def __init__(self, problem: Problem) -> None:
        ok, why = self.supports(problem)
        if not ok:
            raise Unsupported(f"{self.backend}/{self.variant}: {why}")
        self.problem = problem
        self.steps_done = 0

    # ------------------------------------------------------------------ capabilities
    @classmethod
    def available(cls) -> tuple[bool, str]:
        """Can this backend run on the current machine?"""
        return True, ""

    @classmethod
    def supports(cls, problem: Problem) -> tuple[bool, str]:
        """Can this backend run ``problem``?"""
        return True, ""

    # ------------------------------------------------------------------ state
    @abstractmethod
    def set_populations(self, f: np.ndarray) -> None:
        """Load a canonical post-collision state."""

    @abstractmethod
    def populations(self) -> np.ndarray:
        """Return the canonical post-collision state (float64 numpy array)."""

    @abstractmethod
    def _advance(self, n: int) -> None:
        """Advance ``n`` time steps (n is a multiple of :attr:`step_multiple`)."""

    def step(self, n: int = 1) -> None:
        """Advance ``n`` time steps."""
        if n % self.step_multiple:
            raise ValueError(f"{self.name} advances in multiples of {self.step_multiple} steps")
        self._advance(n)
        self.steps_done += n

    def synchronize(self) -> None:
        """Block until queued device work is done (no-op on the CPU)."""
        return None

    def macroscopic(self) -> tuple[np.ndarray, np.ndarray]:
        """(rho, u) as numpy arrays; ``u`` has shape (d, *array_shape)."""
        rho, u = macroscopic(self.populations(), self.problem)
        return np.asarray(rho), np.stack([np.asarray(c) for c in u])

    # ------------------------------------------------------------------ misc
    @property
    def name(self) -> str:
        return f"{self.backend}/{self.variant}"

    @property
    def bytes_per_node(self) -> int:
        """Minimal memory traffic per lattice update: read q + write q populations."""
        return 2 * self.problem.lattice.q * self.itemsize

    def memory_bytes(self) -> int:
        """Approximate device memory held by the population buffers."""
        return 2 * self.problem.lattice.q * self.itemsize * self.problem.n_nodes


def check_box(problem: Problem) -> tuple[bool, str]:
    """Fused kernels implement only box problems."""
    if not problem.is_box():
        return False, "fused kernels support box geometries only (no obstacles/inlet/outlet)"
    return True, ""
