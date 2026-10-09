"""PyTorch rungs of the ladder: eager roll, indexed gather, torch.compile, legacy original."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..core import Ops, apply_outlet, collide, make_geometry, step, stream
from ..problem import Problem
from .base import Solver

try:
    import torch

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    _HAVE_TORCH = False


def _cuda_ok() -> tuple[bool, str]:
    if not _HAVE_TORCH:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    return True, ""


class _TorchBase(Solver):
    device = "cuda"
    dtype_name = "float32"

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return _cuda_ok()

    def __init__(self, problem: Problem, device: str | None = None) -> None:
        super().__init__(problem)
        self.dev = torch.device(device or self.device)
        self.tdtype = getattr(torch, self.dtype_name)
        self.ops = Ops(torch)
        self.f = self._to(problem.initial_populations())

    def _to(self, a: np.ndarray) -> Any:
        if a.dtype == bool:
            return torch.as_tensor(a, device=self.dev)
        return torch.as_tensor(np.asarray(a), device=self.dev, dtype=self.tdtype)

    def set_populations(self, f: np.ndarray) -> None:
        self.f = self._to(f).contiguous()

    def populations(self) -> np.ndarray:
        return self.f.detach().to("cpu", torch.float64).numpy()

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.dev)


class TorchEagerRoll(_TorchBase):
    """``torch.roll`` streaming + elementwise collision, one kernel launch per operation."""

    backend = "torch-eager"
    variant = "roll"
    description = "PyTorch eager, torch.roll streaming (unfused)"

    def __init__(self, problem: Problem, device: str | None = None) -> None:
        super().__init__(problem, device)
        self.geo = make_geometry(problem, self._to)

    @torch.no_grad() if _HAVE_TORCH else (lambda f: f)
    def _advance(self, n: int) -> None:
        for _ in range(n):
            self.f = step(self.f, self.geo, self.ops)


def _gather_tables(problem: Problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source index and sparse wall corrections for the gather formulation.

    The tables are produced by running the reference :func:`pylbm.core.stream` on an array
    that holds each entry's own flat index (with moving walls switched off), and on zeros
    (with moving walls on). The gather backend is therefore boundary-for-boundary identical
    to the reference by construction. The pressure outlet is not a gather; it is applied
    afterwards with the same :func:`pylbm.core.apply_outlet`.
    """
    import dataclasses

    q = problem.lattice.q
    shp = (q,) + problem.array_shape
    ops = Ops(np)
    still = dataclasses.replace(problem, u_lid=0.0, inlet=None, outlet=False)
    src = stream(np.arange(np.prod(shp), dtype=np.float64).reshape(shp),
                 make_geometry(still, np.asarray), ops)
    src = np.stack(src).reshape(-1).astype(np.int64)
    nob = dataclasses.replace(problem, outlet=False)
    corr = np.stack(stream(np.zeros(shp), make_geometry(nob, np.asarray), ops)).reshape(-1)
    pos = np.nonzero(corr)[0].astype(np.int64)
    return src, pos, corr[pos]


class TorchEagerGather(_TorchBase):
    """Streaming + all boundaries as one ``torch.take`` with a precomputed index table."""

    backend = "torch-eager"
    variant = "gather"
    description = "PyTorch eager, precomputed index gather (any geometry)"

    def __init__(self, problem: Problem, device: str | None = None) -> None:
        super().__init__(problem, device)
        src, pos, val = _gather_tables(problem)
        self.src = torch.as_tensor(src, device=self.dev)
        self.pos = torch.as_tensor(pos, device=self.dev)
        self.val = torch.as_tensor(val, device=self.dev, dtype=self.tdtype)
        self.shape = (problem.lattice.q,) + problem.array_shape

    def _gather(self, f: Any) -> Any:
        g = torch.take(f, self.src)
        if self.pos.numel():
            g.index_add_(0, self.pos, self.val)
        g = g.view(self.shape)
        if self.problem.outlet:
            apply_outlet([g[i] for i in range(self.shape[0])], f.view(self.shape), self.problem)
        return g

    @torch.no_grad() if _HAVE_TORCH else (lambda f: f)
    def _advance(self, n: int) -> None:
        q = self.problem.lattice.q
        for _ in range(n):
            g = self._gather(self.f)
            self.f = collide([g[i] for i in range(q)], self.problem, self.ops)

    @property
    def bytes_per_node(self) -> int:  # minimal model, same as everyone else
        return super().bytes_per_node


class TorchCompileRoll(TorchEagerRoll):
    """The eager roll formulation, fused by ``torch.compile`` (Inductor -> Triton)."""

    backend = "torch-compile"
    variant = "roll"
    description = "torch.compile of the roll formulation (Inductor-generated Triton)"

    def __init__(self, problem: Problem, device: str | None = None) -> None:
        super().__init__(problem, device)
        geo, ops = self.geo, self.ops
        self._step = torch.compile(lambda f: step(f, geo, ops), dynamic=False)

    @torch.no_grad() if _HAVE_TORCH else (lambda f: f)
    def _advance(self, n: int) -> None:
        for _ in range(n):
            self.f = self._step(self.f)


class TorchCompileGather(TorchEagerGather):
    """The gather formulation fused by ``torch.compile``."""

    backend = "torch-compile"
    variant = "gather"
    description = "torch.compile of the gather formulation"

    def __init__(self, problem: Problem, device: str | None = None) -> None:
        super().__init__(problem, device)
        q, p, ops = problem.lattice.q, problem, self.ops

        def _step(f: Any) -> Any:
            g = self._gather(f)
            return collide([g[i] for i in range(q)], p, ops)

        self._step = torch.compile(_step, dynamic=False)

    @torch.no_grad() if _HAVE_TORCH else (lambda f: f)
    def _advance(self, n: int) -> None:
        for _ in range(n):
            self.f = self._step(self.f)


# --------------------------------------------------------------------------- legacy
def _load_legacy() -> Any:
    path = Path(__file__).resolve().parents[3] / "legacy" / "Pytorch_LBM_accelerated.py"
    if not path.exists():
        raise ImportError(f"legacy script not found at {path}")
    name = "pylbm_legacy_original"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class TorchLegacyOriginal(_TorchBase):
    """The original pre-2026 script (AoS layout, three compiled functions per step).

    Kept for the performance baseline only: its boundary treatment differs from the
    reference (see ``legacy/Pytorch_LBM_accelerated.py``), so it is excluded from the
    equivalence tests (``equivalent = False``).
    """

    backend = "torch-compile"
    variant = "original"
    description = "original pylbm script (torch.compile, AoS, legacy boundaries)"
    equivalent = False

    @classmethod
    def available(cls) -> tuple[bool, str]:
        ok, why = _cuda_ok()
        if not ok:
            return ok, why
        try:
            _load_legacy()
        except Exception as e:  # pragma: no cover
            return False, f"legacy script not importable: {e}"
        return True, ""

    @classmethod
    def supports(cls, problem: Problem) -> tuple[bool, str]:
        if problem.dim != 2 or not problem.is_box() or any(problem.periodic):
            return False, "legacy code only runs the 2D lid-driven cavity"
        if problem.collision != "bgk" or problem.has_force:
            return False, "legacy code is BGK without forcing"
        return True, ""

    def __init__(self, problem: Problem, device: str | None = None) -> None:
        Solver.__init__(self, problem)
        self.m = _load_legacy()
        self.dev = self.m.DEVICE
        self.tdtype = torch.float32
        self.ops = Ops(torch)
        ny, nx = problem.array_shape
        m = self.m
        self.rho = torch.ones((ny, nx), device=self.dev)
        self.u = torch.zeros((ny, nx, 2), device=self.dev)
        self.f = (self.rho[..., None] * m.w).contiguous()
        self.f_temp = torch.empty_like(self.f)
        self.u_lid = torch.tensor([problem.u_lid, 0.0], device=self.dev)
        self.inv_tau = 1.0 / problem.tau

    # The legacy code moves c_y=+1 populations towards row-1, so its "lid" at row -1 is the
    # mirror image (in y) of our top lid. Mapping its populations with the y-mirror
    # permutation turns its state into our convention.
    _MIRROR_Y = [0, 1, 4, 3, 2, 8, 7, 6, 5]

    def set_populations(self, f: np.ndarray) -> None:
        t = torch.as_tensor(np.ascontiguousarray(f[self._MIRROR_Y]), device=self.dev,
                            dtype=torch.float32)
        self.f = t.permute(1, 2, 0).contiguous()

    def populations(self) -> np.ndarray:
        g = self.f.permute(2, 0, 1).to("cpu", torch.float64).numpy()
        return g[self._MIRROR_Y].copy()

    @torch.no_grad() if _HAVE_TORCH else (lambda f: f)
    def _advance(self, n: int) -> None:
        m = self.m
        for _ in range(n):
            self.f, self.rho, self.u = m.lbm_step_strategic(
                self.f, self.f_temp, self.rho, self.u, self.u_lid, m.cx, m.cy, m.w,
                self.inv_tau)

    def memory_bytes(self) -> int:
        return super().memory_bytes() + 3 * 4 * self.problem.n_nodes
