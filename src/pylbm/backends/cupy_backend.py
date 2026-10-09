"""Hand-written CUDA C fused kernel (``cuda/lbm.cu``) compiled at runtime by CuPy/NVRTC."""

from __future__ import annotations

import ctypes
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..lattice import MRT_M_D2Q9, MRT_NORM_D2Q9
from ._fused import FusedSolver, cuda_ok

_HAVE = False
_WHY = ""


def _preload_nvrtc() -> None:
    """CuPy needs libnvrtc.so.12; use the copy shipped with the PyTorch wheels if needed."""
    try:
        import nvidia.cuda_nvrtc as m  # type: ignore[import-not-found]

        lib = Path(list(m.__path__)[0]) / "lib" / "libnvrtc.so.12"
        if lib.exists():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


try:
    _preload_nvrtc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp
    _HAVE = True
except Exception as e:  # pragma: no cover
    _WHY = f"cupy not importable: {e}"

_SRC = (Path(__file__).parent / "cuda" / "lbm.cu").read_text()
_PATTERNS = {"ab": 0, "aa-odd": 1, "aa-even": 2}
_STORAGE = {"f32": 0, "f16": 1, "bf16": 2}
_COLL = {"bgk": 0, "trt": 1, "mrt": 2}


def _arr(vals: Any, fmt: str = "{}") -> str:
    return "{" + ", ".join(fmt.format(v.item() if hasattr(v, "item") else v) for v in vals) + "}"


def kernel_source(problem: Any, storage: str, pattern: str, block_x: int) -> str:
    """Full CUDA source: configuration header + ``lbm.cu``."""
    lat = problem.lattice
    c = np.zeros((lat.q, 3), dtype=int)
    c[:, : lat.d] = lat.c
    per = tuple(problem.periodic) + (True,) * (3 - problem.dim)
    defs = {
        "Q": lat.q, "D": lat.d, "BLOCK_X": block_x,
        "CX_INIT": _arr(c[:, 0]), "CY_INIT": _arr(c[:, 1]), "CZ_INIT": _arr(c[:, 2]),
        "W_INIT": _arr(lat.w, "{!r}f"), "OPP_INIT": _arr(lat.opp),
        "PER_X": int(per[0]), "PER_Y": int(per[1]), "PER_Z": int(per[2]),
        "LID": int(problem.u_lid != 0.0), "FORCE": int(problem.has_force),
        "COLL": _COLL[problem.collision], "STORAGE": _STORAGE[storage],
        "PATTERN": _PATTERNS[pattern],
    }
    if problem.collision == "mrt":
        defs["M_INIT"] = "{" + ", ".join(_arr(r, "{!r}f") for r in MRT_M_D2Q9) + "}"
        defs["NORM_INIT"] = _arr(MRT_NORM_D2Q9, "{!r}f")
    head = "\n".join(f"#define {k} {v}" for k, v in defs.items())
    return head + "\n" + _SRC


class CupyFused(FusedSolver):
    """CUDA C kernel via ``cupy.RawModule``; raw device pointers of torch-owned buffers."""

    backend = "cupy-cuda-c"
    variant = "fused"
    description = "hand-written CUDA C (NVRTC via CuPy RawModule), fused, two buffers"
    BLOCK_X = 128

    @classmethod
    def available(cls) -> tuple[bool, str]:
        if not _HAVE:
            return False, _WHY
        return cuda_ok()

    def _compile(self) -> None:
        p = self.problem
        pats = ("aa-odd", "aa-even") if self.aa else ("ab",)
        self.kernels = {}
        for pat in pats:
            src = kernel_source(p, self.storage, pat, self.BLOCK_X)
            mod = cp.RawModule(code=src, options=("--std=c++14",))
            self.kernels[pat] = mod.get_function("lbm_kernel")
        shp = tuple(p.shape) + (1,) * (3 - p.dim)
        self.grid = ((shp[0] + self.BLOCK_X - 1) // self.BLOCK_X, shp[1], shp[2])
        a = self.scalar_args()
        self.args = tuple(np.int32(v) for v in a[:4]) + tuple(np.float32(v) for v in a[4:])

    def _launch(self, which: str, src: Any, dst: Any) -> None:
        self.kernels[which](self.grid, (self.BLOCK_X, 1, 1),
                            (np.uint64(src.data_ptr()), np.uint64(dst.data_ptr()), *self.args))


class CupyFusedAA(CupyFused):
    variant = "fused-aa"
    description = "hand-written CUDA C, AA-pattern in-place streaming (single buffer)"
    aa = True


class CupyFusedFP16(CupyFused):
    variant = "fused-fp16"
    description = "hand-written CUDA C, fp16 storage of f - w_i, fp32 arithmetic"
    storage = "f16"


class CupyFusedAAFP16(CupyFused):
    variant = "fused-aa-fp16"
    description = "hand-written CUDA C, AA pattern + fp16 storage of f - w_i"
    storage = "f16"
    aa = True


os.environ.setdefault("CUPY_CACHE_SAVE_CUDA_SOURCE", "0")
