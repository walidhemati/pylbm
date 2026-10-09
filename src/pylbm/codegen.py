"""Fused stream-collide kernel generator for the Python GPU DSLs (Triton, Warp, Numba).

One symbolic description of the fused LBM update (pull streaming, half-way bounce-back
with Ladd's moving-wall term, BGK/TRT/MRT collision with Guo forcing, optional shifted
16-bit storage, two-buffer or AA-pattern in-place streaming) is printed as straight-line
source code for each target. All targets therefore execute exactly the same arithmetic;
differences in speed come from the compilers. The generated source is written to a cache
directory and imported (Triton and Warp need real source files); ``pylbm kernels`` prints it.

Pattern semantics (populations stored post-collision, SoA, plane stride ``N``):

* ``ab``      : read f_in[i, x - c_i] (or bounce-back), collide, write f_out[i, x].
* ``aa-odd``  : read A[opp(i), x - c_i] (or A[i, x] on a wall link), collide, write
                A[i, x + c_i] (or A[opp(i), x] on a wall link).  Leaves A in natural layout.
* ``aa-even`` : read A[i, x], collide, write A[opp(i), x].  Leaves A swapped (canonical
                post-collision state of population i lives in slot opp(i)).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from .lattice import LATTICES, MRT_M_D2Q9, MRT_NORM_D2Q9

TARGETS = ("triton", "numba", "warp")


@dataclass(frozen=True)
class KernelSpec:
    """Compile-time configuration of one fused kernel."""

    lattice: str = "D2Q9"
    periodic: tuple[bool, ...] = (False, False)
    lid: bool = False
    force: bool = False
    collision: str = "bgk"
    storage: str = "f32"  # f32 | f16 | bf16 (16-bit formats store f - w_i)
    pattern: str = "ab"  # ab | aa-odd | aa-even

    @property
    def tag(self) -> str:
        s = repr(self).encode()
        return hashlib.sha1(s).hexdigest()[:12]


# ----------------------------------------------------------------------------- printers
class _Printer:
    """Target-specific syntax."""

    def __init__(self, target: str) -> None:
        if target not in TARGETS:
            raise ValueError(target)
        self.t = target

    def lit(self, v: float) -> str:
        r = repr(float(v))
        if self.t == "numba":
            return f"float32({r})"
        return r

    def sel(self, c: str, a: str, b: str) -> str:
        if self.t == "triton":
            return f"tl.where({c}, {a}, {b})"
        if self.t == "warp":
            return f"wp.where({c}, {a}, {b})"
        return f"({a} if {c} else {b})"

    def or_(self, terms: list[str]) -> str:
        if not terms:
            return "False"
        op = " | " if self.t == "triton" else " or "
        return "(" + op.join(f"({t})" for t in terms) + ")"

    def int_sel(self, c: str, a: str, b: str) -> str:
        return self.sel(c, a, b)


def _signed_sum(terms: list[tuple[int, str]], pr: _Printer | None = None) -> str:
    """Render sum of (integer coefficient, name) pairs."""
    out = ""
    for coef, name in terms:
        if coef == 0:
            continue
        mag = abs(coef)
        if mag == 1:
            piece = name
        else:
            piece = f"{pr.lit(mag) if pr else repr(float(mag))} * {name}"
        if not out:
            out = piece if coef > 0 else f"-{piece}"
        else:
            out += f" + {piece}" if coef > 0 else f" - {piece}"
    return out or (pr.lit(0.0) if pr else "0.0")


# ----------------------------------------------------------------------------- body
def _body(spec: KernelSpec, pr: _Printer) -> list[str]:
    """Per-node statements; assumes x, y(, z), n, and (Triton) ``valid`` are defined."""
    lat = LATTICES[spec.lattice]
    d, q = lat.d, lat.q
    coords = ["x", "y", "z"][:d]
    sizes = ["nx", "ny", "nz"][:d]
    strides = ["1", "nx", "nxy"][:d]
    top = d - 1
    L: list[str] = []
    tri = pr.t == "triton"
    lit = pr.lit

    def load(index: str) -> str:
        if tri:
            v = f"tl.load(A + ({index}), mask=valid)"
            return v if spec.storage == "f32" else f"{v}.to(tl.float32)"
        return f"A[{index}]"

    def stored(val: str, i: int) -> str:
        if spec.storage == "f32":
            return val
        dt = "tl.float16" if spec.storage == "f16" else "tl.bfloat16"
        return f"({val} - {lit(lat.w[i])}).to({dt})"

    def store(index: str, val: str) -> list[str]:
        if tri:
            return [f"tl.store(B + ({index}), {val}, mask=valid)"]
        return [f"B[{index}] = {val}"]

    def shift(i: int) -> str:
        return "" if spec.storage == "f32" else f" + {lit(lat.w[i])}"

    def neighbour(i: int, sign: int) -> tuple[str, list[str]]:
        """Flat index of x + sign*c_i (with periodic wrap) and the wall conditions."""
        parts = ["n"]
        walls = []
        for a in range(d):
            ca = int(lat.c[i, a]) * sign
            if ca == 0:
                continue
            edge = "0" if ca == -1 else f"{sizes[a]} - 1"
            hit = f"{coords[a]} == {edge}"
            step = strides[a]
            delta = f"{step}" if ca == 1 else f"-{step}"
            if spec.periodic[a]:
                span = f"{sizes[a]} * {step}" if step != "1" else sizes[a]
                wrap = f"-{span}" if ca == 1 else span
                parts.append(pr.int_sel(hit, f"{delta} + ({wrap})", delta))
            else:
                walls.append((a, hit))
                parts.append(delta)
        idx = " + ".join(f"({p})" if k else p for k, p in enumerate(parts))
        return idx, walls

    def lid_term(i: int, sign: int, walls: list[tuple[int, str]]) -> str:
        """Ladd correction for a link whose far end (x + sign*c_i) is behind the lid."""
        if not spec.lid or lat.c[i, 0] == 0:
            return ""
        far_top = int(lat.c[i, top]) * sign
        if far_top != 1:
            return ""
        hit = [h for a, h in walls if a == top][0]
        # pulling direction i from the lid (sign=-1): +6 w_i c_ix U.  Pushing i into the
        # lid (sign=+1) gives opp(i) the term 6 w_opp c_opp,x U = -6 w_i c_ix U.
        k = 6.0 * lat.w[i] * lat.c[i, 0] * (-1 if sign == 1 else 1)
        return pr.sel(hit, f"{lit(k)} * ulid", lit(0.0))

    # ---------------------------------------------------------------- read
    L.append("# --- streaming (pull) with half-way bounce-back")
    for i in range(q):
        o = int(lat.opp[i])
        if spec.pattern == "aa-even" or i == 0:
            slot = i if spec.pattern != "aa-odd" else o
            L.append(f"f{i} = {load(f'{slot} * N + n')}{shift(i)}")
            continue
        nidx, walls = neighbour(i, -1)
        if spec.pattern == "ab":
            far = f"{i} * N + {nidx}"
            near = f"{o} * N + n"
        else:  # aa-odd: neighbour holds it in slot opp(i), own wall value in slot i
            far = f"{o} * N + {nidx}"
            near = f"{i} * N + n"
        if walls:
            bb = pr.or_([h for _, h in walls])
            L.append(f"bb{i} = {bb}")
            L.append(f"f{i} = {load(pr.int_sel(f'bb{i}', near, far))}{shift(i)}")
            lt = lid_term(i, -1, walls)
            if lt:
                L.append(f"f{i} = f{i} + {lt}")
        else:
            L.append(f"f{i} = {load(far)}{shift(i)}")

    # ---------------------------------------------------------------- moments
    L.append("# --- moments")
    L.append("rho = " + " + ".join(f"f{i}" for i in range(q)))
    L.append(f"inv_rho = {lit(1.0)} / rho")
    fnames = ["fx", "fy", "fz"][:d]
    for a in range(d):
        j = _signed_sum([(int(lat.c[i, a]), f"f{i}") for i in range(q)], pr)
        if spec.force:
            j = f"({j}) + {lit(0.5)} * {fnames[a]}"
        L.append(f"u{coords[a]} = ({j}) * inv_rho")
    L.append("usq = " + " + ".join(f"u{c} * u{c}" for c in coords))
    for i in range(1, q):
        L.append(f"cu{i} = " + _signed_sum([(int(lat.c[i, a]), f"u{coords[a]}")
                                             for a in range(d)], pr))
    L.append(f"base = {lit(1.0)} - {lit(1.5)} * usq")
    L.append(f"rw0 = rho * {lit(lat.w[0])}")
    L.append("feq0 = rw0 * base")
    wvals = sorted(set(lat.w[1:].tolist()))
    for k, wv in enumerate(wvals):
        L.append(f"rw{k + 1} = rho * {lit(wv)}")
    for i in range(1, q):
        k = wvals.index(lat.w[i]) + 1
        L.append(f"feq{i} = rw{k} * (base + cu{i} * ({lit(3.0)} + {lit(4.5)} * cu{i}))")
    if spec.force:
        L.append("uf = " + " + ".join(f"u{coords[a]} * {fnames[a]}" for a in range(d)))
        for i in range(q):
            cf = _signed_sum([(int(lat.c[i, a]), fnames[a]) for a in range(d)], pr)
            if i == 0:
                L.append(f"S0 = {lit(lat.w[0])} * ({lit(-3.0)} * uf)")
            else:
                L.append(f"cf{i} = {cf}")
                L.append(f"S{i} = {lit(lat.w[i])} * ({lit(3.0)} * (cf{i} - uf)"
                         f" + {lit(9.0)} * cu{i} * cf{i})")

    # ---------------------------------------------------------------- collision
    L.append(f"# --- collision ({spec.collision})")
    if spec.collision == "bgk":
        if spec.force:
            L.append(f"gf = {lit(1.0)} - {lit(0.5)} * r0")
        for i in range(q):
            e = f"f{i} + r0 * (feq{i} - f{i})"
            if spec.force:
                e += f" + gf * S{i}"
            L.append(f"p{i} = {e}")
    elif spec.collision == "trt":
        if spec.force:
            L.append(f"gp = {lit(1.0)} - {lit(0.5)} * r0")
            L.append(f"gm = {lit(1.0)} - {lit(0.5)} * r1")
        e = "f0 - r0 * (f0 - feq0)"
        if spec.force:
            e += " + gp * S0"
        L.append(f"p0 = {e}")
        done = {0}
        for i in range(1, q):
            o = int(lat.opp[i])
            if i in done:
                continue
            done |= {i, o}
            L.append(f"np{i} = r0 * ({lit(0.5)} * (f{i} + f{o}) - {lit(0.5)} * "
                     f"(feq{i} + feq{o}))")
            L.append(f"nm{i} = r1 * ({lit(0.5)} * (f{i} - f{o}) - {lit(0.5)} * "
                     f"(feq{i} - feq{o}))")
            if spec.force:
                L.append(f"sp{i} = gp * {lit(0.5)} * (S{i} + S{o})")
                L.append(f"sm{i} = gm * {lit(0.5)} * (S{i} - S{o})")
                L.append(f"p{i} = f{i} - np{i} - nm{i} + sp{i} + sm{i}")
                L.append(f"p{o} = f{o} - np{i} + nm{i} + sp{i} - sm{i}")
            else:
                L.append(f"p{i} = f{i} - np{i} - nm{i}")
                L.append(f"p{o} = f{o} - np{i} + nm{i}")
    elif spec.collision == "mrt":
        if spec.lattice != "D2Q9":
            raise ValueError("MRT only for D2Q9")
        M, nrm = MRT_M_D2Q9, MRT_NORM_D2Q9
        # rates per moment: rho, e, eps, jx, qx, jy, qy, pxx, pxy
        rate = [None, "r1", "r2", None, "r3", None, "r3", "r0", "r0"]
        for i in range(q):
            L.append(f"ne{i} = f{i} - feq{i}")
        dm = {}
        for k in range(q):
            row = [(int(M[k, i]), i) for i in range(q) if M[k, i] != 0]
            if rate[k] is None:
                if spec.force:
                    L.append(f"dm{k} = " + _signed_sum([(c, f"S{i}") for c, i in row], pr))
                    dm[k] = True
                continue
            e = f"-{rate[k]} * ({_signed_sum([(c, f'ne{i}') for c, i in row], pr)})"
            if spec.force:
                e += (f" + ({lit(1.0)} - {lit(0.5)} * {rate[k]}) * "
                      f"({_signed_sum([(c, f'S{i}') for c, i in row], pr)})")
            L.append(f"dm{k} = {e}")
            dm[k] = True
        for k in dm:
            L.append(f"dn{k} = dm{k} * {lit(1.0 / nrm[k])}")
        for i in range(q):
            terms = [(int(M[k, i]), f"dn{k}") for k in dm if M[k, i] != 0]
            L.append(f"p{i} = f{i} + " + _signed_sum(terms, pr))
    else:
        raise ValueError(spec.collision)

    # ---------------------------------------------------------------- write
    L.append("# --- write")
    for i in range(q):
        o = int(lat.opp[i])
        if spec.pattern == "ab":
            L += store(f"{i} * N + n", stored(f"p{i}", i))
        elif spec.pattern == "aa-even":
            L += store(f"{o} * N + n", stored(f"p{i}", i))
        else:  # aa-odd: push to x + c_i, or bounce into own slot opp(i)
            if i == 0:
                L += store("n", stored("p0", 0))
                continue
            nidx, walls = neighbour(i, +1)
            if walls:
                bb = pr.or_([h for _, h in walls])
                L.append(f"wb{i} = {bb}")
                val = f"p{i}"
                lt = lid_term(i, +1, walls)
                if lt:
                    val = f"(p{i} + {lt})"
                if tri:
                    L += [f"tl.store(B + ({i} * N + {nidx}), {stored(f'p{i}', i)}, "
                          f"mask=valid & ~wb{i})",
                          f"tl.store(B + ({o} * N + n), {stored(val, i)}, mask=valid & wb{i})"]
                else:
                    L.append(f"if {'wb' + str(i)}:")
                    L.append(f"    B[{o} * N + n] = {stored(val, i)}")
                    L.append("else:")
                    L.append(f"    B[{i} * N + {nidx}] = {stored(f'p{i}', i)}")
            else:
                L += store(f"{i} * N + {nidx}", stored(f"p{i}", i))
    return L


# ----------------------------------------------------------------------------- modules
_ARGS = "A, B, nx, ny, nz, N, ulid, fx, fy, fz, r0, r1, r2, r3"


def _triton_module(spec: KernelSpec) -> str:
    pr = _Printer("triton")
    d = LATTICES[spec.lattice].d
    head = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        f"def lbm_kernel({_ARGS}, BLOCK: tl.constexpr):",
        "    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "    y = tl.program_id(1)",
        "    valid = x < nx",
    ]
    if d == 2:
        head += ["    n = x + nx * y"]
    else:
        head += ["    z = tl.program_id(2)", "    nxy = nx * ny", "    n = x + nx * y + nxy * z"]
    body = ["    " + ln for ln in _body(spec, pr)]
    return "\n".join(head + body) + "\n"


def _numba_module(spec: KernelSpec) -> str:
    pr = _Printer("numba")
    d = LATTICES[spec.lattice].d
    head = [
        "from numba import cuda, float32",
        "",
        "",
        "@cuda.jit(fastmath=False)",
        f"def lbm_kernel({_ARGS}):",
    ]
    if d == 2:
        head += ["    x, y = cuda.grid(2)", "    if x >= nx or y >= ny:", "        return",
                 "    nxy = nx * ny", "    n = x + nx * y"]
    else:
        head += ["    x, y, z = cuda.grid(3)", "    if x >= nx or y >= ny or z >= nz:",
                 "        return", "    nxy = nx * ny", "    n = x + nx * y + nxy * z"]
    body = ["    " + ln for ln in _body(spec, pr)]
    return "\n".join(head + body) + "\n"


def _warp_module(spec: KernelSpec) -> str:
    pr = _Printer("warp")
    d = LATTICES[spec.lattice].d
    head = [
        "import warp as wp",
        "",
        "",
        "@wp.kernel",
        "def lbm_kernel(A: wp.array(dtype=wp.float32), B: wp.array(dtype=wp.float32),",
        "               nx: int, ny: int, nz: int, N: int, ulid: float, fx: float,",
        "               fy: float, fz: float, r0: float, r1: float, r2: float, r3: float):",
    ]
    if d == 2:
        head += ["    y, x = wp.tid()", "    nxy = nx * ny", "    n = x + nx * y"]
    else:
        head += ["    z, y, x = wp.tid()", "    nxy = nx * ny", "    n = x + nx * y + nxy * z"]
    body = ["    " + ln for ln in _body(spec, pr)]
    return "\n".join(head + body) + "\n"


_MODULES = {"triton": _triton_module, "numba": _numba_module, "warp": _warp_module}


def source(spec: KernelSpec, target: str) -> str:
    """Generated module source for ``target``."""
    return f'"""Generated by pylbm.codegen for {spec!r}. Do not edit."""\n' + \
        _MODULES[target](spec)


def cache_dir() -> Path:
    root = os.environ.get("PYLBM_CACHE", os.path.join(Path.home(), ".cache", "pylbm"))
    p = Path(root) / "kernels"
    p.mkdir(parents=True, exist_ok=True)
    return p


_LOADED: dict[tuple[str, str], ModuleType] = {}


def load(spec: KernelSpec, target: str) -> ModuleType:
    """Generate (if needed) and import the kernel module for ``spec`` on ``target``."""
    key = (spec.tag, target)
    if key in _LOADED:
        return _LOADED[key]
    src = source(spec, target)
    digest = hashlib.sha1(src.encode()).hexdigest()[:12]
    name = f"pylbm_{target}_{digest}"
    path = cache_dir() / f"{name}.py"
    if not path.exists() or path.read_text() != src:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)
    modspec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(modspec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    modspec.loader.exec_module(mod)  # type: ignore[union-attr]
    _LOADED[key] = mod
    return mod


def runtime_rates(problem) -> tuple[float, float, float, float]:  # noqa: ANN001
    """(r0, r1, r2, r3) runtime relaxation parameters for ``problem.collision``."""
    if problem.collision == "bgk":
        return 1.0 / problem.tau, 0.0, 0.0, 0.0
    if problem.collision == "trt":
        sp, sm = problem.trt_rates()
        return sp, sm, 0.0, 0.0
    s = problem.mrt_rates()
    return float(s[7]), float(s[1]), float(s[2]), float(s[4])


def spec_for(problem, storage: str = "f32", pattern: str = "ab") -> KernelSpec:  # noqa: ANN001
    """KernelSpec for a box :class:`~pylbm.problem.Problem`."""
    return KernelSpec(
        lattice=problem.lattice.name,
        periodic=tuple(problem.periodic),
        lid=problem.u_lid != 0.0,
        force=problem.has_force,
        collision=problem.collision,
        storage=storage,
        pattern=pattern,
    )


def to_storage(f: np.ndarray, storage: str, w: np.ndarray) -> np.ndarray:
    """Canonical float64 populations -> stored values (shifted for 16-bit formats)."""
    if storage == "f32":
        return f.astype(np.float32)
    shape = (-1,) + (1,) * (f.ndim - 1)
    return (f - w.reshape(shape)).astype(np.float32)
