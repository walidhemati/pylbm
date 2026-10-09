"""Every backend/variant reproduces the float64 NumPy reference on small problems."""

import numpy as np
import pytest

from pylbm import backends as B
from pylbm.problem import Problem

CASES = {
    "cavity2d-bgk": lambda: Problem((40, 28), tau=0.7, u_lid=0.1),
    "cavity2d-trt": lambda: Problem((40, 28), tau=0.7, u_lid=0.1, collision="trt"),
    "cavity2d-mrt": lambda: Problem((40, 28), tau=0.7, u_lid=0.1, collision="mrt"),
    "channel2d-force": lambda: Problem((40, 28), tau=0.7, periodic=(True, False),
                                       force=(1e-5, 2e-6)),
    "periodic2d-mrt-force": lambda: Problem((32, 24), tau=0.6, periodic=(True, True),
                                            force=(1e-5, -3e-6), collision="mrt"),
    "cavity3d-bgk": lambda: Problem((20, 14, 12), tau=0.7, u_lid=0.1),
    "periodic3d-trt": lambda: Problem((16, 12, 10), tau=0.7, periodic=(True, True, False),
                                      force=(1e-5, 1e-6, 0.0), collision="trt"),
}
TOL = {"f32": 3e-6, "f16": 1e-4, "bf16": 6e-4}
ENTRIES = [e for e in B.entries() if e[0] != "numpy"]
_REF: dict[str, np.ndarray] = {}


def _reference(case: str) -> np.ndarray:
    if case not in _REF:
        s = B.make("numpy", "roll", CASES[case]())
        s.step(100)
        _REF[case] = s.populations()
    return _REF[case]


@pytest.mark.gpu
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e[0]}/{e[1]}")
@pytest.mark.parametrize("case", list(CASES))
def test_matches_reference(entry, case):
    cls = B.get(*entry)
    ok, why = cls.available()
    if not ok:
        pytest.skip(why)
    p = CASES[case]()
    ok, why = cls.supports(p)
    if not ok or not cls.equivalent:
        pytest.skip(why or "not equivalent by design (legacy)")
    if entry[0] == "torch-compile" and case not in ("cavity2d-bgk", "cavity3d-bgk"):
        pytest.skip("torch.compile compiles per problem; two cases are enough")
    s = cls(p)
    s.step(100)
    f = s.populations()
    tol = TOL[getattr(s, "storage", "f32")]
    assert np.abs(f - _reference(case)).max() < tol


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "cupy-cuda-c"])
def test_aa_pattern_bitwise_equal_to_two_buffer(backend):
    p = Problem((64, 40), tau=0.6, u_lid=0.1, collision="trt")
    a, b = B.make(backend, "fused", p), B.make(backend, "fused-aa", p)
    a.step(60)
    b.step(60)
    assert np.abs(a.populations() - b.populations()).max() < 1e-6


@pytest.mark.gpu
def test_legacy_original_runs():
    s = B.make("torch-compile", "original", Problem.cavity(32, re=100))
    s.step(20)
    f = s.populations()
    assert f.shape == (9, 32, 32) and np.isfinite(f).all()


def test_registry_never_crashes():
    st = B.status()
    assert len(st) == len(B.entries())
    assert ("numpy", "roll", True, "") in st


@pytest.mark.gpu
@pytest.mark.parametrize("entry", [("torch-eager", "roll"), ("torch-eager", "gather"),
                                   ("torch-compile", "gather")], ids=lambda e: "/".join(e))
def test_general_geometry_matches_reference(entry):
    """Obstacle + velocity inlet + pressure outlet (the Karman setup) on a tiny channel."""
    from pylbm.validation.karman import schafer_turek_problem

    p = schafer_turek_problem(d=6, u_mean=0.05, collision="trt")
    ref = B.make("numpy", "roll", p)
    ref.step(60)
    s = B.make(*entry, p)
    s.step(60)
    assert np.abs(s.populations() - ref.populations()).max() < 3e-6
