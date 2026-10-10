"""CPU-only checks of the head-to-head harness (bench/counterparts.py); no GPU, no clones."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cp():
    path = ROOT / "bench" / "counterparts.py"
    spec = importlib.util.spec_from_file_location("counterparts", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["counterparts"] = mod  # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


def test_minimal_traffic_model(cp) -> None:
    r = cp._record("X", "v", "D2Q9", "fp32", "p", [8, 8], [1000.0, 1100.0, 900.0], 10, 360.0)
    assert r["bytes_per_unit"] == 72 and r["median"] == 1000.0
    assert r["effective_GBs"] == pytest.approx(72.0)
    assert r["pct_copy_bw"] == pytest.approx(20.0)
    assert cp._record("X", "v", "D3Q19", "fp16s", "p", [4] * 3, [1.0], 1, 1.0)[
        "bytes_per_unit"] == 76


def test_fluidx3d_alteration(cp) -> None:
    defines = "\n".join(["//#define D2Q9 // a", "#define D3Q19 // b", "//#define D3Q15",
                         "//#define D3Q27", "#define SRT", "//#define TRT", "#define FP16S // c",
                         "//#define FP16C", "//#define BENCHMARK // d"])
    out = cp._fx_defines(defines, "D2Q9", "fp32")
    assert out.startswith("// ALTERED SOURCE")
    assert "\n#define D2Q9 // a" in out and "//#define D3Q19 // b" in out
    assert "//#define FP16S" in out and "\n#define BENCHMARK // d" in out
    setup = ('#include "setup.hpp"\n#ifdef BENCHMARK\nvoid main_setup() { // benchmark; x\n}\n'
             "#endif // BENCHMARK\nrest\n")
    new = cp._fx_setup_cpp(setup)
    assert "CP_RESULT" in new and "void main_setup() { // benchmark" not in new
    assert new.rstrip().endswith("rest")
