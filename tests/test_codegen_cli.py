import itertools

import pytest

from pylbm.cli import main
from pylbm.codegen import TARGETS, KernelSpec, source


@pytest.mark.parametrize("target", TARGETS)
def test_generated_sources_are_valid_python(target):
    for lat, coll, storage, pattern, force in itertools.product(
            ("D2Q9", "D3Q19"), ("bgk", "trt", "mrt"), ("f32", "f16"),
            ("ab", "aa-odd", "aa-even"), (False, True)):
        if lat == "D3Q19" and coll == "mrt":
            continue
        if target != "triton" and storage != "f32":
            continue
        d = 2 if lat == "D2Q9" else 3
        spec = KernelSpec(lat, (True,) + (False,) * (d - 1), True, force, coll, storage,
                          pattern)
        compile(source(spec, target), f"<{target}>", "exec")


def test_cli_backends_and_kernels(capsys):
    main(["backends"])
    out = capsys.readouterr().out
    assert "numpy/roll" in out
    main(["kernels", "--target", "numba", "--lattice", "D3Q19"])
    assert "lbm_kernel" in capsys.readouterr().out
