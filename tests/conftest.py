"""Shared pytest configuration: GPU tests are skipped without CUDA."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore", module="cupy")


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


HAS_CUDA = _has_cuda()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
