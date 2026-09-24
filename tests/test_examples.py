"""Smoke tests for the examples/ scripts (headless, tiny scenes)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("warp")
pytest.importorskip("viser")

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visualize_pointcloud_smoke():
    module = _load_example("visualize_pointcloud")
    summary = module.run_demo(num_envs=2, num_rays=64, frames=2, port=0, frame_dt=0.0)
    assert summary["frames"] == 2
    assert summary["hits"].sum() > 0
