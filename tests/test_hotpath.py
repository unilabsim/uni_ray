"""Hot-path audits as tests (#299).

Proves that ``update_pose``/``trace`` never touch MuJoCo (no ``MjData``
access, no XML parsing), never rebuild the descriptor/BVH/meshes, and perform
zero new Warp allocations after warmup.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the hot-path audits")
wp = pytest.importorskip("warp", reason="warp-lang is required for the hot-path audits")

import uni_ray  # noqa: E402
from uni_ray.mjbatch import build_collision_description  # noqa: E402
from uni_ray.warp_caster import WarpRayCaster  # noqa: E402

AUDIT_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <geom type="mesh" mesh="tet" pos="3 0 0"/>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.3" pos="0.4 0 0"/>
      <geom type="capsule" size="0.05 0.2" pos="0 0.5 0"/>
    </body>
  </worldbody>
</mujoco>
"""


class _Spy:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._wrapped(*args, **kwargs)


def _make_caster():
    model = mujoco.MjModel.from_xml_string(AUDIT_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=64, collision=collision)
    caster.materialize(collision.scene)
    return caster, data


def _drive_hot_path(caster, data, iterations: int) -> None:
    origins = np.zeros((64, 3))
    directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (64, 1))
    for iteration in range(iterations):
        pos = np.array(data.xpos, copy=True)[None].repeat(2, axis=0)
        pos[:, 1, 2] += 0.001 * iteration
        caster.update_pose(pos, np.array(data.xquat, copy=True)[None].repeat(2, axis=0))
        caster.trace(origins, directions, max_distance=10.0)


def test_no_new_warp_allocations_or_bvh_mesh_rebuilds(monkeypatch) -> None:
    spies = {
        name: _Spy(getattr(wp, name)) for name in ("array", "zeros", "empty", "full")
    }
    for name, spy in spies.items():
        monkeypatch.setattr(wp, name, spy)
    bvh_spy = _Spy(wp.Bvh)
    mesh_spy = _Spy(wp.Mesh)
    monkeypatch.setattr(wp, "Bvh", bvh_spy)
    monkeypatch.setattr(wp, "Mesh", mesh_spy)
    rebuild_spy = _Spy(WarpRayCaster.rebuild)
    monkeypatch.setattr(WarpRayCaster, "rebuild", rebuild_spy)

    # Cold path: exactly one BVH and one wp.Mesh (the single tet mesh).
    caster, data = _make_caster()
    assert bvh_spy.calls == 1
    assert mesh_spy.calls == 1

    # Warmup covers lazy kernel compilation and any one-time refit scratch.
    _drive_hot_path(caster, data, iterations=3)
    for spy in (*spies.values(), bvh_spy, mesh_spy):
        spy.calls = 0

    _drive_hot_path(caster, data, iterations=25)
    for name, spy in {**spies, "Bvh": bvh_spy, "Mesh": mesh_spy}.items():
        assert spy.calls == 0, f"hot path called wp.{name} {spy.calls} times"
    assert rebuild_spy.calls == 0, "hot path invoked the cold-path rebuild"
    caster.close()


def test_no_mujoco_or_xml_or_descriptor_on_hot_path() -> None:
    """Subprocess: after materialize, mujoco is removed entirely and the
    descriptor builder is poisoned; the update_pose/trace loop must still run.
    """
    code = """
        import sys

        import numpy as np

        import mujoco
        import uni_ray
        import uni_ray.mjbatch
        from uni_ray.mjbatch import build_collision_description

        xml = '''
        <mujoco>
          <worldbody>
            <geom type="plane" size="0 0 0.1"/>
            <body pos="0 0 1.0">
              <freejoint/>
              <geom type="sphere" size="0.3"/>
            </body>
          </worldbody>
        </mujoco>
        '''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        collision = build_collision_description(model)
        caster = uni_ray.create_ray_caster(num_envs=1, num_rays=16, collision=collision)
        caster.materialize(collision.scene)

        # Cold path done. Remove mujoco entirely: any MjData access or XML
        # parsing on the hot path would now raise.
        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "mujoco":
                    raise ImportError("mujoco blocked on the hot path")
                return None

        sys.meta_path.insert(0, _Blocker())
        for module_name in [name for name in sys.modules if name.split(".")[0] == "mujoco"]:
            del sys.modules[module_name]
        del mujoco

        # Poison the descriptor builder: the hot path must not rebuild it.
        def _poisoned(*args, **kwargs):
            raise AssertionError("descriptor rebuilt on the hot path")

        uni_ray.mjbatch.build_collision_description = _poisoned

        pos = np.tile(data.xpos[None], (1, 1, 1))
        quat = np.tile(data.xquat[None], (1, 1, 1))
        del data
        origins = np.zeros((16, 3))
        directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (16, 1))
        for iteration in range(10):
            pos[0, 1, 2] = 1.0 + 0.01 * iteration
            caster.update_pose(pos, quat)
            result = caster.trace(origins, directions, max_distance=10.0)
        assert result.hit.any()
        caster.close()
        print("hotpath-clean-ok")
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "hotpath-clean-ok" in result.stdout
