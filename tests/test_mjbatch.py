"""Cold-path mjbatch builder tests (#298)."""

from __future__ import annotations

import numpy as np
import pytest
from unisim.errors import UnsupportedCapabilityError
from unisim.ray_query import RayGeomType

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the mjbatch builder")

from uni_ray.mjbatch import build_collision_description  # noqa: E402

SCENE_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <geom type="mesh" mesh="tet" pos="3 0 0"/>
    <geom type="mesh" mesh="tet" pos="6 0 0"/>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.3" pos="0.4 0 0"/>
      <geom type="capsule" size="0.05 0.2" pos="0 0.5 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_descriptor_matches_mjmodel() -> None:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    collision = build_collision_description(model)
    scene = collision.scene

    assert scene.num_bodies == model.nbody
    assert scene.geom_types == (
        RayGeomType.PLANE,
        RayGeomType.MESH,
        RayGeomType.MESH,
        RayGeomType.SPHERE,
        RayGeomType.CAPSULE,
    )
    np.testing.assert_array_equal(scene.geom_body_ids, model.geom_bodyid)
    np.testing.assert_allclose(scene.geom_local_pos, model.geom_pos)
    np.testing.assert_allclose(scene.geom_local_quat, model.geom_quat)
    # The plane size (a rendering grid in MuJoCo) is zeroed so the caster
    # honors the contract's infinite-plane semantics; capsule keeps the
    # MuJoCo-style (radius, half_length, 0) layout.
    np.testing.assert_array_equal(scene.geom_sizes[0], np.zeros(3))
    np.testing.assert_allclose(scene.geom_sizes[4], [0.05, 0.2, 0.0])


def test_meshes_are_deduplicated_by_data_id() -> None:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    collision = build_collision_description(model)

    assert len(collision.meshes) == 1
    mesh = collision.meshes[0]
    assert mesh.points.shape == (4, 3)
    assert mesh.indices.shape == (12,)
    np.testing.assert_array_equal(collision.geom_mesh_ids, [-1, 0, 0, -1, -1])
    # MuJoCo re-centers/re-orients inline mesh vertices (mesh_pos/mesh_quat
    # compensation); the descriptor must carry the compiled mesh_vert slice.
    vert_adr = int(model.mesh_vertadr[0])
    vert_num = int(model.mesh_vertnum[0])
    np.testing.assert_allclose(
        mesh.points, model.mesh_vert[vert_adr : vert_adr + vert_num], atol=1e-6
    )


def test_hfield_geom_fails_closed() -> None:
    xml = """
    <mujoco>
      <asset>
        <hfield name="hf" nrow="2" ncol="2" size="1 1 0.1 0.1"/>
      </asset>
      <worldbody>
        <geom type="hfield" hfield="hf"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    with pytest.raises(UnsupportedCapabilityError, match="hfield"):
        build_collision_description(model)


def test_non_mjmodel_input_rejected() -> None:
    with pytest.raises(TypeError, match="MjModel"):
        build_collision_description(object())
