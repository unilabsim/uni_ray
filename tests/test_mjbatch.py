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


def test_hfield_geom_is_triangulated_into_a_static_mesh() -> None:
    xml = """
    <mujoco>
      <asset>
        <hfield name="hf" nrow="3" ncol="2" size="2 1 0.5 0.1"
                elevation="0 1  0.5 0.5  1 0"/>
      </asset>
      <worldbody>
        <geom type="hfield" hfield="hf" pos="1 0 0.2"/>
        <geom type="hfield" hfield="hf" pos="8 0 0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    collision = build_collision_description(model)

    # Hfields bind as static meshes, deduplicated by (geom type, data id).
    assert collision.scene.geom_types == (RayGeomType.MESH, RayGeomType.MESH)
    np.testing.assert_array_equal(collision.geom_mesh_ids, [0, 0])
    assert len(collision.meshes) == 1
    mesh = collision.meshes[0]
    # Top surface: 6 grid vertices (ported triangulation), then skirt-wall and
    # base-box vertices for the closed mj_ray solid.
    assert mesh.points.shape == (6 + 24 + 8, 3)
    # 4 top triangles + 12 skirt + 12 base box.
    assert mesh.indices.shape == ((4 + 12 + 12) * 3,)
    # Geom-local frame: x in [-2, 2] (ncol), y in [-1, 1] (nrow),
    # z = elevation * 0.5; the base box reaches z = -0.1 (size[3]). MuJoCo
    # compiles the XML elevation text with the first row at +y (the compiled
    # hfield_data row 0 is y = -1), so the corner heights are flipped
    # vertically relative to the XML text.
    np.testing.assert_allclose(
        mesh.points[:6],
        [
            [-2.0, -1.0, 0.5],
            [2.0, -1.0, 0.0],
            [-2.0, 0.0, 0.25],
            [2.0, 0.0, 0.25],
            [-2.0, 1.0, 0.0],
            [2.0, 1.0, 0.5],
        ],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        mesh.indices[:12], [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]
    )
    assert mesh.points[:, 2].min() == pytest.approx(-0.1)
    assert mesh.points[:, 0].min() >= -2.0 and mesh.points[:, 0].max() <= 2.0
    assert mesh.points[:, 1].min() >= -1.0 and mesh.points[:, 1].max() <= 1.0


def test_unknown_geom_code_fails_closed() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>"
    )
    model.geom_type[0] = 42  # not a real mjtGeom code; exercises the fail-closed path
    with pytest.raises(UnsupportedCapabilityError, match="geom type codes"):
        build_collision_description(model)


def test_non_mjmodel_input_rejected() -> None:
    with pytest.raises(TypeError, match="MjModel"):
        build_collision_description(object())
