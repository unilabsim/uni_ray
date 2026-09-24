"""Cold-path ``motrixsim.SceneModel`` → collision descriptor builder (#301).

This is the MotrixSim profile of the uni_ray adapter: it reads geom shapes,
sizes, local poses, link ownership, and hfield height data out of a loaded
``motrixsim.SceneModel`` (e.g. from ``motrixsim.load_model`` or
``load_mjcf_str``), converts them to plain NumPy on the cold path, and feeds
the shared, profile-neutral descriptor build in ``uni_ray.mjbatch``. The
result is the same immutable :class:`MjBatchCollision` — the caster, the
pose-sync contract, and the hot path are identical for all profiles. No
``motrixsim`` API is touched after descriptor construction; in particular
nothing Motrix-side runs on the ``update_pose``/``trace`` hot path.

Semantic mapping (MotrixSim → contract), verified by probes against
motrixsim-core 0.10.1 and pinned by tests/test_motrix.py:

- MotrixSim ``Shape`` values map onto MuJoCo-style primitives with the same
  size conventions (sphere radius, cuboid half extents, capsule/cylinder
  ``(radius, half_length)`` with a z-aligned axis, ellipsoid radii).
- Geom ``local_pose`` is ``(x, y, z, qx, qy, qz, qw)``; the contract uses
  wxyz quaternions, so the quat part is reordered here.
- MotrixSim links are the rigid bodies; the world is not a link. The
  descriptor numbers the world body 0 and link ``i`` as body ``i + 1``.
  Consumers read link poses via ``SceneModel.get_link_poses(data)``
  (``(num_links, 7)``, xyz + xyzw), prepend the identity world row, convert
  to wxyz, and pass them to ``update_pose`` — the pose-sync profile.
- Hfields are triangulated from MotrixSim's compiled ``HField.height_matrix``
  (meters, row 0 at y = −ry, col 0 at x = −rx — the same layout the shared
  triangulation consumes). Note MotrixSim normalizes elevation to
  ``(z − min) / (max − min) * z_scale`` where MuJoCo uses ``z * z_scale``
  raw, and its collision solid extends to a very deep z bound instead of
  MuJoCo's finite base box; using the compiled heights matches the Motrix
  collision world, with the remaining differences marked approximate in the
  capability audit (docs/backends.md).
- ``Shape.Mesh`` fails closed: motrixsim-core 0.10.1 exposes no mesh
  vertex/face introspection (only name/AABB), so a mesh descriptor cannot be
  built without silently changing the collision world. ``Shape.Plane``
  (finite) and ``Shape.Sdf`` have no contract equivalent and likewise fail
  closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from unisim.errors import UnsupportedCapabilityError
from unisim.optional import OptionalDependencyError

from .mjbatch import MjBatchCollision, _build_collision_description, _ModelSceneData

if TYPE_CHECKING:
    from unisim.ray_query import RayCaster

# MotrixSim Shape -> MuJoCo mjtGeom code (the shared descriptor build keys on
# the MuJoCo numbering). Verified against motrixsim-core 0.10.1.
_MOTRIX_SHAPE_TO_MJ_CODE = {
    "InfinitePlane": 0,
    "HField": 1,
    "Sphere": 2,
    "Capsule": 3,
    "Ellipsoid": 4,
    "Cylinder": 5,
    "Cuboid": 6,
}

_MOTRIX_UNSUPPORTED_SHAPES = {
    "Mesh": "motrixsim-core 0.10.1 exposes no mesh vertex/face introspection "
    "(only mesh_name/AABB), so a collision descriptor cannot be built without "
    "changing the collision world",
    "Plane": "MotrixSim finite planes have no contract equivalent; the contract "
    "plane is the infinite double-sided local z = 0 plane",
    "Sdf": "SDF geoms have no contract equivalent",
}

_XYZW_TO_WXYZ = [3, 0, 1, 2]


def _import_motrixsim() -> Any:
    try:
        import motrixsim
    except ImportError as error:
        raise OptionalDependencyError(
            "uni_ray.motrix requires the optional dependency 'motrixsim-core==0.10.1', "
            "which is not installed; it ships cp310-only wheels, so install the "
            "'uni-ray[motrix]' extra in a Python 3.10 environment (see docs/backends.md)"
        ) from error
    return motrixsim


def _shape_name(geom: Any) -> str:
    shape = geom.shape
    return getattr(shape, "name", None) or str(shape).split(".")[-1]


def build_collision_description(scene_model: Any) -> MjBatchCollision:
    """Read a ``motrixsim.SceneModel`` once and freeze it into a descriptor.

    Plane sizes are zeroed (contract infinite-plane semantics) and hfields
    are triangulated once from the compiled ``height_matrix`` — both via the
    shared profile-neutral build. Unsupported shapes (mesh, finite plane,
    SDF) fail closed with :class:`unisim.errors.UnsupportedCapabilityError`;
    nothing is silently dropped or replaced.
    """
    motrixsim = _import_motrixsim()
    if not isinstance(scene_model, motrixsim.SceneModel):
        raise TypeError(
            f"scene_model must be a motrixsim.SceneModel, got {type(scene_model).__name__}"
        )
    return _build_collision_description(_scene_data_from_motrix(scene_model))


def _scene_data_from_motrix(scene_model: Any) -> _ModelSceneData:
    """Snapshot the descriptor inputs out of a ``motrixsim.SceneModel``."""
    geoms = list(scene_model.geoms)
    num_geoms = len(geoms)
    geom_type_codes = np.zeros(num_geoms, dtype=np.int64)
    geom_sizes = np.zeros((num_geoms, 3))
    geom_pos = np.zeros((num_geoms, 3))
    geom_quat = np.zeros((num_geoms, 4))
    geom_body_ids = np.zeros(num_geoms, dtype=np.intp)
    geom_data_ids = np.full(num_geoms, -1, dtype=np.int64)

    unsupported: list[str] = []
    for geom in geoms:
        name = _shape_name(geom)
        if name in _MOTRIX_UNSUPPORTED_SHAPES:
            label = geom.name or f"#{geom.index}"
            unsupported.append(
                f"geom {label} ({name}): {_MOTRIX_UNSUPPORTED_SHAPES[name]}"
            )
            continue
        code = _MOTRIX_SHAPE_TO_MJ_CODE.get(name)
        if code is None:
            unsupported.append(f"geom #{geom.index} has unknown MotrixSim shape '{name}'")
            continue
        index = int(geom.index)
        geom_type_codes[index] = code
        geom_sizes[index] = np.asarray(geom.size, dtype=np.float64)[:3]
        local_pose = np.asarray(geom.local_pose, dtype=np.float64)
        geom_pos[index] = local_pose[:3]
        geom_quat[index] = local_pose[3:7][_XYZW_TO_WXYZ]  # Motrix xyzw -> contract wxyz
        link = geom.link
        # The world is not a link in MotrixSim; world geoms bind to body 0.
        geom_body_ids[index] = 0 if link is None else int(link.index) + 1
        if name == "HField":
            geom_data_ids[index] = int(geom.hfield.index)
    if unsupported:
        details = "; ".join(unsupported)
        raise UnsupportedCapabilityError(
            f"uni_ray motrix profile does not support: {details}. No geometry was "
            "dropped or substituted; remove the unsupported geoms or use a "
            "different profile"
        )

    # Hfields: feed the shared triangulation MotrixSim's compiled heights
    # (meters) with a unit z scale and no base box, and the x/y extents from
    # the hfield bound. All model hfields are snapshotted so geom_data_ids
    # (the raw HField.index) stays aligned with the slot arrays.
    num_hfields = int(scene_model.num_hfields)
    hfield_nrow = np.zeros(num_hfields, dtype=np.int64)
    hfield_ncol = np.zeros(num_hfields, dtype=np.int64)
    hfield_adr = np.zeros(num_hfields, dtype=np.int64)
    hfield_size = np.zeros((num_hfields, 4))
    hfield_data_parts: list[np.ndarray] = []
    for hfield_id in range(num_hfields):
        hfield = scene_model.get_hfield(hfield_id)
        heights = np.asarray(hfield.height_matrix, dtype=np.float64)
        nrow, ncol = int(hfield.nrow), int(hfield.ncol)
        if heights.shape != (nrow, ncol):
            raise UnsupportedCapabilityError(
                f"motrixsim hfield {hfield_id} height_matrix shape {heights.shape} "
                f"does not match ({nrow}, {ncol}); cannot build the collision world"
            )
        bound = np.asarray(hfield.bound, dtype=np.float64)
        hfield_nrow[hfield_id] = nrow
        hfield_ncol[hfield_id] = ncol
        hfield_adr[hfield_id] = sum(part.size for part in hfield_data_parts)
        hfield_size[hfield_id] = [
            0.5 * (bound[3] - bound[0]),
            0.5 * (bound[4] - bound[1]),
            1.0,  # heights are already in meters
            0.0,  # no finite base box in the MotrixSim solid
        ]
        hfield_data_parts.append(heights.reshape(-1))
    hfield_data = (
        np.concatenate(hfield_data_parts) if hfield_data_parts else np.zeros(0)
    )

    return _ModelSceneData(
        num_bodies=int(scene_model.num_links) + 1,
        geom_type_codes=geom_type_codes,
        geom_sizes=geom_sizes,
        geom_pos=geom_pos,
        geom_quat=geom_quat,
        geom_body_ids=geom_body_ids,
        geom_data_ids=geom_data_ids,
        mesh_vert_adr=np.zeros(0, dtype=np.int64),
        mesh_vert_num=np.zeros(0, dtype=np.int64),
        mesh_face_adr=np.zeros(0, dtype=np.int64),
        mesh_face_num=np.zeros(0, dtype=np.int64),
        mesh_vert=np.zeros((0, 3), dtype=np.float32),
        mesh_face=np.zeros(0, dtype=np.int32),
        hfield_nrow=hfield_nrow,
        hfield_ncol=hfield_ncol,
        hfield_adr=hfield_adr,
        hfield_size=hfield_size,
        hfield_data=hfield_data,
    )


def create_ray_caster(
    scene_model: Any,
    num_envs: int = 1,
    num_rays: int = 1,
    **kwargs: Any,
) -> RayCaster:
    """Build the descriptor from ``scene_model`` and return a bound caster.

    Convenience for the MotrixSim profile: equivalent to
    ``uni_ray.create_ray_caster(num_envs, num_rays, collision=
    build_collision_description(scene_model), **kwargs)`` followed by
    ``materialize``. The returned object is a plain
    :class:`unisim.ray_query.RayCaster`; body poses are the contract-specified
    identity until the first ``update_pose``.
    """
    import uni_ray

    collision = build_collision_description(scene_model)
    caster = uni_ray.create_ray_caster(
        num_envs=num_envs, num_rays=num_rays, collision=collision, **kwargs
    )
    caster.materialize(collision.scene)
    return caster


def rebuild_caster_from_motrix(caster: Any, scene_model: Any) -> MjBatchCollision:
    """Re-read an updated ``motrixsim.SceneModel`` and rebuild ``caster``'s scene.

    The MotrixSim-profile counterpart of
    ``uni_ray.mjbatch.rebuild_caster_from_model``: geometry randomization
    produces a fresh ``SceneModel``, which this binds through the explicit
    cold-path ``rebuild``. Never invoked by ``update_pose``/``trace``.
    """
    from .warp_caster import WarpRayCaster

    if not isinstance(caster, WarpRayCaster):
        raise TypeError(f"caster must be a WarpRayCaster, got {type(caster).__name__}")
    collision = build_collision_description(scene_model)
    caster.rebuild(collision)
    return collision


__all__ = [
    "build_collision_description",
    "create_ray_caster",
    "rebuild_caster_from_motrix",
]
