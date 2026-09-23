# Warp trace/AABB kernels, copied from MuJoCo-LiDAR
# (https://github.com/discoverse-dev/MuJoCo-LiDAR),
# src/mujoco_lidar/core_warp/kernels.py, with adaptations for the UniSim
# ray-query contract:
#   * rays arrive as explicit world-space origin/direction arrays instead of a
#     per-ray theta/phi sensor pattern;
#   * environment rows are selected through an env_map indirection instead of
#     launching one thread block per physical env;
#   * the trace kernel also reports the hit flag and nearest geom index and
#     writes distance == max_distance on misses (the UniSim miss convention).
# compute_oriented_box_aabb, update_aabbs_batch_kernel, read_mat33_batch,
# trace_geom, and ray_mesh_distance are otherwise verbatim, including the
# infinite-plane AABB substitution bugfix. write_body_poses_kernel and
# compute_geom_poses_kernel are new (pose sync per mj_kinematics semantics).
# Copyright (c) 2025 Yufei Jia, MIT License. See NOTICE.
import warp as wp

from .geometry import (
    ray_box_distance,
    ray_capsule_distance,
    ray_cylinder_distance,
    ray_ellipsoid_distance,
    ray_plane_distance,
    ray_sphere_distance,
)


@wp.func
def compute_oriented_box_aabb(center: wp.vec3, size: wp.vec3, rot: wp.mat33):
    extent = wp.vec3(
        wp.abs(rot[0, 0]) * size[0] + wp.abs(rot[0, 1]) * size[1] + wp.abs(rot[0, 2]) * size[2],
        wp.abs(rot[1, 0]) * size[0] + wp.abs(rot[1, 1]) * size[1] + wp.abs(rot[1, 2]) * size[2],
        wp.abs(rot[2, 0]) * size[0] + wp.abs(rot[2, 1]) * size[1] + wp.abs(rot[2, 2]) * size[2],
    )
    return center - extent, center + extent


@wp.kernel
def update_aabbs_batch_kernel(
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_aabb_center: wp.array(dtype=wp.vec3),
    geom_aabb_size: wp.array(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array3d(dtype=wp.float32),
    env_map: wp.array(dtype=wp.int32),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    row, geom_id = wp.tid()
    env_id = env_map[row]
    flat_id = env_id * geom_types.shape[0] + geom_id
    geom_type = geom_types[geom_id]
    rot = read_mat33_batch(geom_xmat, env_id, geom_id)
    pos = geom_xpos[env_id, geom_id]

    lower = wp.vec3(1.0e10, 1.0e10, 1.0e10)
    upper = wp.vec3(-1.0e10, -1.0e10, -1.0e10)
    if geom_type < 0:
        lower = wp.vec3(0.0, 0.0, 0.0)
        upper = wp.vec3(0.0, 0.0, 0.0)
    elif geom_type == 0:
        # MuJoCo's "infinite" plane convention is size[0] == size[1] == 0.0 (the
        # intersection test in ray_plane_distance already special-cases this as
        # unbounded). Using that literal 0 as the AABB half-extent instead makes
        # the plane's broad-phase bounding box razor-thin, so the BVH culls
        # almost every ray before the intersection test ever runs - the ground
        # becomes effectively invisible while finite geoms (legs, boxes) are
        # unaffected. Substitute a large-but-finite half-extent for the AABB
        # only, so the BVH actually considers rays anywhere near the ground.
        plane_x = geom_sizes[geom_id][0]
        plane_y = geom_sizes[geom_id][1]
        if plane_x <= 0.0:
            plane_x = 1000.0
        if plane_y <= 0.0:
            plane_y = 1000.0
        lower, upper = compute_oriented_box_aabb(
            pos,
            wp.vec3(plane_x, plane_y, 1.0e-3),
            rot,
        )
    else:
        aabb_center = pos + rot * geom_aabb_center[geom_id]
        lower, upper = compute_oriented_box_aabb(aabb_center, geom_aabb_size[geom_id], rot)

    eps = wp.vec3(1.0e-4, 1.0e-4, 1.0e-4)
    lowers[flat_id] = lower - eps
    uppers[flat_id] = upper + eps


@wp.func
def read_mat33_batch(mats: wp.array3d(dtype=wp.float32), env_id: int, geom_id: int):
    return wp.mat33(
        mats[env_id, geom_id, 0],
        mats[env_id, geom_id, 1],
        mats[env_id, geom_id, 2],
        mats[env_id, geom_id, 3],
        mats[env_id, geom_id, 4],
        mats[env_id, geom_id, 5],
        mats[env_id, geom_id, 6],
        mats[env_id, geom_id, 7],
        mats[env_id, geom_id, 8],
    )


@wp.func
def trace_geom(
    geom_type: int,
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
):
    t = -1.0
    if geom_type == 0:
        t = ray_plane_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 2:
        t = ray_sphere_distance(ray_origin, ray_dir, center, size[0])
    elif geom_type == 3:
        t = ray_capsule_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 4:
        t = ray_ellipsoid_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 5:
        t = ray_cylinder_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 6:
        t = ray_box_distance(ray_origin, ray_dir, center, size, rot)
    return t


@wp.func
def ray_mesh_distance(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    rot: wp.mat33,
    max_t: float,
):
    rot_t = wp.transpose(rot)
    local_origin = rot_t * (ray_origin - center)
    local_dir = wp.normalize(rot_t * ray_dir)
    query = wp.mesh_query_ray(mesh_id, local_origin, local_dir, max_t)
    t = -1.0
    if query.result:
        t = query.t
    return t


@wp.kernel
def write_body_poses_kernel(
    env_map: wp.array(dtype=wp.int32),
    src_pos: wp.array2d(dtype=wp.vec3),
    src_quat: wp.array2d(dtype=wp.quat),
    body_xpos: wp.array2d(dtype=wp.vec3),
    body_xquat: wp.array2d(dtype=wp.quat),
):
    """Scatter validated body pose rows into the persistent batched pose state."""
    row, body_id = wp.tid()
    env_id = env_map[row]
    body_xpos[env_id, body_id] = src_pos[row, body_id]
    body_xquat[env_id, body_id] = src_quat[row, body_id]


@wp.kernel
def compute_geom_poses_kernel(
    env_map: wp.array(dtype=wp.int32),
    geom_body_ids: wp.array(dtype=wp.int32),
    geom_local_pos: wp.array(dtype=wp.vec3),
    geom_local_quat: wp.array(dtype=wp.quat),
    body_xpos: wp.array2d(dtype=wp.vec3),
    body_xquat: wp.array2d(dtype=wp.quat),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array3d(dtype=wp.float32),
):
    """Rebuild geom world poses from body poses (MuJoCo mj_kinematics semantics).

    geom_xpos[g] = body_xpos[b] + quat_rotate(body_quat[b], geom_local_pos[g])
    geom_xmat[g] = quat_to_mat(body_quat[b]) * quat_to_mat(geom_local_quat[g])
    """
    row, geom_id = wp.tid()
    env_id = env_map[row]
    body_id = geom_body_ids[geom_id]
    body_quat = body_xquat[env_id, body_id]
    geom_xpos[env_id, geom_id] = body_xpos[env_id, body_id] + wp.quat_rotate(
        body_quat, geom_local_pos[geom_id]
    )
    rot = wp.quat_to_matrix(body_quat) * wp.quat_to_matrix(geom_local_quat[geom_id])
    geom_xmat[env_id, geom_id, 0] = rot[0, 0]
    geom_xmat[env_id, geom_id, 1] = rot[0, 1]
    geom_xmat[env_id, geom_id, 2] = rot[0, 2]
    geom_xmat[env_id, geom_id, 3] = rot[1, 0]
    geom_xmat[env_id, geom_id, 4] = rot[1, 1]
    geom_xmat[env_id, geom_id, 5] = rot[1, 2]
    geom_xmat[env_id, geom_id, 6] = rot[2, 0]
    geom_xmat[env_id, geom_id, 7] = rot[2, 1]
    geom_xmat[env_id, geom_id, 8] = rot[2, 2]


@wp.kernel
def trace_rays_batch_bvh_kernel(
    bvh_id: wp.uint64,
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_mesh_ids: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array3d(dtype=wp.float32),
    env_map: wp.array(dtype=wp.int32),
    ray_origins: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    max_distance: float,
    distances: wp.array2d(dtype=wp.float32),
    hits: wp.array2d(dtype=wp.bool),
    hit_geoms: wp.array2d(dtype=wp.int32),
):
    row, ray_id = wp.tid()
    env_id = env_map[row]
    origin = ray_origins[row, ray_id]
    ray_dir = ray_directions[row, ray_id]
    ngeom = geom_types.shape[0]

    best = max_distance
    best_geom = int(-1)
    root = wp.bvh_get_group_root(bvh_id, env_id)
    query = wp.bvh_query_ray(bvh_id, origin, ray_dir, root)
    flat_id = int(0)
    while wp.bvh_query_next(query, flat_id):
        geom_id = flat_id - env_id * ngeom
        if geom_id >= 0 and geom_id < ngeom:
            geom_type = geom_types[geom_id]
            if geom_type >= 0:
                t = -1.0
                geom_rot = read_mat33_batch(geom_xmat, env_id, geom_id)
                if geom_type == 1 or geom_type == 7:
                    mesh_idx = geom_mesh_ids[geom_id]
                    if mesh_idx >= 0:
                        t = ray_mesh_distance(
                            mesh_ids[mesh_idx],
                            origin,
                            ray_dir,
                            geom_xpos[env_id, geom_id],
                            geom_rot,
                            best,
                        )
                else:
                    t = trace_geom(
                        geom_type,
                        origin,
                        ray_dir,
                        geom_xpos[env_id, geom_id],
                        geom_sizes[geom_id],
                        geom_rot,
                    )
                if t >= 0.0 and t < best:
                    best = t
                    best_geom = geom_id

    if best_geom >= 0:
        distances[row, ray_id] = best
        hits[row, ray_id] = True
        hit_geoms[row, ray_id] = best_geom
    else:
        distances[row, ray_id] = max_distance
        hits[row, ray_id] = False
        hit_geoms[row, ray_id] = -1
