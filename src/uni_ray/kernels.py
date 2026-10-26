# Warp trace/AABB kernels, copied from MuJoCo-LiDAR
# (https://github.com/discoverse-dev/MuJoCo-LiDAR),
# src/mujoco_lidar/core_warp/kernels.py, with adaptations for the UniSim
# ray-query contract:
#   * rays arrive as explicit world-space origin/direction arrays instead of a
#     per-ray theta/phi sensor pattern;
#   * environment rows are selected through an env_map indirection instead of
#     launching one thread block per physical env;
#   * the trace kernel also reports the hit flag and nearest geom index,
#     applies a static geom mask and inclusive minimum distance, optionally
#     writes an against-ray normal, and writes distance == max_distance on
#     misses (the UniSim miss convention).
# compute_oriented_box_aabb, update_aabbs_batch_kernel, read_mat33_batch,
# and trace_geom are otherwise verbatim, including the
# infinite-plane AABB substitution bugfix. write_body_poses_kernel and
# compute_geom_poses_kernel are new (pose sync per mj_kinematics semantics);
# ray_mesh_hit and trace_geom_normal are UniSim rich-query additions.
# Copyright (c) 2025 Yufei Jia, MIT License. See NOTICE.
import warp as wp

from .geometry import (
    ray_box_distance,
    ray_capsule_distance,
    ray_cylinder_distance,
    ray_ellipsoid_distance,
    ray_plane_distance,
    ray_sphere_distance,
    transform_ray_to_local,
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
    geom_sizes: wp.array2d(dtype=wp.vec3),
    geom_mesh_ids: wp.array2d(dtype=wp.int32),
    mesh_aabb_center: wp.array(dtype=wp.vec3),
    mesh_aabb_size: wp.array(dtype=wp.vec3),
    geom_enabled: wp.array2d(dtype=wp.bool),
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
    if geom_type < 0 or not geom_enabled[env_id, geom_id]:
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
        plane_x = 1000.0
        plane_y = 1000.0
        lower, upper = compute_oriented_box_aabb(
            pos,
            wp.vec3(plane_x, plane_y, 1.0e-3),
            rot,
        )
    else:
        size = geom_sizes[env_id, geom_id]
        if geom_type == 1 or geom_type == 7:
            mesh_index = geom_mesh_ids[env_id, geom_id]
            if mesh_index < 0:
                lower = wp.vec3(0.0, 0.0, 0.0)
                upper = wp.vec3(0.0, 0.0, 0.0)
            else:
                center = pos + rot * mesh_aabb_center[mesh_index]
                lower, upper = compute_oriented_box_aabb(center, mesh_aabb_size[mesh_index], rot)
        elif geom_type == 2:
            radius = size[0]
            lower, upper = compute_oriented_box_aabb(pos, wp.vec3(radius, radius, radius), rot)
        elif geom_type == 6 or geom_type == 4:
            lower, upper = compute_oriented_box_aabb(pos, size, rot)
        elif geom_type == 5:
            radius = size[0]
            lower, upper = compute_oriented_box_aabb(pos, wp.vec3(radius, radius, size[1]), rot)
        elif geom_type == 3:
            radius = size[0]
            lower, upper = compute_oriented_box_aabb(
                pos, wp.vec3(radius, radius, size[1] + radius), rot
            )

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
        t = ray_plane_distance(ray_origin, ray_dir, center, wp.vec3(0.0), rot)
    elif geom_type == 2:
        t = ray_sphere_distance(ray_origin, ray_dir, center, size[0])
    elif geom_type == 3:
        t = ray_capsule_distance(
            ray_origin,
            ray_dir,
            center,
            wp.vec3(size[0], size[0], size[1]),
            rot,
        )
    elif geom_type == 4:
        t = ray_ellipsoid_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 5:
        t = ray_cylinder_distance(
            ray_origin,
            ray_dir,
            center,
            wp.vec3(size[0], size[0], size[1]),
            rot,
        )
    elif geom_type == 6:
        t = ray_box_distance(ray_origin, ray_dir, center, size, rot)
    return t


@wp.func
def trace_geom_normal(
    geom_type: int,
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
    hit_distance: float,
):
    ro, _rd = transform_ray_to_local(ray_origin, ray_dir, center, rot)
    point = ro + hit_distance * _rd
    local_normal = wp.vec3(0.0, 0.0, 1.0)
    if geom_type == 2:
        local_normal = point / size[0]
    elif geom_type == 6:
        axis = 0
        best = -1.0
        for candidate in range(3):
            ratio = wp.abs(point[candidate]) / size[candidate]
            if ratio > best:
                best = ratio
                axis = candidate
        sign = wp.sign(point[axis])
        if axis == 0:
            local_normal = wp.vec3(sign, 0.0, 0.0)
        elif axis == 1:
            local_normal = wp.vec3(0.0, sign, 0.0)
        else:
            local_normal = wp.vec3(0.0, 0.0, sign)
    elif geom_type == 5:
        radius = wp.sqrt(point[0] * point[0] + point[1] * point[1])
        side_error = wp.abs(radius - size[0])
        cap_error = wp.abs(wp.abs(point[2]) - size[1])
        if cap_error <= side_error:
            local_normal = wp.vec3(0.0, 0.0, wp.sign(point[2]))
        else:
            local_normal = wp.vec3(point[0], point[1], 0.0) / size[0]
    elif geom_type == 3:
        if wp.abs(point[2]) < size[1]:
            local_normal = wp.vec3(point[0], point[1], 0.0) / size[0]
        else:
            cap_center = wp.vec3(0.0, 0.0, size[1] * wp.sign(point[2]))
            local_normal = (point - cap_center) / size[0]
    elif geom_type == 4:
        gradient = wp.vec3(
            point[0] / (size[0] * size[0]),
            point[1] / (size[1] * size[1]),
            point[2] / (size[2] * size[2]),
        )
        length = wp.length(gradient)
        if length > 1.0e-12:
            local_normal = gradient / length
    world_normal = rot * local_normal
    if wp.dot(world_normal, ray_dir) > 0.0:
        world_normal = -world_normal
    return world_normal


@wp.func
def ray_mesh_hit(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    rot: wp.mat33,
    min_distance: float,
    max_distance: float,
):
    rot_t = wp.transpose(rot)
    local_origin = rot_t * (ray_origin - center)
    local_dir = wp.normalize(rot_t * ray_dir)
    advanced_origin = local_origin + min_distance * local_dir
    query = wp.mesh_query_ray(mesh_id, advanced_origin, local_dir, max_distance - min_distance)
    t = -1.0
    normal = wp.vec3(0.0, 0.0, 0.0)
    if query.result:
        t = min_distance + query.t
        normal = rot * query.normal
        if wp.dot(normal, ray_dir) > 0.0:
            normal = -normal
    return t, normal


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
def write_body_poses_from_tensors_kernel(
    body_xpos: wp.array2d(dtype=wp.vec3),
    body_xquat: wp.array2d(dtype=wp.quat),
    pos_w: wp.array3d(dtype=wp.float32),
    quat_w: wp.array3d(dtype=wp.float32),
):
    """Alias public full-batch wxyz pose tensors into caster-owned pose state.

    ``pos_w`` and ``quat_w`` have the exact public ``(env, body, component)``
    layout. The quaternion conversion is deliberately performed by this
    device kernel; reinterpreting the public float32 tensor as ``wp.quat``
    would swap its wxyz components for Warp's xyzw representation.
    """
    env_id, body_id = wp.tid()
    body_xpos[env_id, body_id] = wp.vec3(
        pos_w[env_id, body_id, 0],
        pos_w[env_id, body_id, 1],
        pos_w[env_id, body_id, 2],
    )
    body_xquat[env_id, body_id] = wp.quat(
        quat_w[env_id, body_id, 1],
        quat_w[env_id, body_id, 2],
        quat_w[env_id, body_id, 3],
        quat_w[env_id, body_id, 0],
    )


@wp.kernel
def write_geometry_kernel(
    env_map: wp.array(dtype=wp.int32),
    sizes: wp.array2d(dtype=wp.vec3),
    local_pos: wp.array2d(dtype=wp.vec3),
    local_quat: wp.array2d(dtype=wp.vec4),
    enabled: wp.array2d(dtype=wp.bool),
    mesh_ids: wp.array2d(dtype=wp.int32),
    sizes_out: wp.array2d(dtype=wp.vec3),
    local_pos_out: wp.array2d(dtype=wp.vec3),
    local_quat_out: wp.array2d(dtype=wp.vec4),
    enabled_out: wp.array2d(dtype=wp.bool),
    mesh_ids_out: wp.array2d(dtype=wp.int32),
):
    """Scatter validated per-row geometry into caster-owned batched state."""
    row, geom_id = wp.tid()
    env_id = env_map[row]
    sizes_out[env_id, geom_id] = sizes[row, geom_id]
    local_pos_out[env_id, geom_id] = local_pos[row, geom_id]
    local_quat_out[env_id, geom_id] = local_quat[row, geom_id]
    enabled_out[env_id, geom_id] = enabled[row, geom_id]
    mesh_ids_out[env_id, geom_id] = mesh_ids[row, geom_id]


@wp.kernel
def write_env_map_and_validate_ids_kernel(
    env_ids: wp.array(dtype=wp.int64),
    num_envs: int,
    seen: wp.array(dtype=wp.int32),
    env_map: wp.array(dtype=wp.int32),
    validation_error: wp.array(dtype=wp.int32),
):
    """Convert, range-check, and de-duplicate a public environment selection.

    ``seen`` is a bounded caster-owned scratch mask. It is zeroed by a preceding
    launch, so duplicate detection remains device-side and never needs host
    readback on the hot path.
    """
    row = wp.tid()
    env_id = env_ids[row]
    if env_id < 0 or env_id >= num_envs:
        validation_error[0] = 1
        return
    if wp.atomic_add(seen, int(env_id), 1) != 0:
        validation_error[0] = 1
        return
    env_map[row] = int(env_id)


@wp.kernel
def zero_int32_kernel(values: wp.array(dtype=wp.int32)):
    """Zero one integer buffer on the caller's current stream."""
    index = wp.tid()
    values[index] = 0


@wp.kernel
def fill_ray_pattern_kernel(
    ray_origins: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    shared_origins: wp.array(dtype=wp.vec3),
    shared_directions: wp.array(dtype=wp.vec3),
):
    """Fan out the current shared Torch ray values into caster-owned staging."""
    env_id, ray_id = wp.tid()
    ray_origins[env_id, ray_id] = shared_origins[ray_id]
    ray_directions[env_id, ray_id] = shared_directions[ray_id]


@wp.kernel
def fill_empty_trace_kernel(
    distances: wp.array2d(dtype=wp.float32),
    hits: wp.array2d(dtype=wp.bool),
    hit_geoms: wp.array2d(dtype=wp.int32),
    hit_points: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.float32),
    body_ids: wp.array2d(dtype=wp.int32),
    max_distance: float,
    write_hit_point: int,
    write_normal: int,
    write_body_id: int,
):
    """Materialize the all-miss convention without a host readback."""
    row, ray_id = wp.tid()
    distances[row, ray_id] = max_distance
    hits[row, ray_id] = False
    hit_geoms[row, ray_id] = -1
    if write_hit_point == 1:
        hit_points[row, ray_id, 0] = 0.0
        hit_points[row, ray_id, 1] = 0.0
        hit_points[row, ray_id, 2] = 0.0
    if write_normal == 1:
        normals[row, ray_id, 0] = 0.0
        normals[row, ray_id, 1] = 0.0
        normals[row, ray_id, 2] = 0.0
    if write_body_id == 1:
        body_ids[row, ray_id] = -1


@wp.kernel
def write_tensor_trace_outputs_kernel(
    ray_origins: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    distances: wp.array2d(dtype=wp.float32),
    hits: wp.array2d(dtype=wp.bool),
    hit_geoms: wp.array2d(dtype=wp.int32),
    geom_body_ids: wp.array(dtype=wp.int32),
    hit_points: wp.array3d(dtype=wp.float32),
    body_ids: wp.array2d(dtype=wp.int32),
    write_hit_point: int,
    write_body_id: int,
):
    """Compute optional tensor outputs on the trace stream."""
    row, ray_id = wp.tid()
    geom_id = hit_geoms[row, ray_id]
    hit = hits[row, ray_id]
    if write_hit_point == 1:
        distance = distances[row, ray_id]
        origin = ray_origins[row, ray_id]
        direction = ray_directions[row, ray_id]
        hit_points[row, ray_id, 0] = origin[0] + distance * direction[0]
        hit_points[row, ray_id, 1] = origin[1] + distance * direction[1]
        hit_points[row, ray_id, 2] = origin[2] + distance * direction[2]
    if write_body_id == 1:
        body_ids[row, ray_id] = geom_body_ids[geom_id] if hit else -1


@wp.kernel
def compute_geom_poses_kernel(
    env_map: wp.array(dtype=wp.int32),
    geom_body_ids: wp.array(dtype=wp.int32),
    geom_local_pos: wp.array2d(dtype=wp.vec3),
    geom_local_quat: wp.array2d(dtype=wp.vec4),
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
    local_quat_wxyz = geom_local_quat[env_id, geom_id]
    local_quat = wp.quat(
        local_quat_wxyz[1],
        local_quat_wxyz[2],
        local_quat_wxyz[3],
        local_quat_wxyz[0],
    )
    geom_xpos[env_id, geom_id] = body_xpos[env_id, body_id] + wp.quat_rotate(
        body_quat, geom_local_pos[env_id, geom_id]
    )
    rot = wp.quat_to_matrix(body_quat) * wp.quat_to_matrix(local_quat)
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
    geom_sizes: wp.array2d(dtype=wp.vec3),
    geom_mesh_ids: wp.array2d(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_enabled: wp.array2d(dtype=wp.bool),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array3d(dtype=wp.float32),
    geom_allowed: wp.array(dtype=wp.int32),
    env_map: wp.array(dtype=wp.int32),
    ray_origins: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    max_distance: float,
    min_distance: float,
    distances: wp.array2d(dtype=wp.float32),
    hits: wp.array2d(dtype=wp.bool),
    hit_geoms: wp.array2d(dtype=wp.int32),
    normals: wp.array3d(dtype=wp.float32),
    write_normal: int,
):
    row, ray_id = wp.tid()
    env_id = env_map[row]
    origin = ray_origins[row, ray_id]
    ray_dir = ray_directions[row, ray_id]
    ngeom = geom_types.shape[0]

    best = max_distance
    best_geom = int(-1)
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    root = wp.bvh_get_group_root(bvh_id, env_id)
    query = wp.bvh_query_ray(bvh_id, origin, ray_dir, root)
    flat_id = int(0)
    while wp.bvh_query_next(query, flat_id):
        geom_id = flat_id - env_id * ngeom
        if (
            geom_id >= 0
            and geom_id < ngeom
            and geom_allowed[geom_id] == 1
            and geom_enabled[env_id, geom_id]
        ):
            geom_type = geom_types[geom_id]
            if geom_type >= 0:
                t = -1.0
                normal = wp.vec3(0.0, 0.0, 0.0)
                geom_rot = read_mat33_batch(geom_xmat, env_id, geom_id)
                if geom_type == 1 or geom_type == 7:
                    mesh_idx = geom_mesh_ids[env_id, geom_id]
                    if mesh_idx >= 0:
                        t, normal = ray_mesh_hit(
                            mesh_ids[mesh_idx],
                            origin,
                            ray_dir,
                            geom_xpos[env_id, geom_id],
                            geom_rot,
                            min_distance,
                            max_distance,
                        )
                else:
                    t = trace_geom(
                        geom_type,
                        origin,
                        ray_dir,
                        geom_xpos[env_id, geom_id],
                        geom_sizes[env_id, geom_id],
                        geom_rot,
                    )
                    if write_normal == 1 and t >= min_distance:
                        normal = trace_geom_normal(
                            geom_type,
                            origin,
                            ray_dir,
                            geom_xpos[env_id, geom_id],
                            geom_sizes[env_id, geom_id],
                            geom_rot,
                            t,
                        )
                if (t >= min_distance and t < best) or (
                    t == best and best_geom >= 0 and geom_id < best_geom
                ):
                    best = t
                    best_geom = geom_id
                    best_normal = normal

    if best_geom >= 0:
        distances[row, ray_id] = best
        hits[row, ray_id] = True
        hit_geoms[row, ray_id] = best_geom
    else:
        distances[row, ray_id] = max_distance
        hits[row, ray_id] = False
        hit_geoms[row, ray_id] = -1
    if write_normal == 1:
        normals[row, ray_id, 0] = best_normal[0]
        normals[row, ray_id, 1] = best_normal[1]
        normals[row, ray_id, 2] = best_normal[2]
