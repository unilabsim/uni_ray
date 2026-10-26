"""Torch-native caster lifecycle, validation, parity, and audit tests (#462 M3/M5)."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from unisim.errors import BackendError
from unisim.ray_query import (
    RayPoseTensorView,
    RaySceneDescription,
    RayTraceFilter,
    RayTraceOutputs,
    RayTraceTensorResult,
)

wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")
torch = pytest.importorskip("torch", reason="torch is required for the tensor profile")

import uni_ray  # noqa: E402


@pytest.fixture()
def scene() -> RaySceneDescription:
    return RaySceneDescription(
        num_bodies=2,
        geom_types=("plane", "sphere"),
        geom_sizes=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        geom_local_pos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0, 1]),
        body_names=("world", "ball"),
    )


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _public_device() -> Any:
    """Represent the device object a cross-package factory caller may use."""
    if not torch.cuda.is_available():
        return "cpu"
    return torch.device("cuda", index=torch.cuda.current_device())


def _expected_close(actual: Any, expected: Any, **kwargs: Any) -> None:
    torch.testing.assert_close(actual, expected.to(device=actual.device), **kwargs)


@pytest.fixture()
def caster(scene: RaySceneDescription) -> Any:
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=3, device=_device())
    caster.materialize(scene)
    caster.bind_pose_layout(scene.body_names)
    yield caster
    caster.close()


def _pose(caster: Any, *, packet: bool = False) -> RayPoseTensorView:
    names = caster._require_tensor_layout()
    device = caster._required_tensor_device()
    if packet:
        packed = torch.zeros((caster.num_envs, 7 * len(names)), dtype=torch.float32, device=device)
        pos = packed[:, : 3 * len(names)].reshape(caster.num_envs, len(names), 3)
        quat = packed[:, 3 * len(names) : 7 * len(names)].reshape(caster.num_envs, len(names), 4)
    else:
        pos = torch.zeros((caster.num_envs, len(names), 3), dtype=torch.float32, device=device)
        quat = torch.zeros((caster.num_envs, len(names), 4), dtype=torch.float32, device=device)
    quat[..., 0] = 1.0
    pos[:, 1, 2] = 5.0
    return RayPoseTensorView(body_names=names, pos_w=pos, quat_w=quat)


def _rays(device: str) -> tuple[Any, Any]:
    origins = torch.tensor(
        [[0.0, 0.0, 2.0], [4.0, 4.0, 2.0], [0.0, 0.0, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    directions = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return origins, directions


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_capabilities_and_import_boundary() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import uni_ray.warp_caster; assert 'torch' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1)
    capabilities = caster.get_ray_capabilities()
    assert capabilities.supports_device_output
    assert capabilities.supports_tensor_pose_sync
    assert capabilities.supports_tensor_rays
    assert capabilities.torch_devices == ("cpu", "cuda")
    assert capabilities.stream_event_ownership == "caller"
    assert capabilities.supports_host_readback
    caster.close()


def test_factory_accepts_public_torch_device_object(scene: RaySceneDescription) -> None:
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=3, device=_public_device())
    try:
        caster.materialize(scene)
        caster.bind_pose_layout(scene.body_names)
        caster.update_pose_tensor(_pose(caster))
        origins, directions = _rays(caster._required_tensor_device())
        result = caster.trace_tensor(origins, directions, 3.0)
        assert result.distance.shape == (2, 3)
    finally:
        caster.close()


def test_pose_layout_and_packet_strided_views(caster: Any) -> None:
    pose = _pose(caster, packet=True)
    assert pose.pos_w.stride() == (14, 3, 1)
    assert pose.quat_w.stride() == (14, 4, 1)
    caster.update_pose_tensor(pose)
    origins, directions = _rays(pose.pos_w.device)
    result = caster.trace_tensor(
        origins,
        directions,
        10.0,
        outputs=RayTraceOutputs(hit_point=True, geom_id=True, body_id=True),
    )
    _synchronize()
    assert isinstance(result, RayTraceTensorResult)
    _expected_close(result.distance[0], torch.tensor([2.0, 2.0, 3.5]))
    assert result.body_id[1].tolist() == [0, 0, 1]


def test_pose_storage_can_be_replaced_after_trace(caster: Any) -> None:
    """Per-frame callers may replace pose storage independently of result views."""
    caster.update_pose_tensor(_pose(caster))
    origins, directions = _rays(caster._required_tensor_device())
    result = caster.trace_tensor(origins, directions, 10.0)
    _synchronize()
    next_pose = _pose(caster)
    next_pose.pos_w[:, 1, 2] = 0.0
    caster.update_pose_tensor(next_pose)
    next_result = caster.trace_tensor(origins, directions, 10.0)
    _synchronize()
    # The sphere's local offset moves with the body, so its origin pose still
    # gives the original ray hits; this pins that replacement took effect when
    # compared with the third packet pose below.
    _expected_close(next_result.distance[0], torch.tensor([0.5, 2.0, 10.0]))
    assert result.distance is next_result.distance
    third_pose = _pose(caster, packet=True)
    caster.update_pose_tensor(third_pose)
    again = caster.trace_tensor(origins, directions, 10.0)
    _synchronize()
    _expected_close(again.distance[0], torch.tensor([2.0, 2.0, 3.5]))


def test_layout_and_pose_validation_fail_closed(caster: Any, scene: RaySceneDescription) -> None:
    with pytest.raises(BackendError, match="already bound"):
        caster.bind_pose_layout(("world", "ball"))
    fresh = uni_ray.create_ray_caster(num_envs=2, num_rays=3, device=_device())
    fresh.materialize(scene)
    with pytest.raises(ValueError, match="scene body layout"):
        fresh.bind_pose_layout(("ball", "world"))
    with pytest.raises(BackendError, match="bind_pose_layout"):
        fresh.trace_tensor(*_rays(_device()), 2.0)
    fresh.close()

    wrong_names = RayPoseTensorView(("other",), torch.zeros((2, 1, 3)), torch.zeros((2, 1, 4)))
    with pytest.raises(ValueError, match="bound layout"):
        caster.update_pose_tensor(wrong_names)
    good = _pose(caster)
    with pytest.raises(TypeError, match="float32"):
        caster.update_pose_tensor(
            RayPoseTensorView(
                good.body_names,
                good.pos_w.double(),
                good.quat_w,
            )
        )
    with pytest.raises(ValueError, match="shape"):
        caster.update_pose_tensor(
            RayPoseTensorView(good.body_names, good.pos_w[:1], good.quat_w[:1])
        )
    if torch.cuda.is_available():
        cpu_pose = RayPoseTensorView(good.body_names, good.pos_w.cpu(), good.quat_w.cpu())
        with pytest.raises(ValueError, match="must live on cuda:0"):
            caster.update_pose_tensor(cpu_pose)
    # Whole-tensor contiguity is not required, but innermost stride and injectivity are.
    storage = torch.empty((2, 3 * 4), dtype=torch.float32, device=good.pos_w.device)
    transposed = torch.as_strided(storage, (2, 2, 3), (8, 4, 2))
    assert transposed.stride()[-1] != 1
    with pytest.raises(ValueError, match="innermost stride"):
        caster.update_pose_tensor(RayPoseTensorView(good.body_names, transposed, good.quat_w))
    overlap = torch.zeros((2, 2, 3), dtype=torch.float32, device=good.pos_w.device)
    overlapped = torch.as_strided(overlap, (2, 2, 3), (3, 3, 1))
    with pytest.raises(ValueError, match="non-overlapping"):
        caster.update_pose_tensor(RayPoseTensorView(good.body_names, overlapped, good.quat_w))


def test_trace_validation_and_hidden_contiguous_is_rejected(caster: Any) -> None:
    caster.update_pose_tensor(_pose(caster))
    origins, directions = _rays(caster._required_tensor_device())
    with pytest.raises(ValueError, match="contiguous"):
        caster.trace_tensor(origins.t(), directions.t(), 2.0)
    with pytest.raises(TypeError, match="int64"):
        caster.trace_tensor(
            origins,
            directions,
            2.0,
            env_ids=torch.tensor([0], dtype=torch.int32),
        )
    normal_result = caster.trace_tensor(
        origins, directions, 2.0, outputs=RayTraceOutputs(normal=True)
    )
    assert normal_result.normal is not None
    with pytest.raises(ValueError, match="does not match"):
        caster.trace_tensor(origins[1:], directions, 2.0)
    with pytest.raises(ValueError, match="one of shapes"):
        caster.trace_tensor(
            origins[:1],
            directions[:1],
            2.0,
            env_ids=torch.tensor([0, 1], device=origins.device),
        )

    contiguous = torch.Tensor.contiguous
    tensor_cpu = torch.Tensor.cpu

    def forbidden(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("hidden Tensor.contiguous() on tensor path")

    def forbidden_cpu(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("hidden Tensor.cpu() host readback on tensor hot path")

    try:
        torch.Tensor.contiguous = forbidden
        torch.Tensor.cpu = forbidden_cpu
        repeated_pose = _pose(caster)
        caster.update_pose_tensor(repeated_pose)
        caster.trace_tensor(origins, directions, 10.0)
        per_env_origins = origins.unsqueeze(0).repeat(2, 1, 1)[1:]
        per_env_directions = directions.unsqueeze(0).repeat(2, 1, 1)[1:]
        caster.trace_tensor(
            per_env_origins,
            per_env_directions,
            10.0,
            env_ids=torch.tensor([1], dtype=torch.int64, device=origins.device),
        )
    finally:
        torch.Tensor.contiguous = contiguous
        torch.Tensor.cpu = tensor_cpu
    _synchronize()


def test_minimal_tensor_output_omits_internal_geom_scratch(caster: Any) -> None:
    caster.update_pose_tensor(_pose(caster))
    origins, directions = _rays(caster._required_tensor_device())
    result = caster.trace_tensor(origins, directions, 10.0)
    selected = caster.trace_tensor(
        origins.unsqueeze(0).repeat(2, 1, 1)[1:],
        directions.unsqueeze(0).repeat(2, 1, 1)[1:],
        10.0,
        env_ids=torch.tensor([1], dtype=torch.int64, device=origins.device),
    )
    _synchronize()
    assert result.geom_id is None
    assert selected.geom_id is None
    assert result.distance.shape == (2, 3)
    assert selected.distance.shape == (1, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream ordering required")
def test_first_use_static_filter_follows_caller_stream(scene: RaySceneDescription) -> None:
    local = uni_ray.create_ray_caster(num_envs=2, num_rays=3, device="cuda:0")
    local.materialize(scene)
    local.bind_pose_layout(scene.body_names)
    local.update_pose_tensor(_pose(local))
    origins, directions = _rays("cuda:0")
    stream = torch.cuda.Stream(device=origins.device)
    with torch.cuda.stream(stream):
        producer = origins * 1.0
        directions = directions * 1.0
        result = local.trace_tensor(
            producer,
            directions,
            10.0,
            min_distance=0.1,
            ray_filter=RayTraceFilter(allowed_geom_ids=(0, 1)),
        )
        consumer = result.distance * 2.0
    stream.synchronize()
    assert torch.isfinite(consumer).all()
    local.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA deferred validation required")
def test_cuda_env_id_validation_is_deferred_without_hot_path_readback(
    scene: RaySceneDescription,
) -> None:
    """CUDA IDs remain device-resident; the bounded marker is checked only on close."""
    local = uni_ray.create_ray_caster(num_envs=2, num_rays=3, device="cuda:0")
    local.materialize(scene)
    local.bind_pose_layout(scene.body_names)
    local.update_pose_tensor(_pose(local))
    origins, directions = _rays("cuda:0")
    tensor_cpu = torch.Tensor.cpu

    def forbidden_cpu(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("hidden Tensor.cpu() host readback on tensor hot path")

    try:
        torch.Tensor.cpu = forbidden_cpu
        local.trace_tensor(
            origins,
            directions,
            10.0,
            env_ids=torch.tensor([2, 2], dtype=torch.int64, device="cuda:0"),
        )
    finally:
        torch.Tensor.cpu = tensor_cpu
    with pytest.raises(ValueError, match="prior tensor trace detected an invalid ID"):
        local.close()


def test_shared_ray_pattern_is_reuploaded_each_call(caster: Any) -> None:
    """Shared staging is persistent, but its values follow the caller's tensors."""
    caster.update_pose_tensor(_pose(caster))
    origins, directions = _rays(caster._required_tensor_device())
    first = caster.trace_tensor(origins, directions, 10.0)
    changed_origins = directions.clone()
    changed_origins[:, 2] = 1.0  # Start each ray one unit higher.
    second = caster.trace_tensor(changed_origins, directions, 10.0)
    _synchronize()
    assert first.distance is second.distance
    _expected_close(second.distance[0], torch.tensor([1.0, 1.0, 4.5]))


def test_shared_per_env_selected_and_result_reuse(caster: Any) -> None:
    caster.update_pose_tensor(_pose(caster))
    origins, directions = _rays(caster._required_tensor_device())
    outputs = RayTraceOutputs(hit_point=True, normal=True, geom_id=True, body_id=True)
    full = caster.trace_tensor(origins, directions, 10.0, outputs=outputs)
    selected_ids = torch.tensor([1], dtype=torch.int64, device=origins.device)
    selected = caster.trace_tensor(
        origins.unsqueeze(0).repeat(2, 1, 1)[1:],
        directions.unsqueeze(0).repeat(2, 1, 1)[1:],
        10.0,
        env_ids=selected_ids,
        outputs=outputs,
    )
    assert selected.distance.shape == (1, 3)
    assert selected.distance.data_ptr() != full.distance.data_ptr()
    first = selected.distance
    again = caster.trace_tensor(
        origins.unsqueeze(0).repeat(2, 1, 1)[1:],
        directions.unsqueeze(0).repeat(2, 1, 1)[1:],
        5.0,
        env_ids=selected_ids,
        outputs=outputs,
    )
    assert first is again.distance
    _expected_close(again.distance, torch.tensor([[2.0, 2.0, 3.5]]))
    empty_ids = torch.empty((0,), dtype=torch.int64, device=origins.device)
    empty = caster.trace_tensor(
        origins.unsqueeze(0).repeat(0, 1, 1),
        directions.unsqueeze(0).repeat(0, 1, 1),
        3.0,
        env_ids=empty_ids,
    )
    assert empty.distance.shape == (0, 3)
    _synchronize()


def test_tensor_host_parity_and_allocation_audit(
    caster: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    origins, directions = _rays(caster._required_tensor_device())
    outputs = RayTraceOutputs(hit_point=True, normal=True, geom_id=True, body_id=True)
    ray_filter = RayTraceFilter(allowed_geom_ids=(0, 1))

    # ``wp.from_torch`` and zero-copy ``wp.array`` wrappers are metadata objects;
    # spy on the storage-allocating constructors instead.
    allocation_names = ("zeros", "empty", "full", "from_numpy")
    allocation_spies = {name: _AllocationSpy(getattr(wp, name)) for name in allocation_names}
    for name, spy in allocation_spies.items():
        monkeypatch.setattr(wp, name, spy)
    # Warm up lazy Warp kernel compilation and one-time output/staging profiles.
    caster.update_pose_tensor(_pose(caster))
    tensor_result = caster.trace_tensor(
        origins,
        directions,
        7.0,
        outputs=outputs,
        min_distance=0.1,
        ray_filter=ray_filter,
    )
    _synchronize()
    steady_pose = _pose(caster)
    for name, spy in allocation_spies.items():
        spy.calls = 0
    for _ in range(3):
        caster.update_pose_tensor(steady_pose)
        tensor_result = caster.trace_tensor(
            origins,
            directions,
            7.0,
            outputs=outputs,
            min_distance=0.1,
            ray_filter=ray_filter,
        )
    _synchronize()
    for name, spy in allocation_spies.items():
        assert spy.calls == 0, f"tensor hot path called wp.{name} {spy.calls} times"
    monkeypatch.undo()

    host_result = caster.trace(
        origins.detach().cpu().numpy(),
        directions.detach().cpu().numpy(),
        7.0,
        outputs=outputs,
        min_distance=0.1,
        ray_filter=ray_filter,
    )
    _synchronize()
    torch.testing.assert_close(
        tensor_result.distance, torch.as_tensor(host_result.distance).to(origins.device)
    )
    _expected_close(tensor_result.hit, torch.as_tensor(host_result.hit))
    torch.testing.assert_close(
        tensor_result.hit_point,
        torch.as_tensor(host_result.hit_point, dtype=torch.float32).to(origins.device),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        tensor_result.normal,
        torch.as_tensor(host_result.normal, dtype=torch.float32).to(origins.device),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        tensor_result.geom_id,
        torch.as_tensor(host_result.geom_id, dtype=torch.int32).to(origins.device),
    )
    torch.testing.assert_close(
        tensor_result.body_id,
        torch.as_tensor(host_result.body_id, dtype=torch.int32).to(origins.device),
    )

    # CUDA memory stats do not distinguish allocator/cache effects sufficiently for a
    # strict bytes assertion here. Pin the allocation-audit invariant structurally: the
    # second identical profile returns exactly the same caster-owned storages.
    steady = caster.trace_tensor(
        origins,
        directions,
        7.0,
        outputs=outputs,
        min_distance=0.1,
        ray_filter=ray_filter,
    )
    for name in ("distance", "hit", "hit_point", "normal", "geom_id", "body_id"):
        assert getattr(steady, name) is getattr(tensor_result, name), name


class _AllocationSpy:
    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.wrapped(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream ordering required")
def test_caller_stream_ordering(caster: Any) -> None:
    caster.update_pose_tensor(_pose(caster))
    origins, directions = _rays(caster._required_tensor_device())
    stream = torch.cuda.Stream(device=origins.device)
    with torch.cuda.stream(stream):
        producer = origins * 1.0
        directions = directions * 1.0
        result = caster.trace_tensor(producer, directions, 10.0)
        consumer = result.distance * 2.0
    stream.synchronize()
    assert torch.isfinite(consumer).all()
    _expected_close(result.distance[0], torch.tensor([2.0, 2.0, 3.5]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer audit required")
def test_cuda_tensor_trace_has_no_hidden_transfers_or_synchronizations(
    scene: RaySceneDescription, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the steady-state tensor profile device-resident and asynchronous."""
    local = uni_ray.create_ray_caster(num_envs=2, num_rays=3, device="cuda:0")
    local.materialize(scene)
    local.bind_pose_layout(scene.body_names)
    pose = _pose(local, packet=True)
    origins, directions = _rays("cuda:0")
    outputs = RayTraceOutputs(hit_point=True, normal=True, geom_id=True, body_id=True)
    ray_filter = RayTraceFilter(allowed_geom_ids=(0, 1))
    local.update_pose_tensor(pose)
    local.trace_tensor(
        origins,
        directions,
        7.0,
        outputs=outputs,
        min_distance=0.1,
        ray_filter=ray_filter,
    )
    torch.cuda.synchronize()

    copy_spy = _AllocationSpy(wp.copy)
    warp_sync_spy = _AllocationSpy(wp.synchronize_device)
    torch_sync_spy = _AllocationSpy(torch.cuda.synchronize)
    tensor_cpu_spy = _AllocationSpy(torch.Tensor.cpu)
    tensor_to_spy = _AllocationSpy(torch.Tensor.to)
    monkeypatch.setattr(wp, "copy", copy_spy)
    monkeypatch.setattr(wp, "synchronize_device", warp_sync_spy)
    monkeypatch.setattr(torch.cuda, "synchronize", torch_sync_spy)
    monkeypatch.setattr(torch.Tensor, "cpu", tensor_cpu_spy)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to_spy)

    for _ in range(3):
        local.update_pose_tensor(pose)
        local.trace_tensor(
            origins,
            directions,
            7.0,
            outputs=outputs,
            min_distance=0.1,
            ray_filter=ray_filter,
        )

    monkeypatch.undo()
    torch.cuda.synchronize()
    for name, spy in (
        ("wp.copy", copy_spy),
        ("wp.synchronize_device", warp_sync_spy),
        ("torch.cuda.synchronize", torch_sync_spy),
        ("Tensor.cpu", tensor_cpu_spy),
        ("Tensor.to", tensor_to_spy),
    ):
        assert spy.calls == 0, f"tensor hot path invoked {name} {spy.calls} times"
    local.close()


def test_close_invalidates_tensor_profile(caster: Any) -> None:
    caster.close()
    with pytest.raises(BackendError, match="closed"):
        caster.bind_pose_layout(("world", "ball"))
    with pytest.raises(BackendError, match="closed"):
        caster.trace_tensor(*_rays(_device()), 2.0)
