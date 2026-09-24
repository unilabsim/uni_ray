"""uni_ray: Warp-accelerated ray caster plugin for the UniSim ray-query contract.

The package is importable without the optional Warp and MuJoCo runtimes; they
are imported lazily, Warp at caster creation and MuJoCo inside the cold-path
``uni_ray.mjbatch`` descriptor builder. A missing runtime fails closed with
:class:`unisim.optional.OptionalDependencyError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from unisim.optional import OptionalDependencyError

if TYPE_CHECKING:
    from unisim.ray_query import RayCaster

__version__ = "0.1.0"

__all__ = ["__version__", "create_ray_caster"]


def create_ray_caster(num_envs: int = 1, num_rays: int = 1, **kwargs: Any) -> RayCaster:
    """Create the uni_ray Warp ray caster (the UniSim plugin entry point).

    The UniSim factory discovers this callable lazily through
    ``create_ray_caster("uni_ray", ...)``; Warp itself is imported here so a
    missing runtime fails closed at creation with an actionable diagnostic
    instead of at import time.
    """
    try:
        import warp  # noqa: F401
    except ImportError as error:
        raise OptionalDependencyError(
            "ray caster 'uni_ray' requires the optional dependency 'warp-lang>=1.11.0', "
            "which is not installed; install the 'uni-ray[warp]' extra to use this caster"
        ) from error
    from .warp_caster import WarpRayCaster

    return WarpRayCaster(num_envs=num_envs, num_rays=num_rays, **kwargs)
