"""Import-boundary tests: uni_ray must import and fail closed without warp/mujoco.

Each check runs in a fresh subprocess with a meta-path blocker that makes
``warp`` and ``mujoco`` unimportable, mirroring an environment where the
optional runtimes are not installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCKER = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"warp", "mujoco"}:
            # Mirrors a genuinely missing package (ModuleNotFoundError), which
            # is also what pytest.importorskip keys on.
            raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
        return None


sys.meta_path.insert(0, _Blocker())
"""

# Variant that blocks only mujoco_warp (#300): the mjwarp adapter module must
# import and fail closed without it, and uni_ray/factory discovery must be
# unaffected.
BLOCKER_MJWARP = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"mujoco_warp"}:
            raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
        return None


sys.meta_path.insert(0, _Blocker())
"""

# Variant that blocks only motrixsim (#301): the motrix adapter module must
# import and fail closed without it, and uni_ray/factory discovery must be
# unaffected.
BLOCKER_MOTRIX = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"motrixsim"}:
            raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run(code: str, blocker: str = BLOCKER) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", blocker + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_uni_ray_without_optional_runtimes() -> None:
    result = _run(
        """
        import sys

        import uni_ray

        assert "warp" not in sys.modules
        assert "mujoco" not in sys.modules
        assert callable(uni_ray.create_ray_caster)
        import uni_ray.mjbatch

        print("import-ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout


def test_caster_creation_fails_closed_without_warp() -> None:
    result = _run(
        """
        from unisim.optional import OptionalDependencyError

        import uni_ray

        try:
            uni_ray.create_ray_caster(num_envs=1, num_rays=1)
        except OptionalDependencyError as error:
            assert "warp-lang" in str(error), str(error)
            print("fail-closed-ok")
        else:
            raise AssertionError("create_ray_caster should fail closed without warp")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "fail-closed-ok" in result.stdout


def test_unisim_factory_discovery_fails_closed_without_warp() -> None:
    result = _run(
        """
        from unisim.factory import create_ray_caster
        from unisim.optional import OptionalDependencyError

        try:
            create_ray_caster("uni_ray", num_envs=1, num_rays=1)
        except OptionalDependencyError as error:
            assert "warp-lang" in str(error), str(error)
            print("factory-fail-closed-ok")
        else:
            raise AssertionError("factory creation should fail closed without warp")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "factory-fail-closed-ok" in result.stdout


def test_mjbatch_builder_fails_closed_without_mujoco() -> None:
    result = _run(
        """
        from unisim.optional import OptionalDependencyError

        from uni_ray.mjbatch import build_collision_description

        try:
            build_collision_description(object())
        except OptionalDependencyError as error:
            assert "mujoco" in str(error), str(error)
            print("mjbatch-fail-closed-ok")
        else:
            raise AssertionError("build_collision_description should fail closed without mujoco")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "mjbatch-fail-closed-ok" in result.stdout


def test_mjwarp_module_imports_and_factory_works_without_mujoco_warp() -> None:
    result = _run(
        """
        import sys

        import uni_ray
        import uni_ray.mjwarp

        assert "mujoco_warp" not in sys.modules
        from unisim.factory import create_ray_caster

        caster = create_ray_caster("uni_ray", num_envs=1, num_rays=1)
        caster.close()
        print("import-and-factory-ok")
        """,
        blocker=BLOCKER_MJWARP,
    )
    assert result.returncode == 0, result.stderr
    assert "import-and-factory-ok" in result.stdout


def test_mjwarp_builder_fails_closed_without_mujoco_warp() -> None:
    result = _run(
        """
        from unisim.optional import OptionalDependencyError

        from uni_ray.mjwarp import build_collision_description

        try:
            build_collision_description(object())
        except OptionalDependencyError as error:
            assert "mujoco-warp" in str(error), str(error)
            print("mjwarp-fail-closed-ok")
        else:
            raise AssertionError(
                "mjwarp build_collision_description should fail closed without mujoco_warp"
            )
        """,
        blocker=BLOCKER_MJWARP,
    )
    assert result.returncode == 0, result.stderr
    assert "mjwarp-fail-closed-ok" in result.stdout


def test_motrix_module_imports_and_factory_works_without_motrixsim() -> None:
    result = _run(
        """
        import sys

        import uni_ray
        import uni_ray.motrix

        assert "motrixsim" not in sys.modules
        from unisim.factory import create_ray_caster

        caster = create_ray_caster("uni_ray", num_envs=1, num_rays=1)
        caster.close()
        print("import-and-factory-ok")
        """,
        blocker=BLOCKER_MOTRIX,
    )
    assert result.returncode == 0, result.stderr
    assert "import-and-factory-ok" in result.stdout


def test_motrix_builder_fails_closed_without_motrixsim() -> None:
    result = _run(
        """
        from unisim.optional import OptionalDependencyError

        from uni_ray.motrix import build_collision_description

        try:
            build_collision_description(object())
        except OptionalDependencyError as error:
            assert "motrixsim-core" in str(error), str(error)
            print("motrix-fail-closed-ok")
        else:
            raise AssertionError(
                "motrix build_collision_description should fail closed without motrixsim"
            )
        """,
        blocker=BLOCKER_MOTRIX,
    )
    assert result.returncode == 0, result.stderr
    assert "motrix-fail-closed-ok" in result.stdout
