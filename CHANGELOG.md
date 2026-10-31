# Changelog

## 0.1.1 - 2026-10-31

- Renamed the Python namespace and import package to `uniray`, including the source directory, build module, workflows, tests, examples, and documentation.
- Moved the MuJoCo, MJWarp, and MotrixSim cold-path adapters under `uniray.backends` and updated every public import path.
- Added the adapted UniSim contract as the `third_party/unisim` submodule for local and CI testing.
- Depends on `unisim-core>=2.3.0`, which registers the `uniray` ray-caster identity.

## 0.1.0 - 2026-10-26

- Initial PyPI release of `uniray`, the Warp-accelerated ray caster plugin for the UniSim ray-query contract.
- Ships the NumPy and Torch-native caster profiles, including shared/per-environment ray batches, caller-stream tensor execution, persistent tensor outputs, static filters, inclusive minimum distance, optional world-space normals, and hit/geom/body outputs.
- Includes primitive, static mesh, hfield, MuJoCo batch, MJWarp, and Motrix descriptor profiles where their optional runtimes are installed.
- Adds per-environment primitive sizes, local geom transforms, optional-slot enablement, and immutable mesh-asset selection without vertex streaming.
- Depends on `unisim-core>=2.2.0`; Warp, MuJoCo, Motrix, and visualization runtimes remain optional extras.
