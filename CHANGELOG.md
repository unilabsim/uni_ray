# Changelog

## 0.1.0 - 2026-10-26

- Initial PyPI release of `uniray`, the Warp-accelerated ray caster plugin for the UniSim ray-query contract.
- Ships the NumPy and Torch-native caster profiles, including shared/per-environment ray batches, caller-stream tensor execution, persistent tensor outputs, static filters, inclusive minimum distance, optional world-space normals, and hit/geom/body outputs.
- Includes primitive, static mesh, hfield, MuJoCo batch, MJWarp, and Motrix descriptor profiles where their optional runtimes are installed.
- Adds per-environment primitive sizes, local geom transforms, optional-slot enablement, and immutable mesh-asset selection without vertex streaming.
- Depends on `unisim-core>=2.2.0`; Warp, MuJoCo, Motrix, and visualization runtimes remain optional extras.
