# Release runbook

`uniray` is published as the pure-Python distribution `uniray`. Production
publishing is automated by `.github/workflows/release.yml`: a matching `v*`
tag first triggers version, prior-CI, package, metadata, pure-wheel, sdist,
wheel-install, and CPU Warp smoke checks. Publishing then uses GitHub trusted
publishing from the protected `pypi` environment. A successful publish creates
the GitHub Release and attaches the source distribution and wheel.

The repository administrator must configure the PyPI trusted publisher once
with owner `unilabsim`, repository `uniray`, workflow `release.yml`, and
environment `pypi`. The GitHub environment must be named `pypi` and remain
protected.

Release procedure:

1. Confirm `CHANGELOG.md` has a section matching `project.version`.
2. Confirm `make check` passes locally.
3. Merge the release change and wait for `ci.yml` on the exact main commit.
4. Create and push an annotated matching tag:

   ```bash
   git tag -a v0.1.1 -m "release: uniray 0.1.1"
   git push origin v0.1.1
   ```

5. Inspect the release workflow, PyPI artifacts, and GitHub Release.

Never overwrite a published version. If an artifact is wrong, fix the source
and publish a new version.
