# Repository standards baseline

Adapted from the shared reference at
`../ReAgent-ci-standards/.github/STANDARDS.md` (relative to this repository's root).
Reuse this structure for GitHub-published Python repositories.

| Area | Baseline | Phantasm implementation |
| --- | --- | --- |
| Triggers | Maintained branches, pull requests, manual runs | `.github/workflows/ci.yml` |
| Setup | Declared Python, isolated locked dependencies, cache | uv-managed Python 3.10/3.11/3.12; `uv.lock` |
| Validation | Blocking lint, format, tests, source/wheel validation | mise tasks; installed-artifact tests outside checkout |
| Permissions | Read-only validation | `contents: read` |
| Concurrency | Cancel obsolete validation | Workflow/ref group |
| Artifacts | Build once, record checksums | Distribution artifacts after tests |
| README | Purpose, installation, quick start, usage/configuration, development, license | Root README and training guide |
| CD | Optional; validate before publishing, protect credentials, verify provenance/publication | No CD or package publication configured |

Do not change dependency constraints merely to adopt this baseline. Validate
workflow syntax and run affected tasks before integration; require the remote CI
matrix before merging. Future CD must consume validated artifacts, use protected
credentials/environments as needed, and verify the published files against checksums.
