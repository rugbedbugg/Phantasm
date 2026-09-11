# Contributing to Phantasm

Thanks for taking the time. This is a local fine-tuning pipeline — contributions that improve robustness, support new export formats, or reduce setup friction are most welcome.

## Dev setup

```bash
mise run install
```

This installs all dependencies via `uv` into the project's virtual environment.

## Running tests

```bash
uv run --locked --extra dev pytest
```

All tests must pass before opening a PR. Tests use synthetic data and mocked external services.

## Linting

```bash
uv run --locked --extra dev ruff check .
```

Fix any reported issues before committing. Auto-fix with `uv run --locked --extra dev ruff check . --fix` for safe transformations.

## Branch naming

```
title/work-being-done-in-short
```

Examples: `fix/burst-grouping-edge-case`, `feat/markdown-export`, `docs/update-readme`.

## Commit format

```
[Title]: Imperative single-line subject
```

Examples:
- `[Fix]: Handle empty transcript files in parser`
- `[Feat]: Add plaintext export format`
- `[Docs]: Update contributing guide`

All commits must be signed with the repository-configured mechanism (`git commit -S`).

## Pull requests

- Keep PRs focused. One concern per PR.
- Include a short description of what changed and why.
- Reference any relevant issue numbers.
- CI must be green before requesting review.

## What not to include

- Real Discord tokens, user IDs, or message content.
- Trained model weights or GGUF files.
- Output files (`outputs/`, `checkpoints/`, `*export*.json`, `*parsed*.json`).

All of the above are gitignored. Keep it that way.

## Validation baseline

Run `mise run lint`, `mise run test`, `mise run build`, and `mise run check-dist`.
The last command tests installed artifacts outside the source checkout. Follow
[repository standards](.github/STANDARDS.md) and [GPU validation](docs/TRAINING.md)
for changes to CI, packaging, or training. The GPU lock is resolved separately from
the lightweight CLI lock; update training dependencies together and record hardware
validation before claiming GPU support.
