# Phantasm

[![CI](https://img.shields.io/github/actions/workflow/status/rugbedbugg/Phantasm/ci.yml?branch=main&style=for-the-badge&labelColor=000000)](https://github.com/rugbedbugg/Phantasm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&labelColor=000000)](LICENSE)

Phantasm prepares conversational datasets, fine-tunes a persona with Unsloth
QLoRA, and chats with exported GGUF models locally. The Python package provides
`scrape`, `parse`, `format`, `train`, `colab`, `recover`, and `chat` commands.

Supply your own authorized chat exports and model access. This repository includes
only synthetic example messages; it does not distribute private archives or weights.

## Installation

The lightweight pipeline supports Python 3.10 and later; development defaults to
Python 3.11. [uv](https://docs.astral.sh/uv/) manages Python and isolated environments.

```bash
git clone https://github.com/rugbedbugg/Phantasm.git
cd Phantasm
uv sync --locked --extra dev --python 3.11
source .venv/bin/activate
phantasm --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. With optional
[mise](https://mise.jdx.dev/), use `mise trust` and `mise run install` instead.
Python compatibility and existing package version constraints are preserved.
The root `uv.lock` defines the lightweight environment; GPU training uses a separate
[locked environment](docs/TRAINING.md).

## Quick start

Run this example without Discord access or a GPU:

```bash
phantasm parse examples/demo.json --user-id 100 --target-id 200 -o parsed.json
phantasm format -i parsed.json -p dataset --val-split 0.5
phantasm train --dataset dataset_train_sharegpt.jsonl \
  --validation-dataset dataset_val_sharegpt.jsonl --dry-run
```

The formatter creates `dataset_train_sharegpt.jsonl` and
`dataset_val_sharegpt.jsonl`. The dry run validates the files and displays their
sample counts and SHA-256 fingerprints without loading model dependencies.

## Usage

### Import and select participants

Use an existing Discord export, or fetch a channel you can access:

```bash
phantasm scrape CHANNEL_ID -o raw_export.json
# On interruption or a later-page failure:
phantasm scrape CHANNEL_ID -o raw_export.json --resume
```

The scraper reads `DISCORD_TOKEN` from the environment, or prompts without echoing
when attached to a terminal. Provide the API authorization value (including the
`Bot ` prefix when using a bot). Tokens are not accepted as command-line arguments.
Noninteractive use requires the environment variable.

Each successful page is saved to `raw_export.json.checkpoint.json`. Failed downloads
exit nonzero and leave the last complete export untouched. Resume uses that
checkpoint's channel and cursor; a successful download removes the checkpoint.
A resume completes the original backward scan; run a fresh scrape to include newer
messages. HTTP 429, server errors, and connection failures receive bounded retries.

```bash
phantasm parse raw_export.json --user-id YOUR_AUTHOR_ID \
  --target-id PERSONA_AUTHOR_ID -o parsed.json
```

IDs remain stable across username changes. Additional participants cause an error
unless `--ignore-others` is supplied; excluded text messages break context so that
the selected speakers are not joined across another person's turn. Legacy
`phantasm parse raw_export.json YOUR_USERNAME` accepts only unambiguous two-person
exports. Attachment URLs and reply IDs are retained in parsed data; training uses
text only and does not resolve reply threads or attachment content.

### Build disjoint datasets

```bash
phantasm format -i parsed.json -p dataset --window 6 \
  --max-gap 300 --session-gap 1800 --val-split 0.05
```

Messages must be chronological. Timestamps normalize to UTC; missing offsets mean
UTC. Inactivity longer than `--session-gap` starts a new session. Crossing between
known and unknown timestamps also starts a session. Consecutive unknown timestamps
use input order within one session.

Whole usable sessions are partitioned chronologically before sliding windows are
created. `--val-split` is a proportion of sessions, so sample ratios may differ.
A positive split needs at least two usable sessions; use `--val-split 0` explicitly
for training-only data. The validation file is then replaced with an empty file,
preventing reuse of stale samples. Do not pass that empty file to training.

### Train and export

For Colab, Phantasm manages runtime creation, dataset transfer, the isolated GPU
environment, training, export, and verified result download:

```bash
phantasm colab setup
phantasm colab login
phantasm train --backend colab --job persona \
  --dataset dataset_train_sharegpt.jsonl \
  --validation-dataset dataset_val_sharegpt.jsonl
```

Google authentication and GPU availability still depend on your account. The command
waits for completion; add `--detach` to return after submission. Results go into
`.phantasm/jobs/persona/result/`. A runtime created by Phantasm is stopped after
successful download unless `--keep` is set. An existing `--session` is kept running.

For automatic checkpoint and result backups, set `PIXELDRAIN_API_KEY` in your
environment and add `--artifact-store pixeldrain`. After runtime loss, use
`phantasm recover persona` to download the latest backup, or
`phantasm recover persona --resume` to restore a checkpoint and continue on Colab.
Recovery uses the last successfully uploaded checkpoint; provisioning a replacement
runtime requires the explicit `--resume` command. See [Colab and recovery](docs/COLAB.md)
for setup, monitoring, transfer retries, storage limitations, and verification status.

For a local GPU, follow [GPU setup and verification](docs/TRAINING.md), then run:

```bash
.venv-training/bin/phantasm train --dataset dataset_train_sharegpt.jsonl \
  --validation-dataset dataset_val_sharegpt.jsonl \
  --max-steps 120 --output-dir phantasm_model --quant-method q4_k_m
```

A run saves adapters, checkpoints, metrics and dataset/configuration fingerprints in
`run.json`, and GGUF files under `phantasm_model/gguf/`. With validation, the lowest
validation-loss checkpoint is restored before export. Use a new or empty output
directory for each run. Samples exceeding the token limit are rejected before
training rather than silently truncating their target responses.

The optional [notebook](notebooks/finetune.ipynb) invokes the same packaged training command.
The root wrapper scripts and `scripts/train.py` accept the same flags as their
corresponding CLI commands. Old positional output paths must become `-o PATH`.

### Chat locally

```bash
uv sync --locked --extra inference
source .venv/bin/activate
phantasm chat --model phantasm_model/gguf/MODEL.gguf \
  --context-size 4096 --max-tokens 256 --threads 4
```

Choose the filename reported by your GGUF export. Models must include their chat
template. The same template is used to count prompt tokens and generate replies.
Old complete exchanges are trimmed to fit; the system prompt and current message
are preserved. `/reset` clears history and `/quit` exits. Failed generations do not
alter history. `--gpu-layers -1` requests full offload when llama-cpp-python was
built with a compatible GPU backend; the default is CPU (`0`). See
[backend installation](https://llama-cpp-python.readthedocs.io/en/latest/#installation).

## Configuration

| Command | Options / environment | Defaults |
| --- | --- | --- |
| `scrape` | `DISCORD_TOKEN`, `--resume`, `-o` | Fresh download; `raw_export.json` |
| `parse` | `--user-id`, `--target-id`, `--ignore-others` | Reject additional participants |
| `format` | `--window`, `--max-gap`, `--session-gap`, `--val-split`, `--system-prompt` | 6 turns, 300s, 1800s, 0.05, empty |
| `train` | `--max-steps`, `--eval-steps`, `--max-seq-length`, `--seed`, `--model-revision` | 120, 10, 2048, 3407, model default revision |
| `train --backend colab` | `--job`, `--gpu`, `--session`, `--detach`, `--keep`, `--artifact-store`, `PIXELDRAIN_API_KEY`, `HF_TOKEN` | T4, new runtime, wait and fetch, Colab storage |
| `colab` / `recover` | `--jobs-dir`, `recover --resume` | `.phantasm/jobs`; recovery downloads only |
| `chat` | `--context-size`, `--max-tokens`, `--threads`, `--gpu-layers`, `--temperature` | 2048, 150, 4, 0, 0.7 |

Use `phantasm COMMAND --help` for all arguments. For reproducible model selection,
supply `--model-revision` with an immutable Hugging Face commit.

## Development and testing

```bash
mise run install
mise run lint
mise run test
mise run build
mise run check-dist
```

Without mise, use `uv run --locked --extra dev pytest`,
`uv run --locked --extra dev ruff check .`,
`uv run --locked --extra dev ruff format --check .`, `uv build --python 3.11`, and
`uv run --locked --extra dev python scripts/check_dist.py`.

CI tests Python 3.10/3.11/3.12 and installs both distributions into fresh environments
outside the checkout. Tests use synthetic data and mock external services; they do
not download models or exercise real GPUs. See [contributing](CONTRIBUTING.md) and
the [shared standards adaptation](.github/STANDARDS.md). No automated release or
package publication workflow is configured. Source is licensed under [MIT](LICENSE).
