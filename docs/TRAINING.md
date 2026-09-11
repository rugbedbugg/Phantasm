# GPU training environment

The training runtime is isolated from the lightweight pipeline. Its lock targets
**Linux x86_64, Python 3.11, NVIDIA CUDA**. The selected package set uses Unsloth
2026.9.4, TRL 0.24.0, Transformers 4.57.6 and PyTorch 2.8.0 with matching torchvision
and xformers. All transitive dependencies and download hashes are recorded in
`src/phantasm/resources/training.txt`.

A live smoke run on **2026-09-11** used a Google Colab Tesla T4 and Python 3.11.16:
two training steps on the synthetic demo, validation/best-checkpoint selection,
Q4_K_M export, a checksum-verified download resumed after an interruption, and local
GGUF generation with llama-cpp-python 0.3.35 all completed. The base model was
`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, revision
`f15c379fb32bb402fa06a7ae9aecb1febf4b79ec`.

The live run exposed an Unsloth output-directory mismatch. Phantasm's collection
logic was corrected, and the existing exported weights were collected and published
without retraining. The corrected path also has regression coverage. Repeat the
smoke test for other models/hardware and after dependency changes. Synthetic demo
results establish execution, not persona quality. Live Pixeldrain backup and
checkpoint resumption onto a replacement runtime still require verification.

Response-only loss, early stopping and the reorganized dataset splits were added
after that run and have **not** been exercised on a GPU. Their preprocessing is
covered by GPU-free tests (marker derivation, masked-span verification, trainer
wiring against a stand-in stack), but a fresh smoke run is required before
treating the current stack as validated end to end.

## Installation

For managed Colab runs, use [the Phantasm Colab commands](COLAB.md); the remote
worker performs environment installation automatically. The steps below are for a
local GPU or an already-open notebook runtime.

From a checkout or unpacked source distribution:

```bash
uv venv --python 3.11 .venv-training
uv pip sync --python .venv-training/bin/python --require-hashes src/phantasm/resources/training.txt
uv pip install --python .venv-training/bin/python --no-deps -e .
uv pip check --python .venv-training/bin/python
# or, in one step:
mise run install-training
.venv-training/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

A compatible NVIDIA driver, sufficient GPU memory for the selected model/context,
and disk space for model downloads/checkpoints/quantization are required. GGUF
conversion may build/download llama.cpp tools and needs the system C/C++ build
toolchain. Consult the upstream
[Unsloth installation guide](https://unsloth.ai/docs/get-started/install) and
[GGUF export guide](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
for machine-specific prerequisites. This lock is not a universal GPU installer;
other operating systems/backends require their own resolved and verified environment.

## Verify before a longer run

```bash
.venv-training/bin/phantasm train --train-dataset dataset_train_sharegpt.jsonl \
  --eval-dataset dataset_val_sharegpt.jsonl --dry-run
.venv-training/bin/phantasm train --train-dataset dataset_train_sharegpt.jsonl \
  --eval-dataset dataset_val_sharegpt.jsonl \
  --max-steps 2 --eval-steps 1 --output-dir phantasm_model-smoke
```

Inspect `run.json` for train/validation loss, the derived loss markers and the
selected checkpoint. Confirm that the adapter files and the exported GGUF exist,
then load it with `phantasm chat` in the inference environment. Each run yields
exactly one GGUF, renamed from the converter's architecture-derived filename to
`<output directory>.<QUANT>.gguf` so a fine-tuned persona is never mistaken for a
stock base model. GPU validation is complete only
when that real training/export/load sequence succeeds. Use another output directory
for the full run; existing run directories are never overwritten automatically.

## Training objective

The default objective is **response-only**: loss is applied to the target
persona's assistant spans, and the prompt receives no gradient. Concretely, for a
Llama 3.1 template the unmasked region of every sample starts immediately after

```text
<|start_header_id|>assistant<|end_header_id|>\n\n
```

and ends at the next

```text
<|start_header_id|>user<|end_header_id|>\n\n
```

so the system prompt, your messages, and any other participant's messages are
masked out. Those two marker strings are not hardcoded: they are derived from the
tokenizer's own chat template at run time by rendering a sentinel conversation,
and rejected if they do not appear exactly once per turn. Masking itself is
performed by Unsloth's `train_on_responses_only`, and the derived markers are
recorded in `run.json` under `loss_markers`. Pass `--loss full` to fall back to
plain full-sequence language modelling; do that if a model's chat template cannot
be analysed, since Phantasm refuses to guess.

## Validation, checkpoints and early stopping

Validation data is optional but recommended. With `--eval-dataset`, evaluation
loss is computed every `--eval-steps`, checkpoints are saved on the same
interval, the lowest validation-loss checkpoint is restored before export, and
`--early-stopping-patience N` stops after N evaluations without improvement.
Without it, training retains checkpoints but performs no validation and selects
no best checkpoint. Exact duplicate train/validation samples are rejected; use
Phantasm's session-based split, which assigns whole conversation sessions to a
split before generating any sliding window.

`--dataset` and `--validation-dataset` are still accepted as the old spellings of
`--train-dataset` and `--eval-dataset`.

## Reproducibility

The code uses `SFTConfig` and `processing_class` from the
[TRL 0.24 API](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer).
`run.json` records package versions, seed, settings, input file hashes and counts,
chat-template hash, derived loss markers, resolved model revision when available,
metrics and output names. `training_config.json` holds the effective
configuration on its own, so a later run can be reproduced from it directly:
every exposed option (base model, LoRA rank/alpha/dropout, learning rate, batch
size, gradient accumulation, epochs or max steps, sequence length, seed, output
directory, quantization method) is stored there. A compact summary of the same
configuration is printed before training starts.

Use an immutable `--model-revision` for repeatable model downloads. Package locks
and seeds alone do not guarantee bitwise deterministic GPU results.

## Why there is no `.[training]` extra

The GPU stack resolves for one platform at a time: declaring `unsloth`,
`unsloth-zoo`, `torch` and `xformers` as a project extra makes this project's
universal `uv.lock` unsatisfiable, so `uv sync --locked` would fail for everyone,
including CI, which never needs the GPU stack. The tested environment is instead
the fully hashed, platform-pinned lock described here. `mise run install-training`
runs the installation steps below in one command.

## Optional transfer

Install [croc](https://github.com/schollz/croc) using your platform's normal package
manager first. `--croc-transfer` reads `CROC_SECRET` from the environment or a hidden
terminal prompt. Use a private, sufficiently random shared codephrase of at least
six characters and arrange for the receiver to use the same value.

The transfer starts only after successful export, requires exactly one generated
GGUF, does not install software, invoke a shell, copy credentials to the clipboard,
or print credentials. Transfer errors produce a nonzero exit status while keeping
all model outputs. If export creates multiple files/shards, transfer those explicitly
with croc. Transfer-only retries can also use croc directly, without repeating training.

## Updating the lock

Update compatible versions together in `requirements/training.in`, then regenerate:

```bash
uv pip compile requirements/training.in --python-version 3.11 \
  --python-platform x86_64-unknown-linux-gnu --generate-hashes --no-build \
  --output-file src/phantasm/resources/training.txt
```

Review the dependency changes and repeat the real GPU smoke run before considering
an updated stack validated. Keep the root `uv.lock` separate so data preparation and
CI do not install the GPU stack.
