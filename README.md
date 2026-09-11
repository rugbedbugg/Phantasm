# Phantasm

[![CI](https://img.shields.io/github/actions/workflow/status/rugbedbugg/Phantasm/ci.yml?branch=main&style=for-the-badge&labelColor=000000)](https://github.com/rugbedbugg/Phantasm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&labelColor=000000)](LICENSE)

Turn a person's conversations into a local chatbot that learns how they talk.

Phantasm reconstructs conversational context from chat history, builds a
persona-focused training dataset, fine-tunes an open model on the target
person's responses, evaluates how closely the result matches their style, and
packages it as a GGUF model that can run locally.

The interesting part is not "fine-tune an LLM on someone's messages". It is the
methodology around it: explicit participant identity, real conversation session
boundaries, splits that cannot leak, loss applied only to the persona's own
replies, a local privacy audit, and a measurable persona-fidelity score.

Phantasm learns conversational style and response patterns from the data you
supply. It does not clone a person, and it cannot reproduce knowledge, judgement
or intent that is not in the transcript.

Supply your own authorized chat exports and model access. This repository
includes only synthetic example messages; it does not distribute private
archives or weights.

## Pipeline

```text
Chat History
    ↓
Participant Resolution      self / target / other, by ID or username
    ↓
Turn Aggregation            one speaker's burst becomes one turn
    ↓
Sessionization              a long silence ends the conversation
    ↓
Train / Validation / Test   whole sessions are assigned before any window exists
    ↓
ShareGPT Formatting         only target turns become assistant answers
    ↓
Persona-Focused QLoRA       loss on the target's responses only
    ↓
Persona Evaluation          style similarity against held-out ground truth
    ↓
GGUF
    ↓
Local Chat
```

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
The root `uv.lock` defines the lightweight environment; GPU training uses a
separate [pinned environment](docs/TRAINING.md), and local inference and
model-based evaluation use the `inference` extra.

## Quick start

Run the whole data pipeline without Discord access or a GPU:

```bash
phantasm parse examples/demo.json --self user --target persona -o parsed.json
phantasm inspect parsed.json
phantasm format -i parsed.json -p dataset
phantasm audit dataset_train_sharegpt.jsonl
phantasm train --train-dataset dataset_train_sharegpt.jsonl \
  --eval-dataset dataset_val_sharegpt.jsonl --dry-run
```

The formatter writes `dataset_train_sharegpt.jsonl`,
`dataset_val_sharegpt.jsonl`, `dataset_test_sharegpt.jsonl` and
`dataset_split_manifest.json`. The dry run validates the files and prints the
effective training configuration with SHA-256 fingerprints, without loading any
model dependency.

## Usage

### Import and resolve participants

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
phantasm parse raw_export.json --self YOUR_AUTHOR_ID --target PERSONA_AUTHOR_ID \
  -o parsed.json
```

Every message is labelled with exactly one participant role:

| Role | Meaning | Becomes |
| --- | --- | --- |
| `self` | you, the person supplying context | a `human` turn |
| `target` | the persona being reconstructed | a `gpt` turn: the training answer |
| `other` | anybody else in the conversation | a `human` turn, never an answer |

`--self` and `--target` accept an immutable author ID or a username. IDs are
preferred because they survive renames. Usernames resolve by exact match first,
then case-insensitively; a name that matches two different speakers is rejected
rather than guessed. An identity you name but that is not in the export is an
error, not a silent empty result. `--self` may be omitted for a group transcript
you are not part of; `--target` may be omitted only for a two-person export,
where the persona is unambiguous.

#### Group-chat behaviour

Phantasm 0.1 mapped every non-self speaker onto the persona, so in a group chat
everybody's messages were learned as the target. That is fixed. Third parties are
kept as conversational context by default (`--others context`) and can never
become a target response. `--others drop` excludes them and inserts a session
break so the remaining speakers are not spliced across the gap, and
`--others error` refuses group transcripts outright. `--ignore-others` remains
accepted as the old spelling of `--others drop`.

For group transcripts, `--speaker-labels others` prefixes context turns with the
speaker's name (`guest: what time?`), which lets the model tell participants
apart. Use the same convention at chat time if you enable it.

Legacy `phantasm parse raw_export.json YOUR_USERNAME` and the `--user-id` /
`--target-id` spellings still work. Attachment URLs and reply IDs are retained in
parsed data; training uses text only and does not resolve reply threads or
attachment content.

### Inspect before you build

```bash
phantasm inspect parsed.json --json
```

Reports message counts per participant, turns, sessions, usable sessions, target
response length distribution, URL/attachment/code-block counts, exact and near
duplicates, timestamp coverage, and an estimate of usable training samples. It
also accepts a `*.jsonl` dataset. `--json` emits the same numbers for scripting.

### Build leakage-safe datasets

```bash
phantasm format -i parsed.json -p dataset --window 6 \
  --turn-gap 300 --session-gap 1800 --val-ratio 0.075 --test-ratio 0.075
```

Two different time thresholds do two different jobs:

* `--turn-gap` (default 300s) merges one speaker's consecutive messages into a
  single turn, so `dude` / `what` / `are you serious` becomes one utterance.
* `--session-gap` (default 1800s) ends the conversation. Monday's exchange and
  Friday's exchange become separate sessions and never appear in one training
  sample.

Messages must be chronological. Timestamps normalize to UTC; missing offsets mean
UTC. A missing or unparseable timestamp is **not** treated as a zero-second gap:
such messages stay separate turns unless `--merge-unknown-timestamps` is given,
and crossing between known and unknown time starts a new session. Consecutive
unknown timestamps keep their input order within one session, because input order
is the only ordering signal that exists; Phantasm never fabricates a timestamp.

#### Split methodology

Whole usable sessions are assigned to train, validation and test **before** any
sliding window is generated, and windows are then generated independently inside
each split. A source message therefore cannot appear in two splits. Splitting is
chronological by default: the oldest sessions train, the newest are held out,
which is the honest arrangement for temporal data. `--split-strategy random`
requires an explicit `--split-seed`, so no run is nondeterministic by accident.

Ratios are shares of usable sessions, so sample ratios differ slightly. Set
`--val-ratio 0` or `--test-ratio 0` to disable a split; the file is then written
empty rather than left stale. Exact duplicate samples are removed within a split,
and a training sample identical to a held-out one is dropped from training so
evaluation stays honest. `dataset_split_manifest.json` records the effective
configuration, filter counts and per-split session/sample/message counts.

#### Data quality filtering

Conservative by default: empty and control-only messages, bot messages, and
deleted-message placeholders are removed. Opt in to more with `--drop-commands`,
`--drop-url-only`, `--drop-duplicate-messages`, `--max-message-chars N` and
`--max-code-blocks N`; opt out with `--keep-bots` and `--keep-deleted`. Every
removal is counted by reason and printed, and stored in the split manifest.

### Audit for private data

```bash
phantasm audit dataset_train_sharegpt.jsonl
```

A local heuristic scan for emails, phone numbers, IP addresses, API credentials
and tokens, private keys, credential assignments, filesystem paths, URLs and very
long numeric identifiers, plus duplicate samples, unusually long responses and
repeated response strings. Matched values are redacted before anything is
printed, and findings are reported by sample number and role.

Nothing is uploaded and no external service is contacted. This is a heuristic,
not a privacy guarantee: a clean report means no obvious problem was found, never
that the dataset is safe to publish. Review your data yourself before training on
someone else's messages, and only use transcripts you are authorized to use.

### Train and export

Training answers one question: given the conversation so far, what would the
target person say next? Loss is therefore applied to the target's assistant spans
only (`--loss response_only`, the default). The chat template's user and
assistant markers are derived from the tokenizer at run time and verified before
training, rather than hardcoded, and masking is applied with Unsloth's
`train_on_responses_only`. Context tokens (your messages, other participants'
messages, and the system prompt) receive no gradient. `--loss full` restores
plain full-sequence language modelling.

For a local GPU, follow [GPU setup and verification](docs/TRAINING.md), then run:

```bash
.venv-training/bin/phantasm train \
  --train-dataset dataset_train_sharegpt.jsonl \
  --eval-dataset dataset_val_sharegpt.jsonl \
  --max-steps 120 --eval-steps 10 --early-stopping-patience 3 \
  --lora-r 16 --lora-alpha 16 --learning-rate 2e-4 \
  --output-dir phantasm_model --quant-method q4_k_m
```

Validation is used during training: evaluation loss is computed every
`--eval-steps`, checkpoints are written on the same interval, and the lowest
validation-loss checkpoint is restored before export. `--early-stopping-patience`
stops after N evaluations without improvement. A run saves adapters, checkpoints,
metrics and fingerprints in `run.json`, the effective configuration in
`training_config.json`, and one GGUF under `phantasm_model/gguf/`. Use a new or
empty output directory for each run. Samples exceeding the token limit are
rejected before training rather than silently truncating their target responses.

`--dataset` and `--validation-dataset` remain accepted as the old spellings.

For Colab, Phantasm manages runtime creation, dataset transfer, the isolated GPU
environment, training, export, and verified result download:

```bash
phantasm colab setup
phantasm colab login
phantasm train --backend colab --job persona \
  --train-dataset dataset_train_sharegpt.jsonl \
  --eval-dataset dataset_val_sharegpt.jsonl
```

Google authentication and GPU availability still depend on your account. The command
waits for completion; add `--detach` to return after submission. Results go into
`.phantasm/jobs/persona/result/`. A runtime created by Phantasm is stopped after
successful download unless `--keep` is set. An existing `--session` is kept running.

For automatic checkpoint and result backups, set `PIXELDRAIN_API_KEY` in your
environment and add `--artifact-store pixeldrain`. After runtime loss, use
`phantasm recover persona` to download the latest backup, or
`phantasm recover persona --resume` to restore a checkpoint and continue on Colab.
See [Colab and recovery](docs/COLAB.md) for setup, monitoring, transfer retries,
storage limitations, and verification status.

The optional [notebook](notebooks/finetune.ipynb) runs the same packaged commands
with the same defaults. The root wrapper scripts and `scripts/train.py` accept the
same flags as their corresponding CLI commands.

### Measure persona fidelity

```bash
phantasm evaluate dataset_test_sharegpt.jsonl \
  --model phantasm_model/gguf/phantasm_model.Q4_K_M.gguf \
  --baseline-model base-model.gguf -o persona_evaluation.json
```

The model answers the held-out test prompts, and its replies are compared with
what the target actually said. Supplying `--baseline-model` scores the un-tuned
base model on the same prompts, which is what makes the improvement attributable
to fine-tuning. `--predictions FILE.jsonl` scores responses you generated
elsewhere, with no model loading at all.

```text
Phantasm Persona Evaluation
──────────────────────────────────────────────

Held-out responses            384

Metric                          fine-tuned      base
──────────────────────────────────────────────────────
Length similarity                     0.88      0.41
Punctuation similarity                0.82      0.55
Emoji style similarity                0.93      0.12
Casing similarity                     0.86      0.44
Lexical similarity                    0.76      0.38
Phrase similarity                     0.79      0.31
Sentence similarity                   0.81      0.47
──────────────────────────────────────────────────────
Persona Fidelity Score                83.1      38.3
```

*(Illustrative layout; the numbers above are not measurements.)*

Every component is a similarity in `[0, 1]` computed from two corpora of
responses:

| Metric | What it compares |
| --- | --- |
| Length similarity | mean, median word count and mean character count |
| Punctuation similarity | punctuation-mark frequency profile and overall rate |
| Emoji style similarity | emoji per response and share of responses with emoji |
| Casing similarity | uppercase letter ratio, all-lowercase share, shouting share |
| Lexical similarity | cosine over unigram frequencies plus type-token ratio |
| Phrase similarity | cosine over the most frequent 2- and 3-grams |
| Sentence similarity | mean sentence length and sentences per response |

The **Persona Fidelity Score** is `100 x` the weighted mean of those components
(currently equal weights, see `METRIC_WEIGHTS` in `src/phantasm/evaluate.py`).
The components are always printed alongside it, because no single number
represents a personality. These are style measurements against held-out data, not
evidence that the model is factually or behaviourally the same person. Evaluation
runs entirely locally: there is no paid API and no LLM judge.

### Chat locally

```bash
uv sync --locked --extra inference
source .venv/bin/activate
phantasm chat --model phantasm_model/gguf/phantasm_model.Q4_K_M.gguf \
  --context-size 4096 --max-tokens 256 --threads 4
```

A run produces exactly one GGUF, named after its output directory and
quantization (`phantasm_model.Q4_K_M.gguf`), not after the base architecture.
Converters name their output after the model they converted, which makes a
fine-tuned persona look like a stock upstream release once the file is moved;
Phantasm renames it so the filename says which run produced it. Models must
include their chat template. The same template is used to count prompt tokens and generate replies.
Old complete exchanges are trimmed to fit; the system prompt and current message
are preserved. `/reset` clears history and `/quit` exits. Failed generations do not
alter history. `--gpu-layers -1` requests full offload when llama-cpp-python was
built with a compatible GPU backend; the default is CPU (`0`). See
[backend installation](https://llama-cpp-python.readthedocs.io/en/latest/#installation).

## Supported input assumptions

* JSON: either a list of messages, or an object with a `messages` list. UTF-8,
  with or without a BOM.
* Each message has text content and an author identity (`author.id`,
  `author.username`, or top-level `author_id`/`username`). Text without any
  author identity is an error.
* Messages are in chronological order. Timestamps are ISO 8601; other formats are
  treated as unknown rather than guessed.
* Optional fields that are preserved when present: `id`, `timestamp`,
  `display_name`/`global_name`, `channel_id`, `attachments`, `message_reference`
  or `reply_to`, and `author.bot`.
* Exports from other platforms work if they are shaped this way; nothing in the
  parser is Discord-specific beyond field names it also accepts generically.

## Configuration

| Command | Options / environment | Defaults |
| --- | --- | --- |
| `scrape` | `DISCORD_TOKEN`, `--resume`, `-o` | Fresh download; `raw_export.json` |
| `parse` | `--self`, `--target`, `--others`, `--speaker-labels` (on `format`) | Others kept as context |
| `format` | `--window`, `--turn-gap`, `--session-gap`, `--val-ratio`, `--test-ratio`, `--split-strategy`, `--system-prompt` | 6 turns, 300s, 1800s, 0.075, 0.075, chronological |
| `format` filters | `--keep-bots`, `--keep-deleted`, `--drop-commands`, `--drop-url-only`, `--drop-duplicate-messages`, `--max-message-chars`, `--max-code-blocks` | Bots and deleted placeholders removed |
| `inspect` / `audit` | `--json`, `--max-response-chars`, `--locations` | Human-readable report |
| `train` | `--loss`, `--max-steps`, `--epochs`, `--eval-steps`, `--early-stopping-patience`, `--lora-r`, `--lora-alpha`, `--lora-dropout`, `--batch-size`, `--gradient-accumulation`, `--learning-rate`, `--max-seq-length`, `--seed`, `--model-revision` | response_only, 120, steps, 10, 0, 16, 16, 0.0, 2, 4, 2e-4, 2048, 3407 |
| `train --backend colab` | `--job`, `--gpu`, `--session`, `--detach`, `--keep`, `--artifact-store`, `PIXELDRAIN_API_KEY`, `HF_TOKEN` | T4, new runtime, wait and fetch, Colab storage |
| `colab` / `recover` | `--jobs-dir`, `recover --resume` | `.phantasm/jobs`; recovery downloads only |
| `evaluate` | `--model`, `--baseline-model`, `--predictions`, `--limit`, `--temperature`, `--max-tokens`, `--seed` | 0.7, 150, 3407 |
| `chat` | `--context-size`, `--max-tokens`, `--threads`, `--gpu-layers`, `--temperature` | 2048, 150, 4, 0, 0.7 |

Use `phantasm COMMAND --help` for all arguments. For reproducible model selection,
supply `--model-revision` with an immutable Hugging Face commit.

## Privacy considerations

* Everything runs locally. The only network calls Phantasm makes are the Discord
  scrape you ask for, the optional Colab/Pixeldrain transfer you opt into, and
  model downloads performed by Unsloth.
* Credentials are read from environment variables or hidden prompts, never from
  command-line arguments, and are never printed or written to job files.
* `phantasm audit` never transmits data and redacts what it matches.
* A fine-tuned model can reproduce fragments of its training data. Audit before
  training, and treat the resulting adapter and GGUF as containing the source
  conversations.
* Use transcripts you are authorized to use, and consider what the other people
  in the conversation would expect.

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
