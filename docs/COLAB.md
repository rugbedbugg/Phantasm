# Colab jobs and recovery

Run the workflow from your local terminal using `phantasm`. The installed package
includes its remote worker and GPU dependency lock; a source checkout or notebook
is not required to submit a job.

## Setup and submission

```bash
phantasm colab setup
phantasm colab login
phantasm train --backend colab --job persona \
  --train-dataset dataset_train_sharegpt.jsonl \
  --eval-dataset dataset_val_sharegpt.jsonl \
  --artifact-store pixeldrain
```

Before that submission, set `PIXELDRAIN_API_KEY` through your environment or secret
manager. Set `HF_TOKEN` too if your selected model requires Hugging Face access.
Credentials are not command arguments and are excluded from saved job metadata and
submission bundles. The worker receives them through a temporary uploaded file,
deletes it on startup, and retains them in its process environment.

`phantasm colab setup` installs the pinned `google-colab-cli==0.6.0` backend using
`uv tool install`. Phantasm invokes it internally. Follow Google's authentication
prompts from `phantasm colab login`: open the URL and paste Google's authorization
code into your local terminal. Login verifies your account without allocating a
runtime. Rerun it when saved credentials expire or are revoked. Account access,
available GPUs, and runtime limits remain Colab requirements. Setup installs
software; it does not allocate a runtime.
See the [official backend documentation](https://github.com/googlecolab/google-colab-cli)
for Google account prerequisites.

Submission validates and snapshots the input files, creates a named T4 runtime,
uploads the package and inputs directly to Colab in bounded chunks, verifies the
uploaded file checksums, installs the hash-locked Python
3.11 training environment, and starts a detached worker. Choose another supported
GPU with `--gpu`, or reuse an existing runtime with `--session NAME`. With
`--dry-run`, submission only validates inputs and prints the plan: it does not
allocate, authenticate, upload, or download models.

By default, the local command waits, downloads and verifies the exported results,
then stops the runtime it created. `--keep` retains it; a supplied `--session` is
always retained. `--detach` returns after submission and leaves fetching and stopping
for later. Closing the local terminal after submission does not stop the worker.

## Monitor and fetch

```bash
phantasm colab status persona
phantasm colab logs persona
phantasm colab fetch persona
phantasm colab stop persona
```

Status reports setup, training, publishing, completion or failure and the latest
backup receipt when available. Logs show the most recent 64 KiB. Fetch saves the
verified manifest, adapter and GGUF files under `.phantasm/jobs/persona/result/`.
Downloads stream to disk, retain partial progress, and resume with byte ranges.
The result bundle excludes full-precision merge files and download caches.
Point `phantasm chat --model` at a generated file in its `gguf/` subdirectory.

If final Pixeldrain publication fails while Colab is still available, `fetch`
repackages the existing output and downloads it directly from Colab without
repeating training. A failed transfer or checksum leaves the runtime running.
Phantasm stops an owned runtime only after successful extraction and verification.
An interrupted wait likewise leaves the runtime running; inspect or stop it with
the commands above.

## Durable backups and recovery

`--artifact-store pixeldrain` opts into uploading checkpoint and result archives
to your Pixeldrain account. At each Trainer checkpoint save (`--eval-steps`, default
10), the worker bundles the checkpoint, optimizer/scheduler state, run manifest,
and best checkpoint when different. After export it uploads the adapter, GGUF and
final manifest. Backup failures are reported and training continues; only a
successful upload provides a durable recovery point.

If your ISP blocks the primary domain, select an
[official alternative](https://docs.pixeldrain.com/questions_and_answers/#alternative-domain-names):

```bash
export PIXELDRAIN_DOMAIN=pixeldrain.net
```

Set this in the terminal used for training, fetching, and recovery. Phantasm passes
the setting to Colab for checkpoint and result uploads as well. The default is
`pixeldrain.com`; only official hostnames are accepted, HTTPS is required, and
authenticated requests do not follow redirects. This setting does not change
your system DNS. The API key still comes from `PIXELDRAIN_API_KEY`.
Verify your account email before uploading: Pixeldrain can accept an API key for
file listing while rejecting uploads with `email_address_not_verified` (HTTP 403).

```bash
# Download the completed result, or the latest available checkpoint:
phantasm recover persona

# Restore the latest checkpoint and continue training on a new runtime:
phantasm recover persona --resume
```

Recovery searches your Pixeldrain account by the saved job ID and works even if the
original runtime is gone. `--resume` explicitly provisions a replacement runtime
(or uses `--session NAME`), restores state and uses the original dataset snapshots
and training settings. It continues to the original `--max-steps` target. It does
not automatically allocate replacements after disconnects. Without any successful
checkpoint upload, there is no checkpoint to resume. Steps since the last uploaded
checkpoint must be repeated.

Keep `.phantasm/jobs/` locally: it contains job IDs, dataset snapshots and settings
needed for this recovery command, and is Git-ignored. Run commands from the same
directory, or pass `--jobs-dir PATH` consistently. Archives do not include the input
datasets; losing both the runtime and local job directory prevents this managed
resume workflow. Checkpoint archives are unencrypted, may be large, and remain in
your Pixeldrain account until removed or expired. Treat their file links as private.
Phantasm does not automatically prune remote backups.

Uploads retry transient failures from the beginning. Downloads retain partial data,
request a byte range on retry, and verify size and SHA-256 before installing the
archive. Invalid archives and checksum mismatches are rejected. Pixeldrain's
retention, storage and download restrictions still apply; Phantasm cannot recover
an expired/deleted backup or bypass service restrictions. See the
[Pixeldrain API](https://pixeldrain.com/api).

The default `--artifact-store colab` transfers results directly and has no durable
backup after runtime loss. Pixeldrain backups require the explicit option and key.

## Verification status

Automated tests cover job orchestration, detached setup, checkpoint/result bundling,
recovery validation, interrupted transfers, checksum rejection, credential handling
and runtime cleanup with synthetic data and mocked services. A live T4 smoke run
on 2026-09-11 verified Google authentication, provisioning, uploads, training,
Q4_K_M export, a resumed streamed download, checksums, automatic shutdown, and local
chat. The output-directory mismatch discovered in that run was corrected and its
existing weights collected without retraining. Chunked uploads and their remote
checksum verification were also checked against the real runtime.

On 2026-09-11, live Pixeldrain authentication, checkpoint uploads, HTTP 206 resumed
downloads, checkpoint recovery, final-result publication and result recovery all
passed through `pixeldrain.net` using synthetic data. Three temporary test files were
deleted and cleanup was verified. This test did not allocate a GPU or resume a real
training run onto a replacement runtime. Start with
`--max-steps 2 --eval-steps 1` and verify the exported model before committing to a
longer run. See [GPU verification](TRAINING.md).
The step count limits training only: downloading base weights, merging and
quantizing an 8B model, and retrieving the multi-gigabyte result can take much
longer than those two steps.
