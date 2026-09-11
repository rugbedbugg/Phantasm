# Phantasm

![GitHub last commit](https://img.shields.io/github/last-commit/rugbedbugg/Phantasm?style=for-the-badge&labelColor=000000)
![GitHub repo size](https://img.shields.io/github/repo-size/rugbedbugg/Phantasm?style=for-the-badge&labelColor=000000)
![Stars](https://img.shields.io/github/stars/rugbedbugg/Phantasm?style=for-the-badge&labelColor=000000)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&labelColor=000000)
[![CI](https://img.shields.io/github/actions/workflow/status/rugbedbugg/Phantasm/ci.yml?branch=main&style=for-the-badge&labelColor=000000)](https://github.com/rugbedbugg/Phantasm/actions/workflows/ci.yml)

A modular developer pipeline for ingesting conversational chat logs, normalizing message bursts, formatting instruction datasets, fine-tuning open-weights LLMs with LoRA/QLoRA via Unsloth, and running quantized GGUF models locally.

> **Pipeline Framework Only**  
> This repository provides the data preparation, fine-tuning, and inference tooling. It does not distribute pre-trained model weights, private chat archives, or personal data. Users must supply their own chat logs and compute resources.

## Features

- **Discord Message Scraper**  
  Robust paginated extractor for channel or direct-message histories with exponential retry backoff and token masking.

- **Transcript Parser & Normalizer**  
  Standardizes raw message dumps, separates conversational roles (`you` vs `them`), filters control characters, and handles attachments/replies.

- **Sliding Turn Aggregator**  
  Merges consecutive rapid messages within configurable time thresholds (`max_gap_seconds`) into cohesive dialogue pairs.

- **ShareGPT Dataset Formatter**  
  Generates Unsloth-ready ShareGPT JSONL datasets with configurable train/validation splits.

- **LoRA / QLoRA Fine-Tuning Integration**  
  Ready-to-run headless scripts and interactive notebooks for training Llama 3.1 8B on cloud/consumer GPUs using Unsloth.

- **Automated GGUF Quantization**  
  Directly merges 16-bit adapters and quantizes models to `q4_k_m`, `q8_0`, or `f16` GGUF formats.

- **Local GGUF Inference**  
  Interactive terminal chat interface running locally on CPU/Metal/Vulkan via `llama-cpp-python`.

## Architecture & Pipeline

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Discord Scraper │ ────> │ Transcript      │ ────> │ Turn Aggregator │
│ (scrape)        │       │ Parser (parse)  │       │ & Formatter     │
└─────────────────┘       └─────────────────┘       └────────┬────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │ ShareGPT Format(.jsonl│
                                                 └───────────┬───────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │ LoRA Fine-Tuning      │
                                                 │ (Unsloth / Llama 3.1) │
                                                 └───────────┬───────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │ GGUF Quantization     │
                                                 │ (Q4_K_M / Q8_0)       │
                                                 └───────────┬───────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │ Local CLI Inference   │
                                                 │ (llama-cpp-python)    │
                                                 └───────────────────────┘
```

## Installation

### Prerequisites

- Python `>=3.10`
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- [mise](https://mise.jdx.dev/) (optional, for toolchain management)

### From Source

```bash
git clone https://github.com/rugbedbugg/Phantasm.git
cd Phantasm
mise trust
mise install
mise run install
```

Without mise:

```bash
git clone https://github.com/rugbedbugg/Phantasm.git
cd Phantasm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To include local inference dependencies (`llama-cpp-python`):

```bash
pip install -e ".[inference]"
```

## Quick Start

The pipeline can be driven either through the unified `phantasm` CLI or standalone scripts.

### 1. Ingest Channel Messages

```bash
phantasm scrape <CHANNEL_ID> <DISCORD_TOKEN> -o raw_export.json
# or
python scrape_discord.py <CHANNEL_ID> <DISCORD_TOKEN> raw_export.json
```

### 2. Parse & Normalize Roles

Separate user messages (`you`) from the target persona (`them`):

```bash
phantasm parse raw_export.json YOUR_USERNAME -o parsed.json
# or
python parse_discord_export.py raw_export.json YOUR_USERNAME parsed.json
```

### 3. Generate Fine-Tuning Datasets

Merge rapid consecutive bursts and create sliding conversation contexts:

```bash
phantasm format -i parsed.json -p dataset --window 6 --max-gap 300
# or
python format_training_data.py --input parsed.json --output-prefix dataset
```

Outputs:
- `dataset_train_sharegpt.jsonl`
- `dataset_val_sharegpt.jsonl`

### 4. Fine-Tune on GPU (Colab / Cloud)

Run the included Jupyter notebook [`notebooks/finetune.ipynb`](notebooks/finetune.ipynb) or execute headless training:

```bash
python scripts/train.py --dataset dataset_train_sharegpt.jsonl --max-steps 120 --quant-method q4_k_m
```

### 5. Chat with the Quantized Model Locally

```bash
phantasm chat --model ./model.gguf
# or
python inference.py --model ./model.gguf
```

## Development & Testing

```bash
uv run pytest           # Running tests
uv run ruff check .     # Linting
uv run ruff format .    # Formatting
```
