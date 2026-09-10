"""Headless fine-tuning and GGUF quantization script for Unsloth / Llama 3.1."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 with Unsloth and export GGUF."
    )
    parser.add_argument(
        "--dataset",
        default="/content/phantasm_train_sharegpt.jsonl",
        help="Path to training dataset JSONL (ShareGPT format)",
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Base model checkpoint name",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Training learning rate",
    )
    parser.add_argument(
        "--output-dir",
        default="/content/phantasm_model",
        help="Output directory for merged model and GGUF",
    )
    parser.add_argument(
        "--quant-method",
        default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16"],
        help="GGUF quantization method",
    )
    parser.add_argument(
        "--croc-transfer",
        action="store_true",
        help="Stream the quantized GGUF via croc peer-to-peer",
    )
    parser.add_argument(
        "--croc-secret",
        default="phantasm-model-gguf",
        help="Codephrase for croc transfer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template, standardize_sharegpt
    except ImportError:
        print("Error: Unsloth and training dependencies are required to run this script.")
        print(
            "Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        )
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset '{args.dataset}' does not exist.")
        sys.exit(1)

    print(f">>> Loading base model: {args.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print(">>> Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f">>> Loading dataset from {dataset_path}...")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f">>> Prepared {len(dataset)} samples for training")

    print(">>> Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    print(">>> Starting training...")
    trainer.train()
    print(">>> Fine-tuning complete!")

    print(f">>> Exporting GGUF ({args.quant_method})...")
    model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method=args.quant_method)
    print(">>> GGUF export complete!")

    if args.croc_transfer:
        print(">>> Installing croc...")
        subprocess.run(
            "curl -sL https://github.com/schollz/croc/releases/download/v11.5.2/croc_v11.5.2_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin/ croc",
            shell=True,
            check=False,
        )
        gguf_dir = Path(f"{args.output_dir}_gguf")
        gguf_files = list(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []
        if not gguf_files:
            print(f"Warning: No GGUF files found in {gguf_dir}")
            return

        target_file = gguf_files[0]
        print(f">>> Transferring {target_file.name} via croc (secret: {args.croc_secret})...")
        subprocess.run(
            f"CROC_SECRET='{args.croc_secret}' croc --ignore-stdin send {target_file}",
            shell=True,
            check=False,
        )
        print(">>> Transfer finished.")


if __name__ == "__main__":
    main()
