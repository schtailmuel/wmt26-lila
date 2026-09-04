#!/usr/bin/env python3
"""
Gemma-4-E4B Full Fine-Tuning Script
==================================================================================
Adhering to recommendations for Gemma/Unsloth stability:
- packing = False (Prevents RoPE cross-contamination)
- SequentialSFTTrainer: Forces sequential batching to perfectly preserve the 40/40/10/10 offline proportions under SFTTrainer.
- Formatting function: converts instructions and outputs to raw chat text and strips the duplicate '<bos>' prefix.
- train_on_responses_only: Unsloth's official helper to apply prompt-masking for target-only loss natively and safely.
- Zero monkey-patching & zero manual padding overrides: Passes the original tokenizer directly to SFTTrainer.
"""

import os
import argparse
import yaml
import torch
from datasets import load_dataset
from unsloth import FastModel
from torch.utils.data import SequentialSampler
from trl import SFTTrainer, SFTConfig


class SequentialSFTTrainer(SFTTrainer):
    """
    Forces the DataLoader to read the dataset sequentially.
    This guarantees our offline sampler's 40/40/10/10 proportions
    and sliding window deduplication are perfectly preserved in the batches.
    """
    def _get_train_sampler(self, dataset=None):
        dataset = dataset if dataset is not None else self.train_dataset
        return SequentialSampler(dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth Full Fine-Tuning of Gemma-4-E4B using SFTTrainer.")
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file")
    parser.add_argument("--train_file", type=str, default=None, help="Path to JSONL training data")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to JSONL validation data")
    parser.add_argument("--instruction_column", type=str, default="instruction", help="Name of instruction column")
    parser.add_argument("--output_column", type=str, default="output", help="Name of output/target column")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E4B", help="Path or HF ID of base model")
    parser.add_argument("--output_dir", type=str, default="./output/gemma_translation_full")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Per device eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for full parameter fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs)")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging step frequency")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation step frequency")
    parser.add_argument("--save_steps", type=int, default=200, help="Save step frequency")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy (steps, epoch, no)")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy (steps, epoch, no)")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True, help="Load best model at end of training")
    parser.add_argument("--metric_for_best_model", type=str, default="loss", help="Metric used to define best model")
    parser.add_argument("--greater_is_better", type=bool, default=False, help="Whether the metric_for_best_model should be maximized")
    parser.add_argument("--report_to", type=str, default="none", help="Metric logging backend (e.g., 'none')")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        print(f"Loading YAML config from {args.config}...")
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
            if config_args:
                for k, v in config_args.items():
                    if hasattr(args, k):
                        if str(v).lower() == 'none':
                            setattr(args, k, None)
                        else:
                            setattr(args, k, v)
                    else:
                        setattr(args, k, v)

    if args.train_file is None or args.model_id is None:
        raise ValueError("You must provide both --train_file and --model_id, either via CLI or YAML config.")

    return args


def load_data(args):
    print(f"Loading datasets from paths:\n - Train: {args.train_file}\n - Eval: {args.validation_file}")
    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    return load_dataset("json", data_files=data_files)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Model {args.model_id} with Unsloth for FFT...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False
    )

    print("Enabling full parameter fine-tuning (FFT) mode...")
    for param in model.parameters():
        param.requires_grad = True
        
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        
    model.gradient_checkpointing_enable()

    # Apply Unsloth's official get_chat_template to align with their standard Gemma-4 notebook
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4",
    )

    raw_datasets = load_data(args)

    # Apply prompt formatting and strip duplicate '<bos>'
    def formatting_prompts_func(examples):
        instructions = examples[args.instruction_column]
        outputs = examples[args.output_column]
        texts = []
        for inst, out in zip(instructions, outputs):
            convo = [
                {"role": "user", "content": [{"type": "text", "text": inst}]},
                {"role": "model", "content": [{"type": "text", "text": out}]}
            ]
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
            texts.append(text)
        return {"text": texts}

    print("Formatting and mapping offline dataset...")
    num_cores = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    
    train_dataset = raw_datasets["train"].map(
        formatting_prompts_func,
        batched=True,
        num_proc=num_cores,
        remove_columns=raw_datasets["train"].column_names,
        desc="Formatting training set"
    )

    eval_dataset = None
    if "validation" in raw_datasets:
        eval_dataset = raw_datasets["validation"].map(
            formatting_prompts_func,
            batched=True,
            num_proc=num_cores,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Formatting validation set"
        )

    # Configure SFTConfig from TRL
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy if eval_dataset else "no",
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end if eval_dataset else False,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_only_model=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=args.report_to,
        optim="adamw_torch",
        max_seq_length=args.max_seq_length,
        packing=False, # dynamic padding, not packed
        dataset_text_field="text"
    )

    # Initialize our sequential subclass of SFTTrainer passing the standard tokenizer object directly
    trainer = SequentialSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Apply Unsloth's native train_on_responses_only utility
    print("Applying target-only loss masking using Unsloth's train_on_responses_only...")
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # =========================================================================
    # 🔍 Programmatic Diagnostics, Tokenization & Attention Verifications 🔍
    # =========================================================================
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTICS: VERIFYING DATASET & TOKENIZATION INTEGRITY 🔍")
    print("=" * 60)
    
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    
    print(f"Batch Input IDs Shape:       {batch['input_ids'].shape}")
    print(f"Batch Attention Mask Shape:  {batch['attention_mask'].shape}")
    print(f"Batch Labels Shape:          {batch['labels'].shape}")
    
    target_idx = 0
    seq_ids = batch["input_ids"][target_idx].tolist()
    seq_mask = batch["attention_mask"][target_idx].tolist()
    seq_labels = batch["labels"][target_idx].tolist()
    
    raw_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    bos_token_id = raw_tok.bos_token_id
    
    bos_count = 0
    for tok in seq_ids:
        if tok == bos_token_id:
            bos_count += 1
        else:
            break
            
    print(f"\nBOS Token ID: {bos_token_id}")
    print(f"First 10 token IDs of sequence 0: {seq_ids[:10]}")
    print(f"First 10 decoded tokens:          {[raw_tok.decode([t]) for t in seq_ids[:10]]}")
    print(f"Consecutive BOS count at start:   {bos_count}")
    
    if bos_count <= 1 and len(seq_ids) > 0 and (seq_ids[0] == 105 or (seq_ids[0] == bos_token_id and len(seq_ids) > 1 and seq_ids[1] == 105)):
        print("✅ BOS ALIGNMENT: Valid BOS sequence structure starting with '<|turn>' detected.")
    else:
        print(f"⚠️ BOS WARNING: Found unexpected BOS structure with {bos_count} BOS tokens at start of sequence. First token ID: {seq_ids[0] if len(seq_ids) > 0 else 'None'}.")

    # Target-only masking check
    masked_count = seq_labels.count(-100)
    unmasked_count = len([x for x in seq_labels if x != -100])
    print(f"\nMasked prompt/padding tokens (loss = -100):  {masked_count}")
    print(f"Unmasked target response tokens (loss active): {unmasked_count}")
    
    if masked_count > 0 and unmasked_count > 0:
        print("✅ TARGET-ONLY LOSS: Masking verified. Gradients will only optimize response generation.")
    else:
        print("⚠️ LOSS MASKING ERROR: Verify template separators in train_on_responses_only.")
        
    print("\nVisual alignment of masked labels (Green = active, [MASKED] = ignored):")
    print("-" * 65)
    decoded_inputs = []
    decoded_labels = []
    for tok_id, label_id in zip(seq_ids[:50], seq_labels[:50]):
        word = raw_tok.decode([tok_id])
        decoded_inputs.append(word)
        if label_id == -100:
            decoded_labels.append("[\033[92mMASKED\033[0m]")
        else:
            decoded_labels.append(f"\033[94m{word}\033[0m")
            
    print(f"INPUTS : {''.join(decoded_inputs).replace(chr(10), ' ')}")
    print(f"LABELS : {''.join(decoded_labels).replace(chr(10), ' ')}")
    print("-" * 65 + "\n")

    print("Beginning Unsloth-optimized Full Fine-Tuning...")
    trainer.train()

    print(f"\nTraining Complete. Merging and saving final model to {args.output_dir}...")
    final_output_path = os.path.join(args.output_dir, "final_gemma_translation_unsloth_full")
    model.save_pretrained_merged(final_output_path, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(final_output_path)
    print(f"✅ Model saved successfully to {final_output_path}.")


if __name__ == "__main__":
    main()