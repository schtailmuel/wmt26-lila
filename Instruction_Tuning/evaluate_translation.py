#!/usr/bin/env python3
"""
Gemma Machine Translation Post-Training Evaluation Script
=========================================================

This script evaluates a fine-tuned Gemma translation model on a test set.
It computes standard machine translation metrics: chrF/chrF++ and BLEU.

Best Practices Implemented:
---------------------------
1. **Instruction Alignment**: Uses the exact same Gemma-4 instruction template format used in training:
   `<|turn>user\n{instruction}<|turn>model\n` (via apply_chat_template).
   This ensures that the model is evaluated in its optimal instruction-following state.

2. **Left Padding Side**: Enforces tokenizer.padding_side = 'left' to prevent generation OOMs/loops.

3. **Greedy Decoding**: Uses greedy decoding (do_sample=False) which is the standard baseline
   in the literature for evaluating translation performance.

4. **Prefix Slicing & Alignment**: Decodes *only* the newly generated tokens, stripping out '<bos>' and using
   add_special_tokens=False to perfectly match the training distribution.

5. **SacreBLEU / chrF**: Evaluates translations using standard metrics with `sacrebleu` and `evaluate` packages.
"""

import os
import gc
import argparse
import yaml
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Use the same list of prompt templates as the training script
PROMPT_TEMPLATES = [
    "Translate the following text from {source_lang} to {target_lang}:\nText: {source_text}",
    "Translate from {source_lang} to {target_lang}.\n\nSource: {source_text}\n\nTranslation:",
    "How would you say this in {target_lang}? Translate from {source_lang}:\n\"{source_text}\"",
    "Provide the {target_lang} translation for this {source_lang} sentence:\n{source_text}",
    "Translate the text below from {source_lang} to {target_lang}.\nSource text: {source_text}\nTarget text:"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma Translation Model.")
    
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file")
    
    # Model & Test dataset paths
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model checkpoint directory")
    parser.add_argument("--test_file", type=str, help="Path to local test dataset file (json/jsonl/csv/tsv)")
    
    # Column mappings
    parser.add_argument("--source_lang_column", type=str, default="source_lang", help="Column containing source language name")
    parser.add_argument("--target_lang_column", type=str, default="target_lang", help="Column containing target language name")
    parser.add_argument("--source_text_column", type=str, default="source_text", help="Column containing source text")
    parser.add_argument("--target_text_column", type=str, default="target_text", help="Column containing target text")
    
    # Prompt Template
    parser.add_argument("--template_index", type=int, default=0, choices=[0, 1, 2, 3, 4], 
                        help="Index of the PROMPT_TEMPLATE to use during evaluation")
    
    # Generation & Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batch inference")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of generated tokens")
    parser.add_argument("--output_file", type=str, help="Optional CSV path to save predictions alongside source/reference text")
    parser.add_argument("--num_samples_to_show", type=int, default=5, help="Number of random translation samples to print to console")
    
    args = parser.parse_args()
    
    # Override arguments with values from YAML config if provided
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config_args = yaml.safe_load(f)
            if config_args:
                for k, v in config_args.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
                    else:
                        print(f"Warning: Ignoring unknown config parameter '{k}'")
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")
            
    # Manually check required arguments after merging with config
    if not args.model_path:
        parser.error("--model_path is required (either via CLI or config)")
    if not args.test_file:
        parser.error("--test_file is required (either via CLI or config)")
        
    return args


def load_data(args):
    """
    Loads test dataset and ensures columns exist.
    """
    ext = os.path.splitext(args.test_file)[-1].lower()
    if ext in [".json", ".jsonl"]:
        dataset = load_dataset("json", data_files={"test": args.test_file})
    elif ext == ".csv":
        dataset = load_dataset("csv", data_files={"test": args.test_file})
    else:
        dataset = load_dataset("text", data_files={"test": args.test_file})
        
    test_ds = dataset["test"]
    
    # Safety checks
    if args.source_text_column not in test_ds.column_names:
        raise ValueError(f"Source text column '{args.source_text_column}' not found in dataset columns: {test_ds.column_names}")
    if args.target_text_column not in test_ds.column_names:
        raise ValueError(f"Target text column '{args.target_text_column}' not found in dataset columns: {test_ds.column_names}")
        
    return test_ds


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer & Model
    print(f"Loading tokenizer and model from: {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Force padding side to left (Crucial for decoder-only batched generation)
    tokenizer.padding_side = "left"
    
    # Handle padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="cuda:0" if device == "cuda" else None
    )
    
    model.eval()
    
    # Inject chat template standardizations
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4",
    )
    
    # 2. Load Dataset
    print(f"Loading evaluation dataset from: {args.test_file}...")
    test_ds = load_data(args)
    
    # Prepare batch execution variables
    sources = test_ds[args.source_text_column]
    references = test_ds[args.target_text_column]
    src_langs = test_ds[args.source_lang_column] if args.source_lang_column in test_ds.column_names else ["English"] * len(sources)
    tgt_langs = test_ds[args.target_lang_column] if args.target_lang_column in test_ds.column_names else ["Ligurian"] * len(sources)
    
    # Define prompt template for consistent evaluation
    template = PROMPT_TEMPLATES[args.template_index]
    print(f"Evaluating with prompt template index {args.template_index}:")
    print(f'"{template}"')
    
    predictions = []
    
    # 3. Run Batch Inference
    print(f"Running inference over {len(sources)} test samples with batch size {args.batch_size}...")
    
    for i in tqdm(range(0, len(sources), args.batch_size)):
        batch_end = min(i + args.batch_size, len(sources))
        
        batch_prompts = []
        for j in range(i, batch_end):
            instruction = template.format(
                source_lang=src_langs[j], 
                target_lang=tgt_langs[j], 
                source_text=sources[j]
            )
            
            # Use native template formatting (with <bos> stripped to align perfectly with training)
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": instruction}]
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            ).removeprefix('<bos>')
            batch_prompts.append(prompt)
            
        # Tokenize batch with add_special_tokens=False
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512,
            add_special_tokens=False
        )
        inputs = {k: v.to("cuda:0" if device == "cuda" else "cpu") for k, v in inputs.items()}
        
        # Determine prompt token length to isolate generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy decoding is standard for translation evaluation
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Extract only newly generated translation tokens (slicing out the prompt)
        batch_generated_tokens = [seq[prompt_len:] for seq in generated_ids]
        
        # Decode
        batch_predictions = tokenizer.batch_decode(batch_generated_tokens, skip_special_tokens=True)
        predictions.extend([pred.strip() for pred in batch_predictions])
        
    # 4. Compute Metrics
    print("Computing metrics...")
    chrf_metric = evaluate.load("chrf")
    bleu_metric = evaluate.load("sacrebleu")
    
    # Prepare references format for sacrebleu (list of lists of reference text)
    sacre_refs = [[ref] for ref in references]
    
    chrf_score = chrf_metric.compute(predictions=predictions, references=sacre_refs)["score"]
    bleu_score = bleu_metric.compute(predictions=predictions, references=sacre_refs)["score"]
    
    print("\n" + "="*50)
    print("                 EVALUATION RESULTS")
    print("="*50)
    print(f"Corpus BLEU Score   : {bleu_score:.2f}")
    print(f"Corpus chrF++ Score : {chrf_score:.2f}")
    print("="*50 + "\n")
    
    # 5. Print random samples for visual examination
    print(f"Showing {args.num_samples_to_show} random prediction samples:")
    sample_indices = random.sample(range(len(sources)), min(args.num_samples_to_show, len(sources)))
    for idx in sample_indices:
        print("-"*40)
        print(f"[{src_langs[idx]} -> {tgt_langs[idx]}]")
        print(f"Source   : {sources[idx]}")
        print(f"Reference: {references[idx]}")
        print(f"Predicted: {predictions[idx]}")
    print("-"*40 + "\n")
    
    # 6. Optional predictions export to CSV
    if args.output_file:
        df_out = pd.DataFrame({
            "source_lang": src_langs,
            "target_lang": tgt_langs,
            "source_text": sources,
            "reference_text": references,
            "prediction": predictions
        })
        df_out.to_csv(args.output_file, index=False)
        print(f"Exported prediction records successfully to: {args.output_file}")


if __name__ == "__main__":
    main()
