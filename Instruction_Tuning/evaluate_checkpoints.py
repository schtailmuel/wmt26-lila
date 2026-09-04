#!/usr/bin/env python3
"""
Gemma Multi-Checkpoint Machine Translation Evaluation Script
============================================================
This script automates the evaluation of multiple fine-tuned model checkpoints
for the Gemma-2 (e.g., Gemma 4B) machine translation model.

It evaluates translation performance on:
1. Flores-200 (Meta) - English (eng_Latn) to target languages (Ligurian & Ladin).
2. BOUQuET (Meta FAIR) - English (eng_Latn) to target languages (Ligurian & Ladin).

Metrics computed:
- BLEU (via sacrebleu)
- chrF (via chrf)

Features:
- Scans a directory for all checkpoints matching 'checkpoint-*' or a final model folder.
- Supports evaluating Hugging Face Hub IDs directly (e.g., google/gemma-4-E4B).
- Evaluates each checkpoint sequentially, printing a live ASCII table.
- Supports both Hugging Face dataset loading and offline local file fallback.
- Explicitly uses left-padding for decoder-only causal LM batch generation.
- Decodes only generated tokens (slices out the prompt tokens).
- Clears PyTorch CUDA cache between runs to prevent VRAM OOM.
- Standardizes formatting using natural language names in prompt templates.
- Supports Unsloth model loading for 2x faster inference if available.
"""

# Optional Unsloth import for 2x faster inference (imported first to avoid warning)
HAS_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    pass

import os
import gc
import json
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import evaluate

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Natural language name mappings for prompt templates
LANG_NAME_MAP = {
    "eng_Latn": "English",
    "lij_Latn": "Ligurian",
    "lld_Latn": "Ladin (Val Badia)",  # Aligned to exact Ladin training string
    "eng": "English",
    "lij": "Ligurian",
    "lld": "Ladin (Val Badia)"
}

# The single standard prompt template explicitly used during instruction tuning
TRAINING_PROMPT_TEMPLATE = "Translate from {source_lang} to {target_lang}:\n{source_text}"
GEMMA_TEMPLATE = "Give a single straightforward translation from {source_lang} to {target_lang} of the following text:\n{source_text}" 
USE_GEMMA_TEMPLATE = False

# Prompt template for standard non-local Hugging Face models to ensure strict formatting
HF_PROMPT_TEMPLATE = (
    "Translate the following text from {source_lang} to {target_lang}.\n"
    "Provide only the direct translation. Do not include any explanation, introduction, preamble, "
    "or extra text. Your output must contain ONLY the translated text and nothing else.\n\n"
    "Text to translate:\n{source_text}\n\n"
    "Translation:"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Checkpoint Gemma Evaluation on Flores & BOUQuET.")
    
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file")
    parser.add_argument("--model_id", type=str, 
                        default="google/gemma-4-E4B-it",
                        help="Base model ID or directory used to load the tokenizer if missing from checkpoints")
    
    # Path configuration
    parser.add_argument("--checkpoints_dir", type=str, 
                        default="./output",
                        help="Base directory containing multiple checkpoints matching 'checkpoint-*'")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific single checkpoint, model directory, or HF Hub ID to evaluate (bypasses directory scan)")
    
    # Dataset configurations
    parser.add_argument("--datasets", type=str, default="flores,bouquet",
                        help="Comma-separated datasets to evaluate on: 'flores', 'bouquet', or both")
    parser.add_argument("--languages", type=str, default="lij_Latn,lld_Latn",
                        help="Comma-separated target languages: 'lij_Latn', 'lld_Latn'")
    
    # Split overrides
    parser.add_argument("--flores_split", type=str, default="devtest", help="Flores-200 split to evaluate on")
    parser.add_argument("--bouquet_split", type=str, default="test", help="BOUQuET split to evaluate on")
    
    # Local Offline File Fallbacks
    parser.add_argument("--flores_src_file", type=str, default=None,
                        help="Path to local Flores source (English) file (line-by-line format)")
    parser.add_argument("--flores_tgt_file_pattern", type=str, default=None,
                        help="Path pattern to local Flores target file, containing '{lang}' (e.g. 'data/flores/{lang}.devtest')")
    parser.add_argument("--bouquet_file", type=str, default=None,
                        help="Path to local BOUQuET JSON/JSONL file for fully offline evaluation")
    
    # Inference Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated translation tokens")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for generation")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Limit evaluation to first N samples per dataset-language pair (useful for fast validation)")
    parser.add_argument("--use_unsloth", type=bool, default=True, 
                        help="Use Unsloth loader and inference mode if library is available")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API token for loading gated datasets")
    parser.add_argument("--trust_remote_code", type=bool, default=True,
                        help="Whether to trust remote code when loading datasets from Hugging Face Hub")
    
    # Output configurations
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save full translation predictions and final summary tables")
    parser.add_argument("--num_samples_to_show", type=int, default=3,
                        help="Number of visual translation samples to print to the console during evaluation")
    
    args = parser.parse_args()
    
    # Override defaults with YAML configurations if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
            if config_args:
                for k, v in config_args.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
                    else:
                        print(f"Warning: Ignoring unknown config parameter '{k}'")
                        
    return args


def load_flores_data(lang_code, split, args):
    """
    Loads English to target language aligned sentences from Flores.
    Supports both Hugging Face Hub and offline aligned text files.
    """
    # 1. Check local file override
    if args.flores_src_file and args.flores_tgt_file_pattern:
        tgt_file = args.flores_tgt_file_pattern.format(lang=lang_code)
        if os.path.exists(args.flores_src_file) and os.path.exists(tgt_file):
            print(f"Loading local offline FLORES files: {args.flores_src_file} and {tgt_file}")
            with open(args.flores_src_file, "r", encoding="utf-8") as f_src, \
                 open(tgt_file, "r", encoding="utf-8") as f_tgt:
                src_lines = [line.strip() for line in f_src if line.strip()]
                tgt_lines = [line.strip() for line in f_tgt if line.strip()]
                
            count = min(len(src_lines), len(tgt_lines))
            records = [{"src_text": src_lines[i], "tgt_text": tgt_lines[i]} for i in range(count)]
            df = pd.DataFrame(records)
            if args.max_samples:
                df = df.head(args.max_samples)
            return df
        else:
            print(f"Warning: Local FLORES files not found ({args.flores_src_file} or {tgt_file}). Falling back to HF Hub.")

    # 2. HF Hub load
    print(f"Loading FLORES ({lang_code}) from Hugging Face Hub (split: {split})...")
    try:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        repo_id = "openlanguagedata/flores_plus" # Updated from facebook/flores
        
        src_ds = load_dataset(
            repo_id, 
            "eng_Latn", 
            split=split, 
            token=hf_token
        )
        tgt_ds = load_dataset(
            repo_id, 
            lang_code, 
            split=split, 
            token=hf_token
        )
        
        count = min(len(src_ds), len(tgt_ds))
        
        # Safely determine the text column name ('text' for flores_plus, 'sentence' for legacy flores)
        src_col = "text" if "text" in src_ds.column_names else "sentence"
        tgt_col = "text" if "text" in tgt_ds.column_names else "sentence"
        
        records = [{"src_text": src_ds[i][src_col], "tgt_text": tgt_ds[i][tgt_col]} for i in range(count)]
        df = pd.DataFrame(records)
        if args.max_samples:
            df = df.head(args.max_samples)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load FLORES from HF Hub: {e}. Please ensure network or local files are configured.")


def load_bouquet_data(lang_code, split, args):
    """
    Loads English to target language sentences from the Meta BOUQuET dataset.
    Supports both Hugging Face Hub and offline JSON/JSONL/CSV files.
    """
    # 1. Check local file override
    if args.bouquet_file and os.path.exists(args.bouquet_file):
        print(f"Loading local offline BOUQuET file: {args.bouquet_file}")
        ext = os.path.splitext(args.bouquet_file)[-1].lower()
        if ext == ".csv":
            df_full = pd.read_csv(args.bouquet_file)
        else:
            df_full = pd.read_json(args.bouquet_file, lines=True)
            
        # Filter for the specific target language and split
        df_filtered = df_full[df_full["src_lang"] == lang_code]
        if "split" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["split"] == split]
            
        # Aligns columns: English is the target in original BOUQuET schema, non-English is source.
        # We translate English (tgt_text) -> target language (src_text)
        records = []
        for _, row in df_filtered.iterrows():
            records.append({
                "src_text": row["tgt_text"],  # English is the input
                "tgt_text": row["src_text"]   # Target dialect is the target output
            })
            
        df = pd.DataFrame(records)
        if args.max_samples:
            df = df.head(args.max_samples)
        return df

    # 2. HF Hub load
    print(f"Loading BOUQuET ({lang_code}) from Hugging Face Hub (split: {split})...")
    try:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        bouquet_ds = load_dataset(
            "facebook/bouquet", 
            "sentence_level", 
            split=split, 
            token=hf_token
        )
        # Filter for the target language (represented in 'src_lang' column of the multi-way parallel set)
        filtered = [x for x in bouquet_ds if x["src_lang"] == lang_code]
        
        records = []
        for item in filtered:
            records.append({
                "src_text": item["tgt_text"],  # English
                "tgt_text": item["src_text"]   # Ligurian or Ladin reference
            })
            
        df = pd.DataFrame(records)
        if args.max_samples:
            df = df.head(args.max_samples)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load BOUQuET from HF Hub: {e}. Please ensure network or local files are configured.")


def scan_checkpoints(args):
    """
    Scans the checkpoints directory for subfolders matching 'checkpoint-*' or
    a final merged model directory. Returns sorted list of paths.
    """
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            return [(args.checkpoint, os.path.basename(args.checkpoint.rstrip("/")))]
        else:
            # Assumes it's a Hugging Face Hub ID if not found locally
            print(f"Checkpoint path '{args.checkpoint}' not found locally. Assuming it's a Hugging Face Hub ID.")
            safe_name = args.checkpoint.replace("/", "_")
            global USE_GEMMA_TEMPLATE
            USE_GEMMA_TEMPLATE = True
            return [(args.checkpoint, safe_name)]
            
    if not os.path.exists(args.checkpoints_dir):
        print(f"Warning: Checkpoints directory '{args.checkpoints_dir}' not found. Cannot perform multi-checkpoint evaluation.")
        return []
        
    checkpoints = []
    print(f"Scanning directory for checkpoints: {args.checkpoints_dir}...")
    for item in os.listdir(args.checkpoints_dir):
        item_path = os.path.join(args.checkpoints_dir, item)
        if os.path.isdir(item_path):
            # Matches standard HF checkpoints and final merged model
            if item.startswith("checkpoint-") or item == "final_gemma_translation_unsloth_full":
                checkpoints.append((item_path, item))
                
    # Sort numerically based on step number, final model goes last
    def get_step_num(name):
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-")[1])
            except ValueError:
                return float("inf")
        elif name == "final_gemma_translation_unsloth_full":
            return float("inf")
        return name
        
    checkpoints.sort(key=lambda x: get_step_num(x[1]))
    print(f"Found and sorted {len(checkpoints)} checkpoint directories to evaluate.")
    return checkpoints


def clean_hf_translation(text, target_lang):
    """
    Cleans up any conversational or formatting artifacts that a Hugging Face model
    might generate, ensuring we capture only the direct translation.
    """
    text = text.strip()
    
    # Remove common conversational/introductory prefixes case-insensitively
    prefixes_to_remove = [
        f"translation in {target_lang.lower()}:",
        f"translation into {target_lang.lower()}:",
        f"translation to {target_lang.lower()}:",
        f"translation:",
        "here is the translation:",
        "here is the direct translation:",
        "translated text:",
    ]
    
    lower_text = text.lower()
    for prefix in prefixes_to_remove:
        if lower_text.startswith(prefix):
            text = text[len(prefix):].strip()
            lower_text = text.lower()
            
    # Strip wrapping quotes or backticks
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    elif text.startswith("`") and text.endswith("`"):
        text = text.strip("`").strip()
        
    # Handle markdown code block wrapping
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
    return text.strip()


def run_batch_inference(model, tokenizer, df, source_lang_name, target_lang_name, args, ckpt_path=None):
    """
    Runs batched greedy inference on the dataset.
    """
    sources = df["src_text"].tolist()
    
    predictions = []
    
    # Check if checkpoint path is a non-local path (e.g. Hugging Face Hub ID)
    is_hf_model = ckpt_path is not None and not os.path.exists(ckpt_path)
    if is_hf_model:
        print(">>> Detected Hugging Face Hub model (non-local). Utilizing format-specifying prompt and cleaning output.")
    
    # Standardize left padding for decoder-only causal language model batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    for i in tqdm(range(0, len(sources), args.batch_size), desc="Translating batches"):
        batch_end = min(i + args.batch_size, len(sources))
        batch_prompts = []
        
        for j in range(i, batch_end):
            # Format instruction with language names using the strict training template
            if is_hf_model:
                instruction = HF_PROMPT_TEMPLATE.format(
                    source_lang=source_lang_name,
                    target_lang=target_lang_name,
                    source_text=sources[j]
                )
            elif not USE_GEMMA_TEMPLATE:
                instruction = TRAINING_PROMPT_TEMPLATE.format(
                    source_lang=source_lang_name,
                    target_lang=target_lang_name,
                    source_text=sources[j]
                )
            else:
                instruction = GEMMA_TEMPLATE.format(
                    source_lang=source_lang_name,
                    target_lang=target_lang_name,
                    source_text=sources[j])
            
            # Generate prompt natively via apply_chat_template (Structured for multimodal processors)
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": instruction}]
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            ).removeprefix('<bos>')  # CRITICAL: Stripping <bos> to match training distribution perfectly
            batch_prompts.append(prompt)
            
        # Tokenize prompt batch using keyword argument for 'text' to support both tokenizers and processors
        inputs = tokenizer(
            text=batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512,
            add_special_tokens=False  # CRITICAL: Do not auto-inject BOS to ensure '<|turn>' is first token
        )
        inputs = {k: v.to("cuda:0" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
        
        prompt_len = inputs["input_ids"].shape[1]
        
        # Define stopping criteria (stops cleanly on standard <eos>)
        stop_tokens = [tokenizer.eos_token_id]
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,              # Clean, deterministic evaluation
                repetition_penalty=args.repetition_penalty,
                eos_token_id=stop_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Extract only the generated suffix (slicing out left-padded prompt tokens)
        batch_generated_tokens = [seq[prompt_len:] for seq in generated_ids]
        batch_predictions = tokenizer.batch_decode(batch_generated_tokens, skip_special_tokens=True)
        
        if is_hf_model:
            cleaned_predictions = [clean_hf_translation(pred, target_lang_name) for pred in batch_predictions]
            predictions.extend(cleaned_predictions)
        else:
            predictions.extend([pred.strip() for pred in batch_predictions])
        
    return predictions


def print_comparison_table(results):
    """
    Formats and prints a clean ASCII summary table of the metrics evaluated so far.
    """
    df_res = pd.DataFrame(results)
    if df_res.empty:
        return
        
    # Reorder columns for readability
    cols = ["checkpoint", "dataset", "language", "bleu", "chrf"]
    df_res = df_res[[c for col in cols if (c := col) in df_res.columns]]
    
    print("\n" + "=" * 80)
    print("                      EVALUATION METRICS SUMMARY")
    print("=" * 80)
    print(df_res.to_string(index=False, formatters={
        "bleu": "{:,.2f}".format,
        "chrf": "{:,.2f}".format
    }))
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Resolve list of target checkpoints
    checkpoints_to_evaluate = scan_checkpoints(args)
    if not checkpoints_to_evaluate:
        print("Error: No valid checkpoints found to evaluate. Exiting.")
        return
        
    # 2. Parse target datasets and languages
    target_datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    target_languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    
    # Initialize evaluation metrics
    chrf_metric = evaluate.load("chrf")
    bleu_metric = evaluate.load("sacrebleu")
    
    evaluation_records = []
    
    # 3. Main Loop over Checkpoints
    for idx, (ckpt_path, ckpt_name) in enumerate(checkpoints_to_evaluate):
        print("\n" + "#" * 80)
        print(f" Evaluating Checkpoint [{idx+1}/{len(checkpoints_to_evaluate)}]: {ckpt_name}")
        print("#" * 80)
        
        # Resolve tokenizer source path dynamically
        if args.checkpoint and not os.path.exists(args.checkpoint):
            # We are using a Hub ID, so pull the tokenizer from there directly
            tokenizer_path = ckpt_path
        else:
            tokenizer_path = args.model_id
            if tokenizer_path and not os.path.exists(tokenizer_path):
                # If it looks like a local file path but isn't found, try fallback
                if tokenizer_path.startswith("/") or tokenizer_path.startswith("./"):
                    local_fallback = "./checkpoints/cpt_model/full_model"
                    if os.path.exists(local_fallback):
                        print(f"Base model path '{tokenizer_path}' not found. Falling back to local: '{local_fallback}'")
                        tokenizer_path = local_fallback
                    else:
                        print(f"Warning: Neither '{tokenizer_path}' nor local fallback '{local_fallback}' found. "
                              f"Using '{ckpt_path}' for tokenizer.")
                        tokenizer_path = ckpt_path
            elif not tokenizer_path:
                tokenizer_path = ckpt_path

        # Load model and tokenizer for this checkpoint
        tokenizer = None
        model = None
        
        if args.use_unsloth and HAS_UNSLOTH:
            print(f"Loading Unsloth-optimized model from: {ckpt_path} using tokenizer from: {tokenizer_path}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=ckpt_path,
                tokenizer_name=tokenizer_path,
                max_seq_length=512,  # Short seq limit is perfect for fast inference
                dtype=None,
                load_in_4bit=False
            )
            FastLanguageModel.for_inference(model)  # Enable Unsloth 2x faster inference engine
        else:
            print(f"Loading standard Transformers model from: {ckpt_path} using tokenizer from: {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Check if this checkpoint is a LoRA adapter
            is_lora = os.path.exists(os.path.join(ckpt_path, "adapter_config.json"))
            if is_lora:
                print(f"Detected PEFT (LoRA) adapter directory. Loading base model {args.model_id} first...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="cuda:0" if device == "cuda" else None
                )
                from peft import PeftModel
                print(f"Attaching LoRA adapters from {ckpt_path} to base model...")
                model = PeftModel.from_pretrained(base_model, ckpt_path)
            else:
                print(f"Detected standard Full FT weights folder. Loading direct model...")
                model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="cuda:0" if device == "cuda" else None
                )
                
            if device != "cuda":
                model = model.to("cpu")
            model.eval()

        # CRITICAL INJECTION: Apply Unsloth's official get_chat_template to align with their standard Gemma-4 SFT format
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-4",
        )

        # Safely handle the padding token bug so we never assign an out-of-bounds pad_token_id
        tok_obj = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None or tokenizer.pad_token_id >= len(tok_obj):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Ensure tokenizer padding side is set to 'left'
        tokenizer.padding_side = "left"
            
        # Evaluate on each dataset and language pair
        for dataset_name in target_datasets:
            for lang_code in target_languages:
                lang_name = LANG_NAME_MAP.get(lang_code, lang_code)
                print(f"\n--- Running Evaluation: Dataset={dataset_name.upper()} | Language={lang_name} ({lang_code}) ---")
                
                # A. Load the aligned evaluation dataset
                if dataset_name == "flores":
                    try:
                        df_eval = load_flores_data(lang_code, args.flores_split, args)
                    except Exception as e:
                        print(f"Skipping Flores due to error: {e}")
                        continue
                elif dataset_name == "bouquet":
                    try:
                        df_eval = load_bouquet_data(lang_code, args.bouquet_split, args)
                    except Exception as e:
                        print(f"Skipping BOUQuET due to error: {e}")
                        continue
                else:
                    print(f"Warning: Unknown dataset '{dataset_name}'. Skipping.")
                    continue
                    
                if df_eval.empty:
                    print(f"Evaluation dataset is empty for {dataset_name} | {lang_code}. Skipping.")
                    continue
                    
                print(f"Evaluating on {len(df_eval)} aligned translation pairs.")
                
                # B. Batch Inference
                predictions = run_batch_inference(
                    model, tokenizer, df_eval,
                    source_lang_name="English",
                    target_lang_name=lang_name,
                    args=args,
                    ckpt_path=ckpt_path
                )                
                # C. Compute Metrics
                references = df_eval["tgt_text"].tolist()
                sacre_refs = [[ref] for ref in references]
                
                chrf_score = chrf_metric.compute(predictions=predictions, references=sacre_refs)["score"]
                bleu_score = bleu_metric.compute(predictions=predictions, references=sacre_refs)["score"]
                
                print(f"Result for {ckpt_name} | {dataset_name.upper()} | {lang_name}: BLEU={bleu_score:.2f}, chrF={chrf_score:.2f}")
                
                # Record metrics
                evaluation_records.append({
                    "checkpoint": ckpt_name,
                    "dataset": dataset_name,
                    "language": lang_name,
                    "bleu": bleu_score,
                    "chrf": chrf_score
                })
                
                # Save full translation outputs for transparency and audits
                df_results = pd.DataFrame({
                    "source": df_eval["src_text"],
                    "reference": references,
                    "prediction": predictions
                })
                output_csv_path = os.path.join(
                    args.output_dir, 
                    f"predictions_{ckpt_name}_{dataset_name}_{lang_code}.csv"
                )
                df_results.to_csv(output_csv_path, index=False)
                print(f"Full predictions exported to: {output_csv_path}")
                
                # D. Print visual comparison samples
                print(f"Random comparison samples ({args.num_samples_to_show}):")
                sample_idxs = random.sample(range(len(references)), min(args.num_samples_to_show, len(references)))
                for sample_idx in sample_idxs:
                    print("-" * 50)
                    print(f"Source   : {df_eval['src_text'].iloc[sample_idx]}")
                    print(f"Reference: {references[sample_idx]}")
                    print(f"Model    : {predictions[sample_idx]}")
                print("-" * 50)
                
        # E. Critical memory clean-up to prevent CUDA Out-Of-Memory during multi-checkpoint loops
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Display running progress summary
        print_comparison_table(evaluation_records)
        
    # 4. Final summary metrics export to disk
    summary_csv_path = os.path.join(args.output_dir, "evaluation_metrics_summary.csv")
    pd.DataFrame(evaluation_records).to_csv(summary_csv_path, index=False)
    print(f"📈 Evaluation complete! Summary metrics table successfully saved to {summary_csv_path}.")


if __name__ == "__main__":
    main()
