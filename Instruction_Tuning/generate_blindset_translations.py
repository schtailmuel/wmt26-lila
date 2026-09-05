#!/usr/bin/env python3
"""
WMT26 Blindset Translation Generation Script (Unified Full FT & LoRA)
=====================================================================
This script automates the generation of machine translation hypotheses for the
WMT26 blindset benchmarks across multiple fine-tuned model checkpoints (both LoRA & Full FT).

It supports:
- Automatic loading/fetching of the online blindset JSONL.
- Filtering for target languages: Ligurian (lij_Latn) and Ladin (lld_Latn).
- Scanning checkpoint folders matching 'checkpoint-*' or a specified single checkpoint.
- Natively loads both LoRA adapters and Full Parameter checkpoints under both Unsloth and standard HF.
- Runs cleanly on a single GPU (cuda:0) to prevent multi-GPU tensor device mismatch errors.
- Clean prefix-alignment: strips '<bos>' and uses add_special_tokens=False to match training distributions.
"""

import os
import gc
import json
import argparse
import urllib.request
import torch
from tqdm import tqdm

# Optional Unsloth import for 2x faster inference
HAS_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate WMT26 Blindset Translations.")
    
    # Path configurations
    parser.add_argument("--checkpoints_dir", type=str, default="./output",
                        help="Base directory containing multiple checkpoints matching 'checkpoint-*'")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific single checkpoint directory to run (bypasses directory scan)")
    parser.add_argument("--model_id", type=str,
                        default="google/gemma-4-E4B-it",
                        help="Base model ID or directory used to load the tokenizer if missing from checkpoints")
    
    # Blindset source configuration
    parser.add_argument("--blindset_file", type=str, 
                        default="https://www2.statmt.org/wmt26/assets/wmt26_genmt_blindset.jsonl",
                        help="Path or URL to the WMT26 blindset JSONL file")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./blindset_predictions",
                        help="Directory where translation predictions will be saved as JSONL")
    
    # Inference Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generated translation tokens")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Limit translation generation to the first N samples (useful for fast verification)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, 
                        help="Repetition penalty for generation to prevent looping")
    parser.add_argument("--use_unsloth", type=bool, default=True, 
                        help="Use Unsloth loader if available for 2x faster inference")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length for model loading")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the model in 4-bit precision")
    
    return parser.parse_args()


def load_blindset(file_or_url):
    """
    Loads blindset JSONL from a local file path or web URL.
    """
    rows = []
    if file_or_url.startswith("http://") or file_or_url.startswith("https://"):
        print(f"Downloading and reading blindset from URL: {file_or_url}...")
        try:
            req = urllib.request.Request(
                file_or_url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response:
                for line in response:
                    if line.strip():
                        rows.append(json.loads(line.decode("utf-8")))
        except Exception as e:
            raise RuntimeError(f"Failed to download blindset from {file_or_url}: {e}")
    else:
        print(f"Reading blindset from local file: {file_or_url}...")
        if not os.path.exists(file_or_url):
            raise FileNotFoundError(f"Blindset file not found locally: {file_or_url}")
        with open(file_or_url, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    
    print(f"Successfully loaded {len(rows)} raw rows from the blindset.")
    return rows


def scan_checkpoints(args):
    """
    Scans the checkpoints directory for subfolders matching 'checkpoint-*'
    or starting with 'final_'. Returns sorted list of (path, name) tuples.
    """
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            return [(args.checkpoint, os.path.basename(args.checkpoint.rstrip("/")))]
        else:
            raise FileNotFoundError(f"Specified checkpoint path does not exist: {args.checkpoint}")
            
    if not os.path.exists(args.checkpoints_dir):
        print(f"Warning: Checkpoints directory '{args.checkpoints_dir}' not found. Cannot perform multi-checkpoint scanning.")
        return []
        
    checkpoints = []
    print(f"Scanning directory for checkpoints: {args.checkpoints_dir}...")
    for item in os.listdir(args.checkpoints_dir):
        item_path = os.path.join(args.checkpoints_dir, item)
        if os.path.isdir(item_path):
            if item.startswith("checkpoint-") or item.startswith("final_"):
                checkpoints.append((item_path, item))
                
    def get_step_num(name):
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-")[1])
            except ValueError:
                return float("inf")
        elif name.startswith("final_"):
            return float("inf")
        return name
        
    checkpoints.sort(key=lambda x: get_step_num(x[1]))
    print(f"Found and sorted {len(checkpoints)} checkpoint directories to process.")
    return checkpoints


def run_batch_inference(model, tokenizer, batch_rows, args, device="cuda"):
    """
    Translates a batch of rows.
    """
    batch_prompts = []
    for row in batch_rows:
        instruction = row.get("instruction", "")
        source_doc = row.get("source_doc", "")
        
        combined_prompt = f"{instruction}\n{source_doc}"
        
        # Generate prompt natively via apply_chat_template (Structured for multimodal processors)
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": combined_prompt}]
                }
            ],
            tokenize=False,
            add_generation_prompt=True
        ).removeprefix('<bos>')  # Stripping <bos> to match training format perfectly
        batch_prompts.append(prompt)
        
    # Tokenize batch using add_special_tokens=False
    inputs = tokenizer(
        text=batch_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024,
        add_special_tokens=False
    )
    inputs = {k: v.to("cuda:0" if device == "cuda" else "cpu") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    
    # Define stopping criteria (stops cleanly on standard <eos>)
    tok_obj = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    stop_tokens = [tok_obj.eos_token_id]
    for token in ["<turn|>", "<|turn>"]:
        token_id = tok_obj.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tok_obj.unk_token_id:
            stop_tokens.append(token_id)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,              # Deterministic greedy decoding for translation submission
            repetition_penalty=args.repetition_penalty,
            eos_token_id=stop_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
        
    # Extract only the generated suffix (slicing out prompt tokens)
    batch_generated_tokens = [seq[prompt_len:] for seq in generated_ids]
    batch_predictions = tokenizer.batch_decode(batch_generated_tokens, skip_special_tokens=True)
    return [pred.strip() for pred in batch_predictions]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}")
    
    # 1. Load the blindset data
    try:
        raw_rows = load_blindset(args.blindset_file)
    except Exception as e:
        print(f"Error loading blindset: {e}. Exiting.")
        return
        
    # 2. Filter for target languages: Ligurian (lij_Latn) and Ladin (lld_Latn)
    target_langs = ["lij_Latn", "lld_Latn"]
    filtered_rows = [row for row in raw_rows if row.get("tgt_lang") in target_langs]
    print(f"Filtered to {len(filtered_rows)} total rows matching target languages: {target_langs}.")
    
    if args.max_samples:
        print(f"Limiting to first {args.max_samples} samples.")
        filtered_rows = filtered_rows[:args.max_samples]
        
    if not filtered_rows:
        print("No rows match target languages. Exiting.")
        return
        
    # 3. Resolve list of target checkpoints
    checkpoints = scan_checkpoints(args)
    if not checkpoints:
        print("Error: No valid checkpoints found to translate. Exiting.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. Main loop over checkpoints
    for idx, (ckpt_path, ckpt_name) in enumerate(checkpoints):
        print("\n" + "=" * 80)
        print(f" Processing Checkpoint [{idx+1}/{len(checkpoints)}]: {ckpt_name}")
        print("=" * 80)
        
        # Resolve tokenizer source path dynamically
        tokenizer_path = args.model_id
        if tokenizer_path and not os.path.exists(tokenizer_path):
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

        # Load model and tokenizer
        tokenizer = None
        model = None
        
        if args.use_unsloth and HAS_UNSLOTH:
            print(f"Loading Unsloth-optimized model from: {ckpt_path} using tokenizer from: {tokenizer_path}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=ckpt_path,
                tokenizer_name=tokenizer_path,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=args.load_in_4bit
            )
            FastLanguageModel.for_inference(model)
        else:
            print(f"Loading standard Transformers model from: {ckpt_path} using tokenizer from: {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Check if checkpoint is a LoRA directory
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

        # Apply standard Unsloth/Gemma-4 chat template format
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-4",
        )
        
        # Prevent any padding tokens boundary bugs
        tok_obj = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None or tokenizer.pad_token_id >= len(tok_obj):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        tokenizer.padding_side = "left"

        # 5. Translating batch rows
        predictions_jsonl = []
        
        for i in tqdm(range(0, len(filtered_rows), args.batch_size), desc="Generating translations"):
            batch_end = min(i + args.batch_size, len(filtered_rows))
            batch_rows = filtered_rows[i:batch_end]
            
            hypotheses = run_batch_inference(model, tokenizer, batch_rows, args, device=device)
            
            for j, row in enumerate(batch_rows):
                # Save predictions back, mirroring exact required schema
                output_row = dict(row)
                output_row["hypothesis"] = hypotheses[j]
                predictions_jsonl.append(output_row)
                
        # 6. Save predictions to disk for the checkpoint
        output_file_path = os.path.join(args.output_dir, f"predictions_{ckpt_name}.jsonl")
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            for row in predictions_jsonl:
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                
        print(f"Predictions successfully generated and exported for {ckpt_name} to: {output_file_path}")
        
        # 7. Clean memory for sequential checkpoints
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n🎉 Submission Blindset Translation Generation Complete!")


if __name__ == "__main__":
    main()
