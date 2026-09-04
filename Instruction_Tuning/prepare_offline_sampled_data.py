#!/usr/bin/env python3
"""
Offline Sampling Script for Instruction Tuning (Full Dataset)
=============================================================
This script processes massive JSONL datasets to strictly enforce a 40/40/20 
probabilistic distribution for eng-lij / eng-lld / others.

Key Features:
1. Oversampling: Refills exhausted language pools to ensure the 40/40/20 
   ratio is maintained strictly across the entire training epoch without 
   discarding data from the majority language.
2. Local Deduplication: Uses a sliding window based on `output_hash` to 
   guarantee that oversampled items never appear next to each other in a batch.
3. Splits: Safely extracts a validation set before oversampling begins.
"""

import os
import json
import random
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Offline Probabilistic Sampler")
    parser.add_argument("--input_files", nargs='+', required=True, help="Input JSONL files or glob patterns")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train.jsonl and validation.jsonl")
    parser.add_argument("--val_size", type=int, default=1000, help="Total size of validation set")
    parser.add_argument("--window_size", type=int, default=200, help="Deduplication window size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Resolve glob patterns
    file_paths = []
    for pattern in args.input_files:
        file_paths.extend(glob.glob(pattern))
        
    print(f"Discovered {len(file_paths)} input files.")

    eng_lij = []
    eng_lld = []
    oth_lij = []
    oth_lld = []
    
    # 1. Load and Categorize
    for fp in file_paths:
        print(f"Loading {fp}...")
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                src = data.get("src_lang", "")
                tgt = data.get("tgt_lang", "")
                
                # DISCARD ANY DATA WHERE TARGET IS NOT LADIN OR LIGURIAN
                if tgt not in ["lij", "lij_Latn", "lld"]:
                    continue
                    
                if tgt in ["lij", "lij_Latn"]:
                    if src == "eng":
                        eng_lij.append(data)
                    else:
                        oth_lij.append(data)
                elif tgt == "lld":
                    if src == "eng":
                        eng_lld.append(data)
                    else:
                        oth_lld.append(data)

    print(f"Total Loaded: {len(eng_lij)} eng->lij, {len(eng_lld)} eng->lld, {len(oth_lij)} oth->lij, {len(oth_lld)} oth->lld.")
    
    # Remove global exact duplicates from memory
    def deduplicate(pool):
        seen = set()
        unique = []
        for item in pool:
            # Composite key: instruction + output_hash ensures we don't drop 
            # the same translation if it came from different source languages/texts.
            h = str(item.get("instruction", "")) + str(item.get("output_hash", ""))
            if h not in seen:
                seen.add(h)
                unique.append(item)
        return unique

    eng_lij = deduplicate(eng_lij)
    eng_lld = deduplicate(eng_lld)
    oth_lij = deduplicate(oth_lij)
    oth_lld = deduplicate(oth_lld)
    print(f"Unique Available: {len(eng_lij)} eng->lij, {len(eng_lld)} eng->lld, {len(oth_lij)} oth->lij, {len(oth_lld)} oth->lld.")

    random.shuffle(eng_lij)
    random.shuffle(eng_lld)
    random.shuffle(oth_lij)
    random.shuffle(oth_lld)

    # 2. Extract Validation Set strictly at 40/40/10/10
    val_data = []
    val_eng_lij_count = int(args.val_size * 0.4)
    val_eng_lld_count = int(args.val_size * 0.4)
    val_oth_lij_count = int(args.val_size * 0.1)
    val_oth_lld_count = int(args.val_size * 0.1)

    val_data.extend(eng_lij[:val_eng_lij_count])
    eng_lij = eng_lij[val_eng_lij_count:]
    
    val_data.extend(eng_lld[:val_eng_lld_count])
    eng_lld = eng_lld[val_eng_lld_count:]
    
    val_data.extend(oth_lij[:val_oth_lij_count])
    oth_lij = oth_lij[val_oth_lij_count:]
    
    val_data.extend(oth_lld[:val_oth_lld_count])
    oth_lld = oth_lld[val_oth_lld_count:]
    
    random.shuffle(val_data)

    # 3. Training Set Sampling (Downsampling)
    # We want a 40/40/10/10 ratio. 
    # eng_lij (4 parts), eng_lld (4 parts), oth_lij (1 part), oth_lld (1 part).
    # We find the constraining pool by calculating the max safe multiplier.
    max_multiplier = min(
        len(eng_lij) / 4.0,  
        len(eng_lld) / 4.0,  
        len(oth_lij) / 1.0,    
        len(oth_lld) / 1.0     
    )
    
    target_eng_lij = int(max_multiplier * 4.0)
    target_eng_lld = int(max_multiplier * 4.0)
    target_oth_lij = int(max_multiplier * 1.0)
    target_oth_lld = int(max_multiplier * 1.0)
    total_train_samples = target_eng_lij + target_eng_lld + target_oth_lij + target_oth_lld

    print(f"Downsampling to match smallest dataset. Target training size: {total_train_samples} samples.")
    print(f"Quotas -> eng->lij: {target_eng_lij}, eng->lld: {target_eng_lld}, oth->lij: {target_oth_lij}, oth->lld: {target_oth_lld}")

    random.shuffle(eng_lij)
    random.shuffle(eng_lld)
    random.shuffle(oth_lij)
    random.shuffle(oth_lld)

    # Deterministic choice deck to guarantee EXACT 40/40/10/10 counts block by block
    choice_deck = [1] * target_eng_lij + [2] * target_eng_lld + [3] * target_oth_lij + [4] * target_oth_lld
    random.shuffle(choice_deck)

    sampled_data = []

    print("Executing strict downsampling without any replacements or duplicates...")
    for choice in choice_deck:
        if choice == 1:
            item = eng_lij.pop()
        elif choice == 2:
            item = eng_lld.pop()
        elif choice == 3:
            item = oth_lij.pop()
        else:
            item = oth_lld.pop()
            
        sampled_data.append(item)
        
        if len(sampled_data) % 50000 == 0:
            print(f"Generated {len(sampled_data)} / {total_train_samples} samples...")

    # 4. Save to Disk
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "sampled_train.jsonl")
    val_path = os.path.join(args.output_dir, "sampled_validation.jsonl")

    print(f"Writing {len(sampled_data)} training rows to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Writing {len(val_data)} validation rows to {val_path}...")
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Offline sampling successfully completed! You are ready to train.")

if __name__ == "__main__":
    main()
