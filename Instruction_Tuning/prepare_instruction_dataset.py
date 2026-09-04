import os
import json
import random
import argparse
from pathlib import Path

def process_datasets(input_files, out_dir, seed=42):
    random.seed(seed)
    
    eng_lij = []
    eng_lld = []
    others = []
    
    print("Reading and categorizing input files...")
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                src = data.get("src_lang")
                tgt = data.get("tgt_lang")
                
                if src == "eng" and tgt in ["lij", "lij_Latn"]:
                    eng_lij.append(data)
                elif src == "eng" and tgt == "lld":
                    eng_lld.append(data)
                else:
                    others.append(data)
                    
    print(f"Loaded samples -> eng-lij: {len(eng_lij)}, eng-lld: {len(eng_lld)}, others: {len(others)}")
    
    # Shuffle each category
    random.shuffle(eng_lij)
    random.shuffle(eng_lld)
    random.shuffle(others)
    
    # Create Validation Set (2000 samples, roughly proportional)
    val_data = []
    
    val_eng_lij = min(800, len(eng_lij))
    val_eng_lld = min(800, len(eng_lld))
    val_others = min(400, len(others))
    
    val_data.extend(eng_lij[:val_eng_lij])
    eng_lij = eng_lij[val_eng_lij:]
    
    val_data.extend(eng_lld[:val_eng_lld])
    eng_lld = eng_lld[val_eng_lld:]
    
    val_data.extend(others[:val_others])
    others = others[val_others:]
    
    random.shuffle(val_data)
    
    # Create Train Set using sliding window deduplication and probabilistic ratios
    train_data = []
    recent_hashes = [] # Sliding window of recent hashes
    window_size = 100 # Ensure no duplicate in a window of 100 samples (safe for 4096 context)
    stuck_counter = 0
    
    print("Sampling training set with probabilistic ratios and local deduplication...")
    
    while eng_lij or eng_lld or others:
        weights = [
            0.4 if eng_lij else 0.0,
            0.4 if eng_lld else 0.0,
            0.2 if others else 0.0
        ]
        total_w = sum(weights)
        if total_w == 0:
            break
            
        weights = [w / total_w for w in weights]
        choice = random.choices([1, 2, 3], weights=weights)[0]
        
        pool = eng_lij if choice == 1 else (eng_lld if choice == 2 else others)
        
        candidate = pool[-1]
        cand_hash = candidate.get("output_hash")
        
        if cand_hash in recent_hashes:
            swap_idx = random.randint(0, len(pool)-1)
            pool[-1], pool[swap_idx] = pool[swap_idx], pool[-1]
            stuck_counter += 1
            if stuck_counter > 50:
                # If we get stuck, just pop it and ignore the deduplication rule for this one to prevent infinite loops
                item = pool.pop()
                train_data.append(item)
                recent_hashes.append(cand_hash)
                if len(recent_hashes) > window_size:
                    recent_hashes.pop(0)
                stuck_counter = 0
            continue
            
        stuck_counter = 0
        item = pool.pop()
        train_data.append(item)
        
        recent_hashes.append(cand_hash)
        if len(recent_hashes) > window_size:
            recent_hashes.pop(0)
            
        if len(train_data) % 100000 == 0:
            print(f"Sampled {len(train_data)} training samples...")
            
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "validation.jsonl")
    
    print(f"Saving {len(train_data)} training samples to {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saving {len(val_data)} validation samples to {val_path}")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Dataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare instruction tuning dataset from JSONL files")
    parser.add_argument("--input_files", nargs="+", required=True, help="Input JSONL files (e.g. datasets/instruction_tuning/mt-ift-lij.jsonl)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for train.jsonl and validation.jsonl")
    args = parser.parse_args()
    
    process_datasets(args.input_files, args.out_dir)