#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create a unified continuous pre-training (CPT) dataset from raw
multilingual text folders. Loads, cleans, and merges raw documents and sentences.
"""

import os
import re
import random
import argparse
from datasets import Dataset

# --- Configuration & Seed ---
SEED = 3407
random.seed(SEED)

# --- List of Words/Phrases to Remove ---
WORDS_TO_REMOVE = [
    # Short initials/words (specifically matched in parentheses to avoid false positives)
    "(pablo)", "(ir)", "(kp)", "(rp)", "(ss)", "(if)", "(sc)", "(Cdp)", "(cdp)",
    "(mr)", "(ddc)", "(usc)", "(osp)", "(dc)", "(pdp)",
    
    # Full names and phrases (matched both in parentheses and as regular text/headings)
    "**PABLO**", "**Intervista Pablo Palfrader**", "Intervista Pablo Palfrader",
    "Pablo Palfrader", "**Pablo Palfrader**", "Christian Ferdigg",
    "Coordinaziun Pablo Palfrader", "(iaco rigo)", "iaco rigo",
    "(Foto Freddy Planinschek)", "Foto Freddy Planinschek", "Freddy Planinschek",
    "(Freddy Planinschek)", "(Lois Trebo)", "Lois Trebo", "(Badia sport)",
    "Badia sport", "(badiasport)", "badiasport", "(katia pizzinini)",
    "katia pizzinini", "(scora alta)", "scora alta"
]

# Sort by length descending to ensure longer phrases are removed first
WORDS_TO_REMOVE_SORTED = sorted(WORDS_TO_REMOVE, key=len, reverse=True)

# Compile case-insensitive regex patterns for safe replacement
CLEAN_PATTERNS = [re.compile(re.escape(word), re.IGNORECASE) for word in WORDS_TO_REMOVE_SORTED]

def is_valid_text(text):
    """Checks if a string has actual alphanumeric content (letters/numbers)."""
    return bool(re.search(r'[a-zA-Z0-9]', text))

def clean_text_content(text):
    """
    Removes forbidden words and performs thorough whitespace/formatting cleanup.
    """
    if not text:
        return ""
    
    # Apply all removal patterns
    for pattern in CLEAN_PATTERNS:
        text = pattern.sub("", text)
    
    # Process line-by-line to preserve structure but strip out empty lines/captions
    cleaned_lines = []
    for line in text.split('\n'):
        cleaned_line = line.strip()
        # If the line was entirely stripped of its author tag/credits, we skip it
        if cleaned_line and is_valid_text(cleaned_line):
            # Clean up double/multiple spaces inside the line
            cleaned_line = re.sub(r' +', ' ', cleaned_line)
            cleaned_lines.append(cleaned_line)
            
    return '\n'.join(cleaned_lines).strip()

# --- Data Loading Functions ---

def get_document_data(doc_dir):
    """
    Reads all documents in their raw structural format (JSON, HTML, LaTeX, etc.)
    Each file is loaded as a single sample to preserve context.
    """
    all_docs = []
    print(f"Loading raw documents from {doc_dir}...")
    
    # Target common document formats
    extensions = ('.md', '.emoji', '.html', '.json', '.latex', '.txt')
    
    if not os.path.exists(doc_dir):
        print(f"  Directory not found: {doc_dir}")
        return all_docs

    for root, _, files in os.walk(doc_dir):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            cleaned = clean_text_content(content)
                            if cleaned and is_valid_text(cleaned):
                                all_docs.append(cleaned)
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
                    
    return all_docs

def get_sentence_data(sentence_dir, auth_only=False):
    """
    Reads sentence-level files line-by-line.
    """
    all_sentences = []
    print(f"Loading sentences from {sentence_dir} (auth_only={auth_only})...")
    
    if not os.path.exists(sentence_dir):
        print(f"  Directory not found: {sentence_dir}")
        return all_sentences

    # Subdirectories to load
    subdirs = ["auth"]
    if not auth_only:
        subdirs.append("synth")
        
    for subdir in subdirs:
        dir_path = os.path.join(sentence_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"  Subdirectory '{subdir}' not found in {sentence_dir}")
            continue
            
        print(f"  Processing sentence subdirectory: {subdir}")
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            count = 0
                            for line in f:
                                line = line.strip()
                                if line:
                                    cleaned = clean_text_content(line)
                                    if cleaned and is_valid_text(cleaned):
                                        all_sentences.append(cleaned)
                                        count += 1
                            if count > 0:
                                print(f"    - {os.path.basename(file_path)}: {count:,} cleaned samples")
                    except Exception as e:
                        print(f"  Error reading {file_path}: {e}")
                        
    return all_sentences

def get_ligurian_data(lig_dir):
    """
    Reads Ligurian sentence-level files line-by-line (already all authentic).
    """
    all_sentences = []
    print(f"Loading Ligurian sentences from {lig_dir}...")
    
    if not os.path.exists(lig_dir):
        print(f"  Directory not found: {lig_dir}")
        return all_sentences

    # Target extensions for the Ligurian dataset
    target_exts = ('.lij', '.eng', '.ita', '.spa', '.fra', '.txt')
    
    for root, _, files in os.walk(lig_dir):
        for file in files:
            if file.endswith(target_exts):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        count = 0
                        for line in f:
                            line = line.strip()
                            if line:
                                cleaned = clean_text_content(line)
                                if cleaned and is_valid_text(cleaned):
                                    all_sentences.append(cleaned)
                                    count += 1
                        if count > 0:
                            rel_path = os.path.relpath(file_path, lig_dir)
                            print(f"    - {rel_path}: {count:,} cleaned samples")
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
                    
    return all_sentences

def main():
    parser = argparse.ArgumentParser(description="Load, clean, and merge new CPT datasets.")
    parser.add_argument("--auth_only", action="store_true", 
                        help="Only include authentic data (exclude synth folder).")
    parser.add_argument("--base_dir", type=str, default="./raw_data",
                        help="Base path of the raw dataset directory containing 'data' and 'linguarian-monolingua'.")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Path to save the resulting combined Hugging Face dataset (matches config.py default).")
    args = parser.parse_args()

    # Verify input directory paths
    data_dir = os.path.join(args.base_dir, "data")
    lig_dir = os.path.join(args.base_dir, "linguarian-monolingua")
    
    if not os.path.exists(data_dir) and not os.path.exists(lig_dir):
        raise FileNotFoundError(
            f"Paths not found. Please ensure '{args.base_dir}' exists and contains "
            f"'data' or 'linguarian-monolingua' subdirectories."
        )

    # 1. Load and clean all components
    new_docs = get_document_data(os.path.join(data_dir, "document"))
    new_sentences = get_sentence_data(os.path.join(data_dir, "sentence"), auth_only=args.auth_only)
    ligurian_sentences = get_ligurian_data(lig_dir)
    
    combined_data = new_docs + new_sentences + ligurian_sentences
    
    if not combined_data:
        print("Error: No data successfully loaded or cleaned. Exiting.")
        return

    print(f"\nSummary of Data Loading & Cleaning:")
    print(f"  - Document level samples: {len(new_docs):,}")
    print(f"  - Sentence level samples: {len(new_sentences):,}")
    print(f"  - Ligurian level samples: {len(ligurian_sentences):,}")
    print(f"  - Total samples loaded:   {len(combined_data):,}")
    
    # 2. Create Hugging Face Dataset
    print("\nCreating Hugging Face Dataset...")
    ds = Dataset.from_dict({"text": combined_data})
    print(f"Original Dataset Size: {len(ds):,}")
    
    # Filter out extremely long samples (keep only under 20k characters as per guidelines)
    ds = ds.filter(lambda x: len(x['text']) < 20000)
    print(f"Filtered Dataset Size (len < 20k): {len(ds):,}")
    
    # 3. Split into Train and Test (99% / 1%)
    print("\nSplitting into train/test (99%/1%)...")
    split_ds = ds.train_test_split(test_size=0.01, seed=SEED)
    
    # 4. Save to disk
    print(f"Saving combined dataset to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    split_ds.save_to_disk(args.output_dir)
    print(f"\nSuccess! Unified dataset saved and ready for training at: {args.output_dir}")

if __name__ == "__main__":
    main()
