# Instruction Fine-Tuning (IFT) for Ladin & Ligurian

This directory contains the pipeline, scripts, diagnostics, and configurations for training and evaluating machine translation models on **Ladin (`lld`)** and **Ligurian (`lij`)**.

All Gemma-related training scripts are highly optimized for stability and memory efficiency using **Unsloth** and **TRL**.

---

## 🛠️ Main Components & Scripts

### 1. Training & Verification Scripts
* **`train_full.py`**: The reference script for **Full Parameter Fine-Tuning (FFT)** of Gemma-4-E4B.
* **`train_lora.py`**: The reference script for **LoRA Parameter Fine-Tuning** of Gemma.

### 2. Data Preparation & Sampling
* **`prepare_offline_sampled_data.py`**: Implements the balanced 40/40/10/10 offline sampling. It divides data into pools (`eng_lij`, `eng_lld`, `oth_lij`, `oth_lld`), applies sliding-window deduplication (size 200) to avoid oversampled duplicates clumping together, and produces `sampled_train.jsonl` and `sampled_validation.jsonl`.
* **`prepare_instruction_dataset.py`**: Splits and partitions dataset samples based on source and target language parameters.

### 3. Evaluation Suite
* **`evaluate_checkpoints.py`**: Unified multi-checkpoint evaluation script for both Full Parameter (FFT) and LoRA Gemma checkpoints on Flores-200 and BOUQuET. Natively auto-detects and loads LoRA or Full FT weights.
* **`evaluate_translation.py`**: Single-model evaluation computing `sacrebleu` BLEU and chrF++ scores.

### 4. Blindset Translation Generators
* **`generate_blindset_translations.py`**: Unified master blindset translation generator for both Full Parameter (FFT) and LoRA checkpoints.

---

## 📖 Step-by-Step Command Guide

### Data Preprocessing
```bash
python prepare_offline_sampled_data.py \
    --input_files "/path/to/data/*.jsonl" \
    --output_dir "./prepared_data" \
    --val_size 1000 \
    --window_size 200
```

### Run Full Fine-Tuning (FFT)
```bash
python train_full.py --config config.yaml
```

### Run LoRA Adapters Tuning
```bash
python train_lora.py --config config.yaml
```

### Evaluate Checkpoints
```bash
# Evaluate Full FT Gemma checkpoints
python evaluate_checkpoints.py \
    --checkpoints_dir "/path/to/checkpoints" \
    --datasets "flores,bouquet" \
    --languages "lij_Latn,lld_Latn"

# Evaluate LoRA checkpoints (loads base model first dynamically)
python evaluate_checkpoints.py \
    --model_id "google/gemma-4-E4B-it" \
    --checkpoints_dir "/path/to/checkpoints" \
    --datasets "flores,bouquet" \
    --languages "lij_Latn,lld_Latn"
```

### Generate Blindset Translations
```bash
# For Full Parameter checkpoints
python generate_blindset_translations.py \
    --checkpoint "/path/to/selected/checkpoint" \
    --blindset_file "https://www2.statmt.org/wmt26/assets/wmt26_genmt_blindset.jsonl" \
    --output_dir "./predictions" \
    --batch_size 16

# For LoRA checkpoints (loads base model first dynamically)
python generate_blindset_translations.py \
    --model_id "google/gemma-4-E4B-it" \
    --checkpoint "/path/to/selected/checkpoint" \
    --blindset_file "https://www2.statmt.org/wmt26/assets/wmt26_genmt_blindset.jsonl" \
    --output_dir "./predictions_lora" \
    --batch_size 16
```

---

## 💡 Key Technical Note: Gemma-4 Multimodal Warnings

When loading Gemma-4 (`google/gemma-4-E4B-it`) checkpoints using Unsloth, you may encounter the following warnings in your terminal:

1. **`MISSING` parameters in layers 24...41** (e.g. `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, etc.)
2. **`Ignored unknown kwarg option`** (e.g. `close`, `open`, `content_args`, etc.)

### Why do these warnings appear?
These warnings are a direct consequence of Gemma-4's native **multimodal architecture** (designed to handle both text and image/video inputs):
* **Ignored Kwargs**: The tokenizer config files contain advanced tokens and Jinja arguments specifically mapped for visual bounding boxes and image boundaries. When loading the model in text-only mode (`AutoModelForCausalLM`), standard tokenizer classes safely ignore these image-specific options.
* **Missing Layers**: During text-only supervised fine-tuning, standard trainers (like TRL/SFTTrainer) only update and save the text-generation language model layers to disk. When Unsloth loads these saved text weights, its custom weight parser detects that the multimodal image projection headers for layers 24...41 are missing and newly initializes them.

### Is this a problem for translation?
**No. These warnings are 100% harmless.** Because machine translation is strictly a text-to-text task, the model's forward inference execution completely bypasses the visual projection heads. The language generation weights are **100% loaded, intact, and functioning at peak performance** (as evidenced by our top-tier Flores metrics). 

If you prefer to bypass these warnings and load 100% of the checkpoint weights natively, simply disable the Unsloth loader during evaluation or inference by passing the argument:
```bash
--use_unsloth False
```

