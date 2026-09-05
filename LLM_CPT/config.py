# -*- coding: utf-8 -*-
# @Author: Lucas Domingues
# @Date:   2026-04-30 17:29:27
# @Last Modified by:   Lucas Domingues
# @Last Modified time: 2026-04-30 17:36:35
import os

class Config:
    # =====================================================================
    # 1. Training Mode Strategy
    # =====================================================================
    USE_LORA = True              # Set to False for Full FT, True for Parameter-Efficient LoRA
    
    # =====================================================================
    # 2. Base Model Parameters
    # =====================================================================
    MODEL_NAME = "google/gemma-4-E4B"    # HF repo ID or local path to base model (use unsloth/gemma-2-2b-it for test runs)
    MAX_SEQ_LENGTH = 4096                # Context sequence limit (use 512 for fast test runs)
    LOAD_IN_4BIT = True                  # Set to True for QLoRA quantized loading (must be False for Full FT)
    
    # =====================================================================
    # 3. LoRA Adapter Hyperparameters (Used when USE_LORA = True)
    # =====================================================================
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    LORA_DROPOUT = 0                     # Set to 0.0 for optimal Unsloth speedups
    BIAS = "none"
    USE_GRADIENT_CHECKPOINTING = 'unsloth'
    
    # =====================================================================
    # 4. Datasets & Checkpoint Directories
    # =====================================================================
    DATA_DIR = "./data"                  # Path to pre-processed CPT dataset on disk (use ./test_cpt_dataset for test runs)
    OUTPUT_DIR = "./output"              # Path where trained checkpoints will be saved (use ./test_output_cpt for test runs)
    DATASET_TEXT_FIELD = "text"          # Column mapping containing monolingual text data
    
    # =====================================================================
    # 5. Training, Optimization, & SFT Parameters
    # =====================================================================
    BATCH_SIZE = 2                       # Per-device SFT train batch size
    GRADIENT_ACCUMULATION_STEPS = 16     # Gradient accumulation steps to achieve desired global batch size
    LEARNING_RATE = 2e-4                 # CPT Learning rate
    MAX_STEPS = None                     # Set to integer (e.g. 2) to terminate after N steps, or None for epoch-based
    NUM_EPOCHS = 2
    WEIGHT_DECAY = 0.01
    SEED = 3407
    LR_SCHEDULER_TYPE = "linear"
    
    # =====================================================================
    # 6. Logging, Evaluation, & Saving Strategies
    # =====================================================================
    LOGGING_STEPS = 1
    SAVE_STEPS = 50                      # Step frequency to save checkpoints (use 2 for test runs)
    EVAL_STEPS = 50                      # Step frequency to calculate validation loss (use 2 for test runs)
    SAVE_TOTAL_LIMIT = 2                 # Maximum number of checkpoints to keep on disk
    LOAD_BEST_MODEL_AT_END = True        # Save and restore the best model checkpoint
    METRIC_FOR_BEST_MODEL = "eval_perplexity"
