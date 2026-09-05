import torch
import os
import math
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from config import Config
from datasets import load_from_disk

def load_cpt_data(dataset_dir):
    """
    Loads a pre-processed Hugging Face Dataset from disk.
    """
    print(f"Loading pre-processed dataset from {dataset_dir}...")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found.")
    
    # Load the already split dataset from disk
    ds = load_from_disk(dataset_dir)
    return ds

# Optimization for Ampere GPUs (A100)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class PerplexityCallback(TrainerCallback):
    """
    A callback that computes perplexity from loss and evaluation loss.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Calculate Training Perplexity
            if "loss" in logs:
                try:
                    logs["perplexity"] = math.exp(logs["loss"])
                except OverflowError:
                    logs["perplexity"] = float("inf")
            # Calculate Evaluation Perplexity if eval_loss is present in logs
            if "eval_loss" in logs:
                try:
                    logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except OverflowError:
                    logs["eval_perplexity"] = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            try:
                metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")

def main():
    # 0. Check configuration
    if not Config.USE_LORA and Config.LOAD_IN_4BIT:
        raise ValueError("Full fine-tuning (USE_LORA=False) is NOT compatible with 4-bit quantization (LOAD_IN_4BIT=True). "
                         "Please set Config.LOAD_IN_4BIT = False in config.py.")

    # 1. Load Model and Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = Config.MODEL_NAME,
        max_seq_length = Config.MAX_SEQ_LENGTH,
        load_in_4bit = Config.LOAD_IN_4BIT,
        # token = "hf_...", # Use if model is gated
    )

    # 2. Add LoRA Adapters if enabled
    if Config.USE_LORA:
        print("Using LoRA fine-tuning...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = Config.LORA_R,
            target_modules = Config.LORA_TARGET_MODULES,
            lora_alpha = Config.LORA_ALPHA,
            lora_dropout = Config.LORA_DROPOUT,
            bias = Config.BIAS,
            use_gradient_checkpointing = Config.USE_GRADIENT_CHECKPOINTING,
            random_state = Config.SEED,
            use_rslora = False,
            loftq_config = None,
        )
    else:
        print("Using FULL fine-tuning...")
        # For full fine-tuning, we MUST ensure all parameters are trainable.
        # Unsloth may default to requires_grad=False for some modules.
        for param in model.parameters():
            param.requires_grad = True
        
        # Explicitly enable gradient checkpointing for full fine-tuning
        if Config.USE_GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable()

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,d} ({100 * trainable_params / total_params:.2f}% of total)")

    # 3. Load Dataset (Split into train and test)
    dataset = load_cpt_data(Config.DATA_DIR)
    print(f"Dataset loaded. Train size: {len(dataset['train'])}, Eval size: {len(dataset['test'])}")

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = Config.DATASET_TEXT_FIELD,
        max_seq_length = Config.MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Enabled for high-efficiency CPT
        callbacks = [PerplexityCallback()],
        args = TrainingArguments(
            per_device_train_batch_size = Config.BATCH_SIZE,
            gradient_accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps = 250,
            # max_steps = Config.MAX_STEPS if Config.NUM_EPOCHS is None else None,
            num_train_epochs=Config.NUM_EPOCHS if Config.NUM_EPOCHS is not None else None,
            learning_rate = Config.LEARNING_RATE,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = Config.LOGGING_STEPS,
            optim = "adamw_torch",
            weight_decay = Config.WEIGHT_DECAY,
            lr_scheduler_type = Config.LR_SCHEDULER_TYPE,
            seed = Config.SEED,
            output_dir = Config.OUTPUT_DIR,
            save_steps = 0.25,
            eval_steps = 0.25,
            eval_strategy = "steps",
            save_total_limit = Config.SAVE_TOTAL_LIMIT,
            load_best_model_at_end = Config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model = Config.METRIC_FOR_BEST_MODEL,
            greater_is_better = False,
            report_to = "none",
            dataloader_drop_last = True
        ),
    )

    # 5. Train
    print("Starting training...")
    # Robust check for existing checkpoints
    last_checkpoint = None
    if os.path.exists(Config.OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(Config.OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            print(f"Found existing checkpoints in {Config.OUTPUT_DIR}")
            print("Note: If you changed training configuration (e.g., from LoRA to Full FT), "
                  "resuming will fail. Delete the output directory if you want to start fresh.")
            last_checkpoint = True
    
    try:
        trainer_stats = trainer.train(resume_from_checkpoint = last_checkpoint)
    except ValueError as e:
        if "size of optimizer's group" in str(e):
            print("\nERROR: Checkpoint incompatibility detected!")
            print("The existing optimizer state does not match the current model configuration.")
            print("Please delete the checkpoints in your OUTPUT_DIR and restart training.\n")
        raise e
    print("Training completed.")

    # 6. Save Model
    save_path = os.path.join(Config.OUTPUT_DIR, "lora_model" if Config.USE_LORA else "full_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # Optional: save to GGUF or Merged
    # model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

if __name__ == "__main__":
    import os
    main()
