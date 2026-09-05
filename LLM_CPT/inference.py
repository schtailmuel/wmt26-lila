from unsloth import FastLanguageModel
import torch
from config import Config
import os

def run_inference(prompt, model_path):
    # 1. Load Model and Tokenizer
    # We load the base model first, then the LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # This points to the folder with your LoRA adapters
        max_seq_length = Config.MAX_SEQ_LENGTH,
        load_in_4bit = Config.LOAD_IN_4BIT,
    )
    
    # 2. Fast Inference Mode
    FastLanguageModel.for_inference(model)

    # 3. Tokenize and Generate
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    print(f"\n--- Prompt ---\n{prompt}")
    print("\n--- Generating Response ---")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 128,
        use_cache = True,
        temperature = 0.7,
        top_p = 0.9,
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    print(response[0])

if __name__ == "__main__":
    # Determine model path based on what's available
    lora_path = os.path.join(Config.OUTPUT_DIR, "lora_model")
    full_path = os.path.join(Config.OUTPUT_DIR, "full_model")
    
    if Config.USE_LORA and os.path.exists(lora_path):
        model_path = lora_path
    elif not Config.USE_LORA and os.path.exists(full_path):
        model_path = full_path
    elif os.path.exists(lora_path):
        model_path = lora_path
    elif os.path.exists(full_path):
        model_path = full_path
    else:
        # Fallback to scan for any checkpoint-* folder inside Config.OUTPUT_DIR
        model_path = None
        if os.path.exists(Config.OUTPUT_DIR):
            for item in os.listdir(Config.OUTPUT_DIR):
                item_path = os.path.join(Config.OUTPUT_DIR, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    model_path = item_path
                    break

    if model_path is None:
        print(f"Error: No model found in {Config.OUTPUT_DIR}. Please run training first.")
    else:
        print(f"Loading model from: {model_path}")
        # Example Ladin prompt
        sample_prompt = "Le tëmp por duman sará" # "The weather for tomorrow will be..."
        run_inference(sample_prompt, model_path)
