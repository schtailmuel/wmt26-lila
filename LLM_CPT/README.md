# Continuous Pre-training (CPT) for Gemma 4 (31B) with Ladin (lld)

This codebase performs continuous pre-training on the Gemma-4 31B model using the Unsloth library for optimized training speed and memory efficiency.

---

## 🛠️ Setup & Installation

It is recommended to use a virtual environment with CUDA support. Install all required dependencies (natively pinned for absolute determinism) by running:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Update **`config.py`** to configure model names, learning rates, sequence lengths, dataset directories, or adapter modules.
* The configuration has been structured as a general template with detailed comments guiding you on how to set up parameters for both a **real run** and a **rapid test run**.

---

## 🚀 Execution & Usage

Run the pre-training script natively:
```bash
python train.py
```

The script will:
1. Load the model (supports standard 16-bit Full FT or 4-bit LoRA/QLoRA quantization).
2. Attach LoRA adapters matching your config projections (if `USE_LORA = True`).
3. Load the pre-processed monolingual Ladin text corpus from disk.
4. Perform next-token prediction continuous pre-training with optimized Unsloth kernels.
5. Save the trained checkpoints and metrics (perplexity and loss callback) to your configured output directory.

### Running CPT Inference:
To test and generate completions from your trained CPT model, run:
```bash
python inference.py
```
* *The script will automatically scan your configured output directory, detect your saved checkpoint, and perform autocomplete generation natively.*

---

## 🧪 Rapid Dry-Run Verification

We have created a lightweight, fully programmatic testing suite directly on disk so you can verify that the CPT pipeline, SFT Trainer, and inference generators work flawlessly on your server in under a minute without downloading heavy datasets:

### Step 1: Create the Mock CPT Dataset
Run this script to generate a 2-split mock CPT dataset inside `test_cpt_dataset/` containing monolingual Ladin sentences:
```bash
python3 -c "
from datasets import Dataset, DatasetDict
import os

train_data = {'text': [
    'Le vënt sffla dales parëis dles munts.',
    'La natöra sará dagnora na gran amia por nos.',
    'La Val Badia é n post dret bel da bocia y iade.',
    'Ël é important da tigní sö la devuziun y la cultura.',
    'Le tëmp sará dagnora n bel mister.'
]}
val_data = {'text': [
    'Düc i di é dret biei tl dles dolomites.',
    'I un dagnora dret d’arjigné ca les spises.'
]}

ds = DatasetDict({
    'train': Dataset.from_dict(train_data),
    'test': Dataset.from_dict(val_data)
})

os.makedirs('test_cpt_dataset', exist_ok=True)
ds.save_to_disk('test_cpt_dataset')
print('Mock CPT dataset successfully saved to ./test_cpt_dataset!')
"
```

### Step 2: Enable CPT Test Profile
In **`config.py`**, set the following parameters:
* `MODEL_NAME = "unsloth/gemma-2-2b-it"`
* `MAX_SEQ_LENGTH = 512`
* `DATA_DIR = "./test_cpt_dataset"`
* `OUTPUT_DIR = "./test_output_cpt"`
* `MAX_STEPS = 2`
* `SAVE_STEPS = 2`
* `EVAL_STEPS = 2`

### Step 3: Run CPT Test Training
```bash
python train.py
```
*SFTTrainer will tokenize your test split, run exactly 2 steps of pre-training, and write `checkpoint-2` to `./test_output_cpt`.*

### Step 4: Run CPT Test Inference
```bash
python inference.py
```
*The inference script will automatically detect `./test_output_cpt/checkpoint-2` and perform a quick completion test on the model!*
