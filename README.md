# WMT26 - LiLa 🐑

Repository accompanying the **LiLa** submission to the **WMT 2026 General Machine Translation Task**.

## Scope

This repository contains the code and workflow used for:

- NLLB fine-tuning for English to Ladin (Val Badia) (`eng_Latn` → `lld_Latn`)

## Main artifact

- `nllb1_3B_eng_lld.ipynb`: end-to-end training and inference notebook (fine-tuning, model export, WMT26 inference generation, and output validation).

## Released models (Hugging Face)

- Fine-tuned NLLB model (eng-lij): https://huggingface.co/ConseggioLigure/nllb1.3B-eng-lij
- Fine-tuned NLLB model (eng-lld): https://huggingface.co/ConseggioLigure/nllb1.3B-eng-lld

## Notes

- The notebook is designed for a Colab-style GPU environment.
- Secrets (e.g., `HF_TOKEN`) and dataset/output paths are read from environment variables where possible.
