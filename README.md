# Pharma TinyLlama: Domain Adaptation with QLoRA

A small proof-of-concept that fine-tunes [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) on a short pharmaceutical PDF (metformin and lipid therapy) using QLoRA, then measures the effect with perplexity on held-out text.

> **Status:** proof of concept. The corpus is tiny (9 paragraphs), so results are noisy and should not be read as evidence of general domain improvement.

## Overview

| Item | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Method | QLoRA (4-bit NF4, float16 compute, double quantisation) |
| LoRA | r = 16, alpha = 32, dropout = 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head` |
| Data | 6-page PDF, 9 cleaned paragraphs |
| Split | 7 train / 2 held-out (`train_test_split(test_size=0.15, seed=42)`) |
| Training | 3 epochs, lr 2e-4, batch size 1, grad accumulation 8, fp16 |
| Hardware | Google Colab T4 |
| Adapter | `piyushg1122/pharma-tinyllama-domain-lora` (private on the Hugging Face Hub) |

## Pipeline

1. **Extract** text from the PDF page by page with PyMuPDF.
2. **Clean** the text: Unicode NFKC normalisation, zero-width character removal, de-hyphenation, whitespace and page-number cleanup.
3. **Split into paragraphs**, dropping paragraphs under 80 characters and exact duplicates. Saved to `paragraph_records.jsonl`.
4. **Split** into train and held-out sets (seed 42).
5. **Tokenise and pack** training text into 512-token blocks.
6. **Fine-tune** a LoRA adapter on the 4-bit base model with the Hugging Face `Trainer`.
7. **Save** the adapter, push it to the Hub, and optionally merge it into the base weights.
8. **Evaluate** base vs. base + adapter with perplexity.

## Setup

```bash
pip install -U pymupdf datasets transformers accelerate peft bitsandbytes sentencepiece
```

A CUDA GPU is needed for 4-bit loading (a free Colab T4 works).

## Usage

1. Upload the source PDF to `/content/Metformin-Lipid-Therapy-Knowledge.pdf`.
2. Run `main.ipynb` top to bottom. It produces:
   - `/content/pharma_processed_data/` (raw pages and `paragraph_records.jsonl`)
   - `/content/pharma_tinyllama_lora_adapter/` (LoRA adapter and tokenizer)
   - `/content/pharma_tinyllama_merged_model/` (optional merged model)

### Loading the adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "/content/pharma_tinyllama_lora_adapter")
tok = AutoTokenizer.from_pretrained(MODEL)
model.eval()
```

## Evaluation

Perplexity is computed as `exp(total NLL / total predicted tokens)`, weighted by tokens rather than averaged per paragraph. Each paragraph is scored as its own sequence (no packing), with `labels = input_ids`, `model.eval()` and `torch.no_grad()`. The base model and the adapter model are each loaded as a fresh 4-bit copy and scored on the same text. Train perplexity is also reported to help detect memorisation.

### Results

Fill in after running the evaluation cell.

| Set | Tokens | Base PPL | Fine-tuned PPL | Change |
|---|---|---|---|---|
| Held-out (2 paragraphs) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Train (7 paragraphs) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### How to read this

- With only 2 held-out paragraphs, the held-out number is noisy. Treat it as a small proof of concept, not proof of improvement.
- A large drop on train with little or no drop on held-out indicates memorisation rather than domain learning.
- If held-out perplexity gets worse, report that honestly. Do not tune hyperparameters against the same 2 paragraphs. Try fewer epochs, a lower learning rate or rank, or removing `lm_head` from the LoRA targets, and ideally use more text or k-fold cross-validation over the 9 paragraphs.

## Limitations

- Very small corpus (one short PDF), so the model can overfit quickly.
- Training packed paragraphs into 512-token blocks and dropped the leftover tail beyond the last full block, so a small part of the training text was never seen.
- Perplexity measures fit to this document's wording, not factual accuracy or clinical usefulness.
- Not for medical use. Outputs can be wrong and must not be used for clinical decisions.

## Repository contents

```
main.ipynb                      # training and inference notebook
README.md
pharma_processed_data/          # cleaned corpus (generated)
pharma_tinyllama_lora_adapter/  # LoRA adapter (generated)
```

## Acknowledgements

- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- Hugging Face `transformers`, `peft`, `bitsandbytes`, `datasets`
