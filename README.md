# TinyLlama QLoRA Fine-Tuning on a Short Pharma PDF

A small proof-of-concept that builds an end-to-end QLoRA fine-tuning pipeline for [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) on a short pharmaceutical PDF (metformin and lipid therapy), then measures the effect with perplexity.

> **Status: proof of concept, no measurable effect.** The pipeline runs end to end, but the first training run was far too short for the adapter to change the model. See [Results](#results) for the numbers and the cause. This is not a domain-adapted model and should not be used for medical purposes.

## Overview

| Item | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Method | QLoRA (4-bit NF4, float16 compute, double quantisation) |
| LoRA | r = 16, alpha = 32, dropout = 0.05 (13.2M trainable parameters, 1.18% of the model) |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head` |
| Data | 6-page PDF, 9 cleaned paragraphs |
| Split | 7 train / 2 held-out (`train_test_split(test_size=0.15, seed=42)`) |
| Training | 3 epochs, learning rate 2e-4, batch size 1, gradient accumulation 8, fp16 |
| Hardware | Google Colab T4 |

## Pipeline

1. **Extract** text from the PDF page by page with PyMuPDF.
2. **Clean** the text: Unicode NFKC normalisation, zero-width character removal, de-hyphenation, whitespace and page-number cleanup.
3. **Split into paragraphs**, dropping paragraphs under 80 characters and exact duplicates. Saved to `paragraph_records.jsonl`.
4. **Split** into train and held-out sets (seed 42).
5. **Tokenise and pack** training text into 512-token blocks.
6. **Fine-tune** a LoRA adapter on the 4-bit base model with the Hugging Face `Trainer`.
7. **Save** the adapter and tokenizer, optionally push to the Hugging Face Hub and merge into the base weights.
8. **Evaluate** the base model against base + adapter with perplexity (final cell of `main.ipynb`).

## Setup

```bash
pip install -U pymupdf datasets transformers accelerate peft bitsandbytes sentencepiece
```

A CUDA GPU is needed for 4-bit loading. A free Colab T4 works.

## Usage

1. Open `main.ipynb` in Google Colab and select a T4 GPU (**Runtime → Change runtime type**).
2. Upload `Metformin-Lipid-Therapy-Knowledge.pdf` to `/content/`.
3. Run the notebook top to bottom in one session. It creates:
   - `/content/pharma_processed_data/` (raw pages and `paragraph_records.jsonl`)
   - `/content/pharma_tinyllama_lora_adapter/` (LoRA adapter and tokenizer)
   - `/content/pharma_tinyllama_merged_model/` (optional merged model)
4. The cell that pushes to the Hugging Face Hub needs a write token. Skip it if you don't have one.

Colab deletes `/content` when the session ends, so run the evaluation cell in the same session as training.

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

Perplexity is computed as `exp(total NLL / total predicted tokens)`, weighted by tokens rather than averaged per paragraph. Each paragraph is scored as its own sequence (no packing), with `labels = input_ids`, `model.eval()` and `torch.no_grad()`. The base model and the adapter model are scored on the same text. Training-set perplexity is reported as well, to help detect memorisation.

## Results

| Set | Tokens | Base PPL | Fine-tuned PPL | Change |
|---|---|---|---|---|
| Held-out (2 paragraphs) | 256 | 5.46 | 5.46 | 0.0% |
| Train (7 paragraphs) | 3,412 | 7.85 | 7.85 | 0.0% |

Per-paragraph held-out perplexity (base → fine-tuned): 9.13 → 9.13 and 3.71 → 3.71.

**No measurable change.** The training run was 3 optimizer steps (one per epoch, about 26 seconds, final training loss 2.14). The 7 training paragraphs came to 3,412 tokens, or 6 blocks of 512, and with a gradient accumulation of 8 that is a single optimizer step per epoch. After training, the largest absolute value in any `lora_B` matrix was about `1.2e-4` (these start at exactly 0), so the adapter barely moved from its initial state and the adapted model behaves almost exactly like the base model.

### How to read this

- The held-out set is only 2 paragraphs, so any held-out number here is noisy.
- Perplexity measures fit to this document's wording, not factual accuracy or clinical usefulness.
- A larger drop on train than on held-out after a longer run would indicate memorisation, not domain learning.

## Limitations

- Very small corpus (one short PDF), so a model trained on it can overfit quickly and cannot learn general domain knowledge.
- Training was far too short to change the adapter (see Results).
- Training packed paragraphs into 512-token blocks and dropped the leftover tail beyond the last full block, so a small part of the training text was never seen.
- Sample generations from the first run still contained factual errors, for example describing metformin as a sulfonylurea and inventing quotes from named doctors. They looked like ordinary base-model text, which is consistent with the adapter having had no effect.
- **Not for medical use.** Outputs can be wrong and must not be used for clinical decisions.

## Next steps

- Re-run with more optimizer steps (for example `gradient_accumulation_steps = 1` and about 10 epochs) and re-check the `lora_B` magnitudes and perplexity.
- Use more text, such as open-access papers, instead of a 6-page PDF.
- Evaluate with k-fold cross-validation over paragraphs instead of 2 held-out paragraphs, and do not tune hyperparameters against the same held-out text.

## Repository contents

```
main.ipynb                              # training, inference and perplexity evaluation
Metformin-Lipid-Therapy-Knowledge.pdf   
README.md
```

## Acknowledgements

- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- Hugging Face `transformers`, `peft`, `bitsandbytes`, `datasets`
