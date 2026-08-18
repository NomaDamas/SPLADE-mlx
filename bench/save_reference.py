"""Save fp32 CPU torch reference outputs for torch<->MLX parity testing.

For each P0 model, over 32 fixed parity texts:
  - input_ids / attention_mask (so the MLX side can bypass tokenizer diffs)
  - SPLADE sparse vectors (32, vocab) fp32
  - full MLM logits for the first PARITY_LOGITS_COUNT inputs

Writes data/reference/{model_key}.npz
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bench.bench_torch import splade_pool
from bench.workloads import (
    P0_MODELS,
    PARITY_LOGITS_COUNT,
    PARITY_MAX_LEN,
    REFERENCE_DIR,
    parity_texts,
)


def main() -> None:
    texts = parity_texts()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    for model_key, spec in P0_MODELS.items():
        print(f"=== reference: {model_key} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
        model = AutoModelForMaskedLM.from_pretrained(spec["hf_id"], dtype=torch.float32)
        model.eval()

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=PARITY_MAX_LEN,
            return_tensors="pt",
        )
        inputs = {
            k: v
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        with torch.inference_mode():
            logits = model(**inputs).logits.float()
            sparse = splade_pool(logits, inputs["attention_mask"])

        active = (sparse > 0).sum(dim=1).float()
        print(
            f"  seq_len={enc['input_ids'].shape[1]}  "
            f"active terms/vec: mean {active.mean():.1f}  max {active.max():.0f}",
            flush=True,
        )

        np.savez_compressed(
            REFERENCE_DIR / f"{model_key}.npz",
            input_ids=enc["input_ids"].numpy(),
            attention_mask=enc["attention_mask"].numpy(),
            sparse=sparse.numpy().astype(np.float32),
            logits_subset=logits[:PARITY_LOGITS_COUNT].numpy().astype(np.float32),
        )
        del model
    print(f"references written to {REFERENCE_DIR}", flush=True)


if __name__ == "__main__":
    main()
