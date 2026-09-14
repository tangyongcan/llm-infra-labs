"""Structured pruning demo with torch.nn.utils.prune.

`prune.ln_structured` only applies a *mask*. The zeroed weights are still
stored (in fact a mask buffer is added, so memory goes UP). Real savings
require physically removing the rows/columns and re-emitting smaller
matrices, or sparse kernels the GPU can actually exploit. This script
reports:

  * how many parameters were driven to zero (the "would-be" saving)
  * the perplexity cost of that sparsity
  * the actual measured memory, to show masking saves nothing by itself
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT_PATH = Path(__file__).with_name("results_prune.json")

PPL_WINDOW = 2048
PPL_NUM_WINDOWS = 24
SPARSITY_LEVELS = [0.0, 0.10, 0.20, 0.30]

# Only the MLP projections: pruning attention head projections structurally
# breaks the head layout unless whole heads are removed together.
TARGET_SUFFIXES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def gib(n: float) -> float:
    return float(n) / (1024**3)


def target_linears(model) -> list[tuple[str, torch.nn.Linear]]:
    out = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            out.append((name, module))
    return out


def sparsity_report(model) -> dict:
    total = 0
    zeros = 0
    for _, module in target_linears(model):
        w = module.weight
        total += w.numel()
        zeros += int((w == 0).sum())
    all_params = sum(p.numel() for p in model.parameters())
    return {
        "target_params": int(total),
        "target_zero_params": int(zeros),
        "target_sparsity": (zeros / total) if total else 0.0,
        "model_total_params": int(all_params),
        "model_zero_fraction": (zeros / all_params) if all_params else 0.0,
    }


@torch.inference_mode()
def perplexity(model, tokenizer) -> float:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    ids = ids[: PPL_WINDOW * PPL_NUM_WINDOWS]

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, ids.numel() - 1, PPL_WINDOW):
        window = ids[start : start + PPL_WINDOW].unsqueeze(0).to("cuda")
        if window.shape[1] < 2:
            break
        logits = model(input_ids=window).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            window[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll)
        total_tokens += int(window[:, 1:].numel())
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


@torch.inference_mode()
def decode_tok_s(model, tokenizer, new_tokens: int = 64) -> float:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is paged attention?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    model.generate(
        **inputs, max_new_tokens=8, min_new_tokens=8, do_sample=False, eos_token_id=[]
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(
        **inputs,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        eos_token_id=[],
    )
    torch.cuda.synchronize()
    return new_tokens / (time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ppl", action="store_true", help="Skip perplexity (smoke test)"
    )
    args = parser.parse_args()

    base_alloc = torch.cuda.memory_allocated()
    print(f"loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gc.collect()
    torch.cuda.empty_cache()

    layers = target_linears(model)
    print(f"pruning targets: {len(layers)} linear layers ({TARGET_SUFFIXES})")

    rows = []
    for level in SPARSITY_LEVELS:
        if level > 0.0:
            for _, module in layers:
                # n=2 -> L2 norm of each output row; dim=0 -> drop whole
                # output neurons, which is the structured (not element-wise) case.
                prune.ln_structured(module, name="weight", amount=level, n=2, dim=0)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        row = {
            "requested_sparsity": level,
            "mem_alloc_gib": gib(torch.cuda.memory_allocated() - base_alloc),
            **sparsity_report(model),
        }
        row["decode_tok_s"] = decode_tok_s(model, tokenizer)
        if not args.skip_ppl:
            row["perplexity"] = perplexity(model, tokenizer)

        print(
            f"sparsity={level:.0%} "
            f"target_zero={row['target_sparsity']:.1%} "
            f"mem={row['mem_alloc_gib']:.3f}GiB "
            f"tok/s={row['decode_tok_s']:.2f} "
            + (
                f"ppl={row['perplexity']:.4f}"
                if "perplexity" in row
                else ""
            )
        )
        rows.append(row)

        if level > 0.0:
            # Remove the reparametrization so the next level prunes from the
            # already-pruned weights rather than stacking masks.
            for _, module in layers:
                prune.remove(module, "weight")

    payload = {
        "model": MODEL_NAME,
        "method": "torch.nn.utils.prune.ln_structured (L2, dim=0, MLP only)",
        "note": (
            "Masking zeroes weights but does not shrink storage; dense GEMMs "
            "still read the zeros, so neither memory nor speed improves. "
            "Real gains need physical removal or sparse kernels."
        ),
        "sparsity_levels": SPARSITY_LEVELS,
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
