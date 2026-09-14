"""Compression pipeline benchmark: BF16 vs INT8 (bitsandbytes) vs FP8 (torchao).

Each mode must run in its own process so that CUDA memory numbers are
clean (no residue from a previously loaded model).

Measured per mode:
  * weight memory      - allocated bytes after load/quantize settles
  * peak memory        - max allocated during a decode run
  * decode tok/s       - batch 1, memory-bandwidth-bound regime
  * batch decode tok/s - batch 8, mixed regime
  * prefill tok/s      - batch 8 x 512 tokens, compute-bound regime
                         (this is where native FP8 tensor cores show up)
  * perplexity         - WikiText-2 raw test, fixed window budget

Usage:
    python compress_bench.py --mode bf16
    python compress_bench.py --mode int8
    python compress_bench.py --mode fp8-weight
    python compress_bench.py --mode fp8-dynamic
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR = Path(__file__).parent

MODES = ["bf16", "int8", "fp8-weight", "fp8-dynamic"]

DECODE_NEW_TOKENS = 128
DECODE_REPEATS = 3
PREFILL_BATCH = 8
PREFILL_SEQ_LEN = 512
PREFILL_REPEATS = 3
BATCH_DECODE_BATCH = 8

PPL_WINDOW = 2048
PPL_NUM_WINDOWS = 24

DECODE_PROMPT = (
    "Explain what a KV cache is in large language model inference, "
    "and why it matters for serving throughput."
)


def gib(n_bytes: int | float) -> float:
    return float(n_bytes) / (1024**3)


def settle() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def nvidia_smi_used_mib() -> int | None:
    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def load_model(mode: str):
    """Return (model, load_info). Only `mode` decides the numeric format."""
    info: dict = {}

    if mode == "int8":
        from transformers import BitsAndBytesConfig

        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="cuda",
        )
        info["load_s"] = time.perf_counter() - t0
        info["quantize_s"] = 0.0
        info["backend"] = "bitsandbytes LLM.int8()"
        model.eval()
        return model, info

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    info["load_s"] = time.perf_counter() - t0
    model.eval()

    if mode == "bf16":
        info["quantize_s"] = 0.0
        info["backend"] = "native bf16"
        return model, info

    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        PerRow,
        quantize_,
    )

    # lm_head stays in bf16: it is a single large GEMM whose output feeds
    # the sampler directly, and quantizing it hurts quality for no gain.
    def only_transformer_linears(module: torch.nn.Module, fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear) and "lm_head" not in fqn

    if mode == "fp8-weight":
        config = Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn)
        info["backend"] = "torchao Float8WeightOnlyConfig (e4m3, weight-only)"
    elif mode == "fp8-dynamic":
        config = Float8DynamicActivationFloat8WeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            granularity=PerRow(),
        )
        info["backend"] = (
            "torchao Float8DynamicActivationFloat8WeightConfig "
            "(e4m3 act+weight, per-row, _scaled_mm)"
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    t0 = time.perf_counter()
    quantize_(model, config, filter_fn=only_transformer_linears)
    settle()
    info["quantize_s"] = time.perf_counter() - t0
    return model, info


def count_params(model: torch.nn.Module) -> dict:
    total = 0
    nonzero = 0
    for p in model.parameters():
        n = p.numel()
        total += n
    return {"total_params": int(total), "nonzero_params": int(nonzero or total)}


@torch.inference_mode()
def bench_decode(
    model, tokenizer, batch_size: int, label: str, compiled: bool = False
) -> dict:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": DECODE_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")

    gen_kwargs = dict(do_sample=False, eos_token_id=[])
    if compiled:
        # A static KV cache keeps decode shapes constant so inductor can
        # reuse one graph instead of recompiling every step.
        gen_kwargs["cache_implementation"] = "static"

    # Warmup pays for compile / autotune / kernel JIT. Compiled runs need
    # the warmup to use the same length as the measured run.
    warmup_tokens = DECODE_NEW_TOKENS if compiled else 8
    model.generate(
        **inputs,
        max_new_tokens=warmup_tokens,
        min_new_tokens=warmup_tokens,
        **gen_kwargs,
    )
    torch.cuda.synchronize()

    durations = []
    new_tokens_total = 0
    for _ in range(DECODE_REPEATS):
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=DECODE_NEW_TOKENS,
            min_new_tokens=DECODE_NEW_TOKENS,
            **gen_kwargs,
        )
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)
        new_tokens_total = (
            out.shape[1] - inputs["input_ids"].shape[1]
        ) * batch_size

    best = min(durations)
    mean = sum(durations) / len(durations)
    return {
        f"{label}_batch": batch_size,
        f"{label}_new_tokens": int(new_tokens_total),
        f"{label}_best_s": best,
        f"{label}_mean_s": mean,
        f"{label}_tok_s": new_tokens_total / best,
    }


@torch.inference_mode()
def bench_prefill(model, tokenizer, forward=None) -> dict:
    """Compute-bound forward pass: big GEMMs, no autoregressive serialization."""
    forward = model if forward is None else forward
    vocab = int(model.config.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, vocab, (PREFILL_BATCH, PREFILL_SEQ_LEN), device="cuda"
    )

    forward(input_ids=input_ids)
    torch.cuda.synchronize()

    durations = []
    for _ in range(PREFILL_REPEATS):
        t0 = time.perf_counter()
        forward(input_ids=input_ids)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)

    best = min(durations)
    tokens = PREFILL_BATCH * PREFILL_SEQ_LEN
    return {
        "prefill_batch": PREFILL_BATCH,
        "prefill_seq_len": PREFILL_SEQ_LEN,
        "prefill_tokens": tokens,
        "prefill_best_s": best,
        "prefill_mean_s": sum(durations) / len(durations),
        "prefill_tok_s": tokens / best,
    }


@torch.inference_mode()
def bench_perplexity(model, tokenizer) -> dict:
    """Non-overlapping-window perplexity on WikiText-2 raw test."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0]

    needed = PPL_WINDOW * PPL_NUM_WINDOWS
    ids = ids[:needed]

    total_nll = 0.0
    total_tokens = 0

    for start in range(0, ids.numel() - 1, PPL_WINDOW):
        window = ids[start : start + PPL_WINDOW].unsqueeze(0).to("cuda")
        if window.shape[1] < 2:
            break
        logits = model(input_ids=window).logits.float()
        # Standard shift: predict token t+1 from position t.
        shift_logits = logits[:, :-1, :]
        shift_labels = window[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll)
        total_tokens += int(shift_labels.numel())

    mean_nll = total_nll / total_tokens
    return {
        "ppl_window": PPL_WINDOW,
        "ppl_num_windows": PPL_NUM_WINDOWS,
        "ppl_eval_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Skip the perplexity pass (for quick smoke tests)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "torch.compile the model. torchao's fp8 kernels need inductor "
            "fusion to beat bf16; eager mode pays the quant overhead raw."
        ),
    )
    parser.add_argument(
        "--only",
        default="all",
        choices=["all", "prefill", "decode1", "decode8", "ppl"],
        help="Run a single stage (for targeted comparisons)",
    )
    args = parser.parse_args()
    mode = args.mode
    stage = args.only
    suffix = "_compiled" if args.compile else ""

    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()
    smi_before = nvidia_smi_used_mib()

    print(f"[{mode}] loading {MODEL_NAME} ...")
    model, load_info = load_model(mode)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    settle()

    weight_alloc = torch.cuda.memory_allocated()
    load_peak = torch.cuda.max_memory_allocated()
    smi_after_load = nvidia_smi_used_mib()

    print(f"[{mode}] weight memory: {gib(weight_alloc - baseline_alloc):.3f} GiB")

    results: dict = {
        "mode": mode,
        "compiled": bool(args.compile),
        "stage": stage,
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "sm_capability": list(torch.cuda.get_device_capability(0)),
        "fp8_native": torch.cuda.get_device_capability(0) >= (8, 9),
        "torch": torch.__version__,
        **load_info,
        **count_params(model),
        "weight_mem_gib": gib(weight_alloc - baseline_alloc),
        "load_peak_mem_gib": gib(load_peak - baseline_alloc),
        "smi_before_mib": smi_before,
        "smi_after_load_mib": smi_after_load,
    }

    prefill_forward = None
    if args.compile:
        print(f"[{mode}] torch.compile ...")
        t0 = time.perf_counter()
        # dynamic=False so inductor specializes on the fixed bench shapes.
        prefill_forward = torch.compile(model, dynamic=False)
        model.forward = torch.compile(model.forward, dynamic=False)
        results["compile_setup_s"] = time.perf_counter() - t0

    def want(name: str) -> bool:
        return stage in ("all", name)

    if want("decode1"):
        print(f"[{mode}] decode bench (batch 1) ...")
        torch.cuda.reset_peak_memory_stats()
        results.update(
            bench_decode(model, tokenizer, 1, "decode1", compiled=args.compile)
        )
        results["decode1_peak_mem_gib"] = gib(
            torch.cuda.max_memory_allocated() - baseline_alloc
        )

    if want("decode8"):
        print(f"[{mode}] decode bench (batch {BATCH_DECODE_BATCH}) ...")
        torch.cuda.reset_peak_memory_stats()
        results.update(
            bench_decode(
                model,
                tokenizer,
                BATCH_DECODE_BATCH,
                "decode8",
                compiled=args.compile,
            )
        )
        results["decode8_peak_mem_gib"] = gib(
            torch.cuda.max_memory_allocated() - baseline_alloc
        )

    if want("prefill"):
        print(f"[{mode}] prefill bench (compute-bound) ...")
        torch.cuda.reset_peak_memory_stats()
        results.update(bench_prefill(model, tokenizer, forward=prefill_forward))
        results["prefill_peak_mem_gib"] = gib(
            torch.cuda.max_memory_allocated() - baseline_alloc
        )

    if want("ppl") and not args.skip_ppl:
        print(f"[{mode}] perplexity on WikiText-2 ...")
        results.update(bench_perplexity(model, tokenizer))

    results["smi_end_mib"] = nvidia_smi_used_mib()

    stage_tag = "" if stage == "all" else f"_{stage}"
    out_path = OUT_DIR / f"results_{mode}{suffix}{stage_tag}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n===== {mode}{suffix} =====")
    print(f"backend:          {results['backend']}")
    print(f"weight mem:       {results['weight_mem_gib']:.3f} GiB")
    for key, label in [
        ("decode1_tok_s", "decode1 tok/s"),
        ("decode8_tok_s", "decode8 tok/s"),
        ("prefill_tok_s", "prefill tok/s"),
        ("perplexity", "perplexity"),
    ]:
        if key in results:
            print(f"{label + ':':18s}{results[key]:.4f}")
    print(f"saved:            {out_path}")


if __name__ == "__main__":
    main()
