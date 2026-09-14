"""Lab 5: FP8 weights on the same vLLM serving stack as Lab 1.

Lab 1 measured BF16 + vLLM. Lab 2 measured FP8 on bare HuggingFace/torchao.
Those two numbers never sat on the same serving path. This script is the
missing joint: load Qwen2.5-3B-Instruct into vLLM with --quantization fp8
and rerun Lab 1's exact 32-concurrency (plus the same concurrency sweep).

Lab 2 never persisted a checkpoint — torchao quantized in-process and
discarded the tensors. vLLM cannot load that in-memory format. The serving
equivalent is vLLM's online FP8 path: same HF BF16 checkpoint, same Ada
sm_89 Tensor Cores, E4M3 W8A8 with dynamic activation scales (CUTLASS, not
the Marlin fallback used on GPUs without native FP8). That is the closest
production counterpart of Lab 2's fp8-dynamic path, not a byte-identical
reload of the torchao tensors (vLLM online FP8 is per-tensor; torchao
fp8-dynamic was per-row).

KV-cache dtype stays auto/fp16. Mixing kv_cache_dtype=fp8 would confound
weight quantization with a different lever.

Headline comparison (c=32, 32 requests, 128 forced tokens, prefix cache off):
  BF16 + naive HF  (Lab 1 baseline)
  BF16 + vLLM      (Lab 1 result)
  FP8  + vLLM      (this script)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 128
MAX_MODEL_LEN = 2048
CONCURRENCIES = [1, 4, 8, 16, 32, 64]
HERE = Path(__file__).resolve().parent
LAB1 = HERE.parent / "lab1_serving"
RESULTS_PATH = HERE / "fp8_vllm_results.json"
COMPARISON_PATH = HERE / "comparison.json"
PLOT_PATH = HERE / "fp8_vs_bf16_serving.png"
BASELINE_RESULTS_PATH = LAB1 / "baseline_results.json"
BF16_VLLM_RESULTS_PATH = LAB1 / "vllm_results.json"

PROMPTS = [
    "Explain what a KV cache is in large language model inference.",
    "What is continuous batching in LLM serving?",
    "Explain the difference between prefill and decode.",
    "Why does LLM inference become memory bound during decoding?",
    "What is paged attention?",
    "Explain GPU memory bandwidth in simple terms.",
    "What is throughput in an inference server?",
    "What is request latency?",
]


def build_prompts(n: int) -> list[str]:
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


def apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def nvidia_smi_snapshot() -> dict | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        name, used, total = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
        return {
            "gpu": name,
            "smi_used_mib": int(used),
            "smi_total_mib": int(total),
        }
    except Exception:
        return None


class EngineLogCatcher(logging.Handler):
    """Grab KV-cache sizing lines that vLLM prints during engine init."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(
            needle in msg
            for needle in (
                "GPU KV cache size",
                "Maximum concurrency",
                "Available KV cache",
                "quantization",
                "Using fp8",
                "FP8",
                "Marlin",
                "Cutlass",
                "CUTLASS",
            )
        ):
            self.lines.append(msg)


def install_log_catcher() -> EngineLogCatcher:
    catcher = EngineLogCatcher()
    logging.getLogger().addHandler(catcher)
    logging.getLogger("vllm").addHandler(catcher)
    return catcher


async def run_one(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
) -> tuple[float, int]:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    final = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        final = output
    latency = time.perf_counter() - start
    if final is None or not final.outputs:
        return latency, 0
    return latency, len(final.outputs[0].token_ids)


async def run_concurrency(
    engine: AsyncLLMEngine,
    prompts: list[str],
    concurrency: int,
    sampling_params: SamplingParams,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def bound(prompt: str) -> tuple[float, int]:
        async with semaphore:
            return await run_one(engine, prompt, sampling_params)

    wall_start = time.perf_counter()
    pairs = await asyncio.gather(*[bound(p) for p in prompts])
    wall_time = time.perf_counter() - wall_start

    latencies = np.asarray([p[0] for p in pairs], dtype=np.float64)
    tokens = int(sum(p[1] for p in pairs))
    n = len(prompts)

    return {
        "concurrency": concurrency,
        "num_requests": n,
        "total_time_s": float(wall_time),
        "mean_e2e_s": float(np.mean(latencies)),
        "p50_e2e_s": float(np.percentile(latencies, 50)),
        "p95_e2e_s": float(np.percentile(latencies, 95)),
        "request_throughput_qps": float(n / wall_time),
        "token_throughput_tok_s": float(tokens / wall_time),
        "generated_tokens": tokens,
        "e2e_latencies_s": latencies.tolist(),
    }


def load_lab1() -> tuple[dict | None, dict | None]:
    baseline = (
        json.loads(BASELINE_RESULTS_PATH.read_text())
        if BASELINE_RESULTS_PATH.exists()
        else None
    )
    bf16_vllm = (
        json.loads(BF16_VLLM_RESULTS_PATH.read_text())
        if BF16_VLLM_RESULTS_PATH.exists()
        else None
    )
    return baseline, bf16_vllm


def three_column(baseline: dict | None, bf16: dict | None, fp8_row: dict) -> dict:
    """Headline c=32 comparison used by the README table."""
    bf16_row = None
    if bf16 is not None:
        bf16_row = bf16.get("headline_concurrency_32") or next(
            (r for r in bf16.get("sweep", []) if r["concurrency"] == 32),
            None,
        )

    def pack(name: str, src: dict | None) -> dict:
        if src is None:
            return {"name": name, "available": False}
        return {
            "name": name,
            "available": True,
            "p95_e2e_s": src["p95_e2e_s"],
            "p50_e2e_s": src["p50_e2e_s"],
            "qps": src["request_throughput_qps"],
            "tok_s": src["token_throughput_tok_s"],
            "generated_tokens": src.get("generated_tokens"),
        }

    cols = [
        pack("BF16 + naive HF (Lab 1 baseline)", baseline),
        pack("BF16 + vLLM (Lab 1)", bf16_row),
        pack("FP8 + vLLM (Lab 5)", fp8_row),
    ]
    naive_qps = cols[0]["qps"] if cols[0]["available"] else None
    bf16_qps = cols[1]["qps"] if cols[1]["available"] else None
    fp8_qps = cols[2]["qps"]
    return {
        "concurrency": 32,
        "num_requests": 32,
        "max_new_tokens": MAX_NEW_TOKENS,
        "columns": cols,
        "speedup_vs_naive": (fp8_qps / naive_qps) if naive_qps else None,
        "speedup_vs_bf16_vllm": (fp8_qps / bf16_qps) if bf16_qps else None,
        "qps_delta_vs_bf16_vllm_pct": (
            100.0 * (fp8_qps / bf16_qps - 1.0) if bf16_qps else None
        ),
    }


def plot_comparison(
    fp8_sweep: list[dict],
    baseline: dict | None,
    bf16: dict | None,
    headline: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    ax = axes[0]
    if bf16 is not None:
        sweep = bf16["sweep"]
        ax.plot(
            [r["request_throughput_qps"] for r in sweep],
            [r["p95_e2e_s"] for r in sweep],
            marker="o",
            color="#2563eb",
            linewidth=2,
            label="BF16 + vLLM (Lab 1)",
        )
        for r in sweep:
            if r["concurrency"] in (1, 32, 64):
                ax.annotate(
                    f"c={r['concurrency']}",
                    (r["request_throughput_qps"], r["p95_e2e_s"]),
                    textcoords="offset points",
                    xytext=(6, -12),
                    fontsize=8,
                    color="#2563eb",
                )
    ax.plot(
        [r["request_throughput_qps"] for r in fp8_sweep],
        [r["p95_e2e_s"] for r in fp8_sweep],
        marker="D",
        color="#7c3aed",
        linewidth=2,
        label="FP8 + vLLM (Lab 5)",
    )
    for r in fp8_sweep:
        if r["concurrency"] in (1, 32, 64):
            ax.annotate(
                f"c={r['concurrency']}",
                (r["request_throughput_qps"], r["p95_e2e_s"]),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=8,
                color="#7c3aed",
            )
    if baseline is not None:
        ax.scatter(
            [baseline["request_throughput_qps"]],
            [baseline["p95_e2e_s"]],
            marker="*",
            s=220,
            color="#dc2626",
            zorder=5,
            label="BF16 + naive HF c=32",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Throughput (req/s, log)")
    ax.set_ylabel("p95 end-to-end latency (s, log)")
    ax.set_title("Same serving protocol, three stacks")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    ax = axes[1]
    labels = []
    qps = []
    p95 = []
    colors = ["#dc2626", "#2563eb", "#7c3aed"]
    for col in headline["columns"]:
        if not col["available"]:
            continue
        short = col["name"].split("(")[0].strip()
        labels.append(short)
        qps.append(col["qps"])
        p95.append(col["p95_e2e_s"])
    x = np.arange(len(labels))
    width = 0.36
    bars_q = ax.bar(x - width / 2, qps, width, color=colors[: len(labels)], label="QPS")
    ax.set_ylabel("QPS (req/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Headline c=32")
    ax.grid(True, axis="y", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, p95, color="#111827", marker="s", linewidth=1.6, label="p95 (s)")
    ax2.set_yscale("log")
    ax2.set_ylabel("p95 e2e (s, log)")
    for bar, val in zip(bars_q, qps):
        ax.annotate(
            f"{val:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for xi, val in zip(x, p95):
        ax2.annotate(
            f"{val:.2f}s",
            (xi, val),
            textcoords="offset points",
            xytext=(10, 4),
            fontsize=7,
            color="#111827",
        )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


def print_table(headline: dict) -> None:
    print("\n========== Three-column serving comparison (c=32) ==========")
    print(
        f"{'stack':<32} {'p95 e2e (s)':>12} {'QPS':>10} {'tok/s':>10} {'vs naive':>10} {'vs BF16 vLLM':>14}"
    )
    naive_qps = headline["columns"][0]["qps"] if headline["columns"][0]["available"] else None
    bf16_qps = headline["columns"][1]["qps"] if headline["columns"][1]["available"] else None
    for col in headline["columns"]:
        if not col["available"]:
            print(f"{col['name']:<32} {'n/a':>12}")
            continue
        vs_naive = f"{col['qps'] / naive_qps:.2f}x" if naive_qps else "—"
        vs_bf16 = f"{col['qps'] / bf16_qps:.2f}x" if bf16_qps else "—"
        print(
            f"{col['name']:<32} {col['p95_e2e_s']:12.3f} "
            f"{col['qps']:10.3f} {col['tok_s']:10.1f} {vs_naive:>10} {vs_bf16:>14}"
        )
    delta = headline.get("qps_delta_vs_bf16_vllm_pct")
    if delta is not None:
        print(f"\nFP8 vs BF16+vLLM QPS delta: {delta:+.1f}%")
    print("============================================================")


async def async_main(args: argparse.Namespace) -> None:
    concs = [32] if args.headline_only else CONCURRENCIES
    catcher = install_log_catcher()
    smi_before = nvidia_smi_snapshot()

    print(f"Loading vLLM engine: {MODEL_NAME}  quantization=fp8")
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        dtype="bfloat16",
        quantization="fp8",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        enable_prefix_caching=False,
        disable_log_stats=True,
        enable_log_requests=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    smi_after = nvidia_smi_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        min_tokens=MAX_NEW_TOKENS,
        ignore_eos=True,
    )

    formatted = [
        apply_chat_template(tokenizer, p)
        for p in build_prompts(max(32, max(concs)))
    ]

    print("Running warmup...")
    await run_one(engine, formatted[0], SamplingParams(temperature=0.0, max_tokens=16))

    sweep: list[dict] = []
    for concurrency in concs:
        n_requests = max(32, concurrency)
        prompts = formatted[:n_requests]
        print(f"\n=== FP8 vLLM concurrency={concurrency} requests={n_requests} ===")
        row = await run_concurrency(engine, prompts, concurrency, sampling_params)
        sweep.append(row)
        print(
            f"p95={row['p95_e2e_s']:.3f}s  "
            f"QPS={row['request_throughput_qps']:.3f}  "
            f"tok/s={row['token_throughput_tok_s']:.1f}"
        )

    headline_row = next(row for row in sweep if row["concurrency"] == 32)
    baseline, bf16 = load_lab1()
    headline = three_column(baseline, bf16, headline_row)
    plot_comparison(sweep, baseline, bf16, headline)

    try:
        import torch

        cap = list(torch.cuda.get_device_capability(0))
        gpu_name = torch.cuda.get_device_name(0)
        native_fp8 = tuple(cap) >= (8, 9)
    except Exception:
        cap, gpu_name, native_fp8 = None, None, None

    try:
        import vllm

        vllm_version = vllm.__version__
    except Exception:
        vllm_version = None

    payload = {
        "model": MODEL_NAME,
        "quantization": "fp8",
        "quantization_notes": (
            "vLLM online FP8 (E4M3 W8A8, dynamic activation scales). "
            "Same HF checkpoint as Lab 1/2; not a reload of Lab 2's in-memory "
            "torchao tensors. kv_cache_dtype left at default (not fp8)."
        ),
        "dtype": "bfloat16",
        "prefix_caching": False,
        "kv_cache_dtype": "auto",
        "p95_measures": (
            "end-to-end request latency "
            "(request issued to complete response)"
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.85,
        "gpu": gpu_name,
        "sm_capability": cap,
        "fp8_native_tensor_cores": native_fp8,
        "vllm": vllm_version,
        "smi_before": smi_before,
        "smi_after_load": smi_after,
        "engine_log_lines": catcher.lines,
        "headline_concurrency_32": headline_row,
        "sweep": sweep,
        "comparison": headline,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    COMPARISON_PATH.write_text(json.dumps(headline, indent=2))

    print_table(headline)
    print(f"Plot:   {PLOT_PATH}")
    print(f"Saved:  {RESULTS_PATH}")
    print(f"Table:  {COMPARISON_PATH}")
    if catcher.lines:
        print("Engine notes:")
        for line in catcher.lines:
            print(f"  {line}")

    # EngineCore is a subprocess; returning from asyncio.run() often hangs
    # on join. Files are already on disk.
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the figure from saved JSON without loading the model",
    )
    parser.add_argument(
        "--headline-only",
        action="store_true",
        help="Skip the 1/4/8/16/64 sweep; only run the c=32 headline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot_only:
        payload = json.loads(RESULTS_PATH.read_text())
        baseline, bf16 = load_lab1()
        headline = payload.get("comparison") or three_column(
            baseline, bf16, payload["headline_concurrency_32"]
        )
        plot_comparison(payload["sweep"], baseline, bf16, headline)
        print_table(headline)
        print(f"Wrote {PLOT_PATH}")
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
