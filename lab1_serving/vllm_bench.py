"""vLLM serving benchmark: continuous batching + paged KV cache.

Uses the same prompt set and a fixed 128-token output length as
baseline_bench.py. Prefix caching is explicitly disabled so the
comparison isolates continuous batching + paged attention.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
RESULTS_PATH = Path(__file__).with_name("vllm_results.json")
PLOT_PATH = Path(__file__).with_name("latency_throughput.png")
BASELINE_RESULTS_PATH = Path(__file__).with_name("baseline_results.json")

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


def plot_curve(sweep: list[dict], baseline: dict | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    qps = [row["request_throughput_qps"] for row in sweep]
    p95 = [row["p95_e2e_s"] for row in sweep]
    conc = [row["concurrency"] for row in sweep]

    ax = axes[0]
    ax.plot(qps, p95, marker="o", color="#2563eb", linewidth=2, label="vLLM")
    for x, y, c in zip(qps, p95, conc):
        ax.annotate(
            f"c={c}",
            (x, y),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=8,
        )
    if baseline is not None:
        ax.scatter(
            [baseline["request_throughput_qps"]],
            [baseline["p95_e2e_s"]],
            marker="*",
            s=200,
            color="#dc2626",
            zorder=5,
            label="baseline HF c=32",
        )
    # Log scales: baseline and vLLM are ~30x apart on both axes.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Throughput (req/s, log)")
    ax.set_ylabel("p95 end-to-end latency (s, log)")
    ax.set_title("Latency–throughput curve")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="center right")

    ax = axes[1]
    ax.plot(conc, qps, marker="o", color="#2563eb", linewidth=2, label="vLLM QPS")
    ax.plot(conc, p95, marker="s", color="#16a34a", linewidth=2, label="vLLM p95 (s)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("QPS / p95 latency (s)")
    ax.set_title("Sweep over concurrency")
    ax.set_xticks(conc)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


async def async_main(args: argparse.Namespace) -> None:
    print(f"Loading vLLM engine: {MODEL_NAME}")
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        enable_prefix_caching=False,
        disable_log_stats=True,
        enable_log_requests=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        min_tokens=MAX_NEW_TOKENS,
        ignore_eos=True,
    )

    formatted = [
        apply_chat_template(tokenizer, p)
        for p in build_prompts(max(32, max(CONCURRENCIES)))
    ]

    print("Running warmup...")
    await run_one(engine, formatted[0], SamplingParams(temperature=0.0, max_tokens=16))

    sweep: list[dict] = []
    for concurrency in CONCURRENCIES:
        n_requests = max(32, concurrency)
        prompts = formatted[:n_requests]
        print(f"\n=== vLLM concurrency={concurrency} requests={n_requests} ===")
        row = await run_concurrency(engine, prompts, concurrency, sampling_params)
        sweep.append(row)
        print(
            f"p95={row['p95_e2e_s']:.3f}s  "
            f"QPS={row['request_throughput_qps']:.3f}  "
            f"tok/s={row['token_throughput_tok_s']:.1f}"
        )

    baseline = None
    if BASELINE_RESULTS_PATH.exists():
        baseline = json.loads(BASELINE_RESULTS_PATH.read_text())

    plot_curve(sweep, baseline)

    headline = next(row for row in sweep if row["concurrency"] == 32)
    payload = {
        "model": MODEL_NAME,
        "prefix_caching": False,
        "p95_measures": (
            "end-to-end request latency "
            "(request issued to complete response)"
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "headline_concurrency_32": headline,
        "sweep": sweep,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))

    print("\n========== vLLM Results ==========")
    print(f"Model:              {MODEL_NAME}")
    print(f"Prefix caching:     False")
    print(f"P95 @ c=32:         {headline['p95_e2e_s']:.3f} s")
    print(f"QPS @ c=32:         {headline['request_throughput_qps']:.3f} req/s")
    print(f"tok/s @ c=32:       {headline['token_throughput_tok_s']:.2f}")
    print(f"Plot:               {PLOT_PATH}")
    print(f"Saved:              {RESULTS_PATH}")
    print("==================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the figure from saved JSON without loading the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot_only:
        payload = json.loads(RESULTS_PATH.read_text())
        baseline = (
            json.loads(BASELINE_RESULTS_PATH.read_text())
            if BASELINE_RESULTS_PATH.exists()
            else None
        )
        plot_curve(payload["sweep"], baseline)
        print(f"Wrote {PLOT_PATH}")
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
