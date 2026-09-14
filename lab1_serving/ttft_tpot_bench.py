"""TTFT / TPOT split on the existing vLLM serving stack.

Does not redo the end-to-end concurrency sweep in vllm_bench.py. Streams
each request from AsyncLLMEngine (the same incremental path as
OpenAI-compatible SSE / stream=True) and records:

  * TTFT  — request issued -> first output token
  * TPOT  — mean inter-token interval after the first token, per request

Same prompts, 128 forced tokens, prefix caching off, as vllm_bench.py.
Measured at concurrency=1 and concurrency=32 (32 requests each).
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
NUM_REQUESTS = 32
CONCURRENCIES = [1, 32]
RESULTS_PATH = Path(__file__).with_name("ttft_tpot_results.json")
PLOT_PATH = Path(__file__).with_name("ttft_tpot.png")

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


def pct(xs: np.ndarray, q: float) -> float:
    return float(np.percentile(xs, q))


async def run_one_stream(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
) -> dict:
    """Issue one request and timestamp every newly arrived output token."""
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    token_times: list[float] = []
    seen = 0
    n_yields = 0
    max_new_in_yield = 0

    async for output in engine.generate(prompt, sampling_params, request_id):
        n_yields += 1
        n = len(output.outputs[0].token_ids) if output.outputs else 0
        now = time.perf_counter()
        gained = n - seen
        max_new_in_yield = max(max_new_in_yield, gained)
        # Stamp each newly visible token at the moment this yield arrived.
        # vLLM decode normally yields one token per step; if a yield carries
        # k>1, the extra tokens share this timestamp (interval 0).
        for _ in range(gained):
            token_times.append(now)
        seen = n

    e2e = time.perf_counter() - t0
    n_tokens = len(token_times)
    if n_tokens == 0:
        raise RuntimeError(f"request {request_id} produced 0 tokens")

    ttft = token_times[0] - t0
    itls = np.diff(token_times) if n_tokens > 1 else np.asarray([], dtype=np.float64)
    tpot = float(np.mean(itls)) if itls.size else float("nan")

    return {
        "e2e_s": e2e,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "n_tokens": n_tokens,
        "n_yields": n_yields,
        "max_new_tokens_in_one_yield": max_new_in_yield,
        "itls_s": itls.tolist(),
    }


async def run_concurrency(
    engine: AsyncLLMEngine,
    prompts: list[str],
    concurrency: int,
    sampling_params: SamplingParams,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def bound(prompt: str) -> dict:
        async with semaphore:
            return await run_one_stream(engine, prompt, sampling_params)

    wall_start = time.perf_counter()
    rows = await asyncio.gather(*[bound(p) for p in prompts])
    wall_s = time.perf_counter() - wall_start

    ttfts = np.asarray([r["ttft_s"] for r in rows], dtype=np.float64)
    tpots = np.asarray([r["tpot_s"] for r in rows], dtype=np.float64)
    e2es = np.asarray([r["e2e_s"] for r in rows], dtype=np.float64)
    all_itls = np.concatenate(
        [np.asarray(r["itls_s"], dtype=np.float64) for r in rows]
    )
    tokens = int(sum(r["n_tokens"] for r in rows))

    return {
        "concurrency": concurrency,
        "num_requests": len(prompts),
        "wall_s": float(wall_s),
        "generated_tokens": tokens,
        "ttft_p50_ms": pct(ttfts, 50) * 1000.0,
        "ttft_p95_ms": pct(ttfts, 95) * 1000.0,
        "tpot_p50_ms": pct(tpots, 50) * 1000.0,
        "tpot_p95_ms": pct(tpots, 95) * 1000.0,
        "ttft_mean_ms": float(np.mean(ttfts) * 1000.0),
        "tpot_mean_ms": float(np.mean(tpots) * 1000.0),
        "e2e_p50_s": pct(e2es, 50),
        "e2e_p95_s": pct(e2es, 95),
        "itl_p50_ms": pct(all_itls, 50) * 1000.0 if all_itls.size else None,
        "itl_p95_ms": pct(all_itls, 95) * 1000.0 if all_itls.size else None,
        "max_new_tokens_in_one_yield": max(
            r["max_new_tokens_in_one_yield"] for r in rows
        ),
        "mean_yields_per_request": float(np.mean([r["n_yields"] for r in rows])),
        "per_request": [
            {
                "e2e_s": r["e2e_s"],
                "ttft_s": r["ttft_s"],
                "tpot_s": r["tpot_s"],
                "n_tokens": r["n_tokens"],
                "n_yields": r["n_yields"],
            }
            for r in rows
        ],
    }


def plot_table(rows: list[dict]) -> None:
    labels = ["TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95"]
    by_c = {r["concurrency"]: r for r in rows}
    c1 = by_c[1]
    c32 = by_c[32]
    v1 = [c1["ttft_p50_ms"], c1["ttft_p95_ms"], c1["tpot_p50_ms"], c1["tpot_p95_ms"]]
    v32 = [
        c32["ttft_p50_ms"],
        c32["ttft_p95_ms"],
        c32["tpot_p50_ms"],
        c32["tpot_p95_ms"],
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - width / 2, v1, width, label="c=1", color="#2563eb")
    ax.bar(x + width / 2, v32, width, label="c=32", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("TTFT vs TPOT at concurrency 1 and 32")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


async def async_main() -> None:
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
    formatted = [apply_chat_template(tokenizer, p) for p in build_prompts(NUM_REQUESTS)]

    print("Running warmup...")
    await run_one_stream(
        engine, formatted[0], SamplingParams(temperature=0.0, max_tokens=16)
    )

    sweep: list[dict] = []
    for concurrency in CONCURRENCIES:
        print(f"\n=== TTFT/TPOT concurrency={concurrency} requests={NUM_REQUESTS} ===")
        row = await run_concurrency(engine, formatted, concurrency, sampling_params)
        sweep.append(row)
        print(
            f"TTFT p50/p95={row['ttft_p50_ms']:.1f}/{row['ttft_p95_ms']:.1f} ms  "
            f"TPOT p50/p95={row['tpot_p50_ms']:.2f}/{row['tpot_p95_ms']:.2f} ms  "
            f"e2e p95={row['e2e_p95_s']:.3f}s  "
            f"yields/req={row['mean_yields_per_request']:.1f}"
        )

    plot_table(sweep)
    payload = {
        "model": MODEL_NAME,
        "prefix_caching": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_requests": NUM_REQUESTS,
        "definitions": {
            "ttft": "request issued to engine -> first streamed output token",
            "tpot": (
                "per-request mean inter-token interval after the first token; "
                "p50/p95 are across those 32 per-request means"
            ),
            "clock_start": (
                "after the concurrency semaphore is acquired, i.e. when this "
                "request is actually submitted — client-side queueing at c=1 "
                "is not counted as TTFT"
            ),
        },
        "sweep": sweep,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {RESULTS_PATH}")
    print(f"Plot  {PLOT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the figure from saved JSON without loading the model",
    )
    args = parser.parse_args()
    if args.plot_only:
        payload = json.loads(RESULTS_PATH.read_text())
        plot_table(payload["sweep"])
        print(f"Wrote {PLOT_PATH}")
        return
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
