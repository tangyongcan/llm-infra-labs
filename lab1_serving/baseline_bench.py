"""Naive HuggingFace generate baseline (no batching).

Models 32 concurrent arrivals into a single-worker FIFO queue:
each request is issued at t=0, then processed one-by-one with
model.generate(). End-to-end latency includes queue wait.

This is intentionally the "no continuous batching" serving baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_REQUESTS = 32
MAX_NEW_TOKENS = 128
RESULTS_PATH = Path(__file__).with_name("baseline_results.json")

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


def main() -> None:
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    raw_prompts = build_prompts(NUM_REQUESTS)
    prompts = [apply_chat_template(tokenizer, p) for p in raw_prompts]

    print("Running warmup...")
    warmup_inputs = tokenizer(prompts[0], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=16,
            min_new_tokens=16,
            do_sample=False,
        )
    torch.cuda.synchronize()

    print(
        f"Running {NUM_REQUESTS} concurrent arrivals on a "
        "single-worker naive generate() queue..."
    )

    generate_latencies: list[float] = []
    e2e_latencies: list[float] = []
    generated_tokens = 0

    # All 32 requests are issued at the same instant, then drained
    # sequentially. e2e latency = queue wait + generate.
    arrival_time = time.perf_counter()

    for i, prompt in enumerate(prompts):
        generate_start = time.perf_counter()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=[],
            )

        torch.cuda.synchronize()
        finish_time = time.perf_counter()

        generate_latency = finish_time - generate_start
        e2e_latency = finish_time - arrival_time
        generate_latencies.append(generate_latency)
        e2e_latencies.append(e2e_latency)
        generated_tokens += outputs.shape[1] - input_length

        print(
            f"Request {i + 1:02d}/{NUM_REQUESTS} "
            f"| generate = {generate_latency:.3f} s "
            f"| e2e = {e2e_latency:.3f} s"
        )

    total_time = time.perf_counter() - arrival_time
    e2e = np.asarray(e2e_latencies, dtype=np.float64)
    gen = np.asarray(generate_latencies, dtype=np.float64)

    results = {
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "num_gpus": torch.cuda.device_count(),
        "num_requests": NUM_REQUESTS,
        "concurrency": NUM_REQUESTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prefix_caching": False,
        "p95_measures": (
            "end-to-end request latency "
            "(issued at t=0 to complete response)"
        ),
        "total_time_s": float(total_time),
        "generated_tokens": int(generated_tokens),
        "mean_e2e_s": float(np.mean(e2e)),
        "p50_e2e_s": float(np.percentile(e2e, 50)),
        "p95_e2e_s": float(np.percentile(e2e, 95)),
        "mean_generate_s": float(np.mean(gen)),
        "p95_generate_s": float(np.percentile(gen, 95)),
        "request_throughput_qps": float(NUM_REQUESTS / total_time),
        "token_throughput_tok_s": float(generated_tokens / total_time),
        "e2e_latencies_s": e2e.tolist(),
        "generate_latencies_s": gen.tolist(),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n========== Baseline Results ==========")
    print(f"Model:                {MODEL_NAME}")
    print(f"GPU:                  {results['gpu']} x{results['num_gpus']}")
    print(f"Requests:             {NUM_REQUESTS}")
    print(f"Max new tokens:       {MAX_NEW_TOKENS}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Mean e2e latency:     {results['mean_e2e_s']:.3f} s")
    print(f"P50 e2e latency:      {results['p50_e2e_s']:.3f} s")
    print(f"P95 e2e latency:      {results['p95_e2e_s']:.3f} s")
    print(f"Mean generate only:   {results['mean_generate_s']:.3f} s")
    print(f"P95 generate only:    {results['p95_generate_s']:.3f} s")
    print(f"Request throughput:   {results['request_throughput_qps']:.3f} req/s")
    print(f"Token throughput:     {results['token_throughput_tok_s']:.2f} tokens/s")
    print(f"Generated tokens:     {generated_tokens}")
    print(f"Saved:                {RESULTS_PATH}")
    print("=======================================")


if __name__ == "__main__":
    main()
