"""SM vs memory-controller utilization probe on the Lab 1 vLLM engine.

Runs two isolated phases while `nvidia-smi dmon -s u` samples at 1 Hz:

  * decode  — concurrency=1, short prompt, 256 new tokens (prefill is ~36 ms,
              so almost every dmon row is pure decode)
  * prefill — concurrency=32, ~1024-token prompts, max_tokens=1 (almost no
              decode; this is the compute-bound contrast)

dmon `sm` is kernel-active % in the sample window, not arithmetic intensity.
`mem` is memory-controller busy %, a proxy for HBM traffic (not GB/s).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
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
MAX_MODEL_LEN = 2048
HERE = Path(__file__).parent
RESULTS_PATH = HERE / "bound_probe_results.json"
DECODE_LOG = HERE / "dmon_decode.log"
PREFILL_LOG = HERE / "dmon_prefill.log"
DECODE_PNG = HERE / "dmon_decode.png"
PREFILL_PNG = HERE / "dmon_prefill.png"
COMPARE_PNG = HERE / "bound_probe_sm_mem.png"

SHORT_PROMPT = "Explain what a KV cache is in large language model inference."
FILLER = (
    "A KV cache stores key and value tensors from previous tokens so decode "
    "does not recompute attention over the full sequence. "
)


def apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def long_prompt(tokenizer, target_tokens: int = 1024) -> str:
    text = FILLER
    while True:
        formatted = apply_chat_template(tokenizer, text)
        n = len(tokenizer.encode(formatted))
        if n >= target_tokens:
            return formatted
        text += FILLER


def parse_dmon(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            rows.append(
                {
                    "gpu": int(parts[0]),
                    "sm": int(parts[1]),
                    "mem": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    sm = np.asarray([r["sm"] for r in rows], dtype=np.float64)
    mem = np.asarray([r["mem"] for r in rows], dtype=np.float64)
    busy = [r for r in rows if r["sm"] >= 50]
    bsm = np.asarray([r["sm"] for r in busy], dtype=np.float64) if busy else sm
    bmem = np.asarray([r["mem"] for r in busy], dtype=np.float64) if busy else mem
    return {
        "n": int(len(rows)),
        "sm_mean": float(np.mean(sm)),
        "sm_p50": float(np.median(sm)),
        "sm_max": float(np.max(sm)),
        "mem_mean": float(np.mean(mem)),
        "mem_p50": float(np.median(mem)),
        "mem_max": float(np.max(mem)),
        "n_busy": int(len(busy)),
        "sm_mean_busy": float(np.mean(bsm)),
        "mem_mean_busy": float(np.mean(bmem)),
    }


def render_dmon_screenshot(path: Path, png: Path, title: str, stats: dict) -> None:
    raw = path.read_text().splitlines()
    # Keep header + up to ~18 sample rows so it still looks like a dmon scroll.
    header = [ln for ln in raw if ln.startswith("#")][:2]
    samples = [ln for ln in raw if ln.strip() and not ln.startswith("#")]
    body = header + samples[:18]
    if len(samples) > 18:
        body.append(f"    … {len(samples) - 18} more samples …")
    body.append("")
    body.append(
        f"all:  mean SM={stats['sm_mean']:.0f}%  mean MEM={stats['mem_mean']:.0f}%  "
        f"n={stats['n']} @ 1 Hz"
    )
    if stats.get("n_busy"):
        body.append(
            f"busy (SM≥50%):  mean SM={stats['sm_mean_busy']:.0f}%  "
            f"mean MEM={stats['mem_mean_busy']:.0f}%  n={stats['n_busy']}"
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.text(
        0.02,
        0.97,
        title,
        transform=ax.transAxes,
        fontsize=11,
        color="#e5e5e5",
        fontfamily="monospace",
        va="top",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.90,
        "\n".join(body) if body else "(empty dmon log)",
        transform=ax.transAxes,
        fontsize=10,
        color="#4ade80",
        fontfamily="monospace",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_compare(decode: dict, prefill: dict) -> None:
    labels = ["SM util %", "mem controller util %"]
    d = [decode["sm_mean_busy"], decode["mem_mean_busy"]]
    p = [prefill["sm_mean_busy"], prefill["mem_mean_busy"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - width / 2, d, width, label="decode  c=1", color="#2563eb")
    ax.bar(x + width / 2, p, width, label="prefill c=32", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("nvidia-smi dmon utilization (%)")
    ax.set_ylim(0, 100)
    ax.set_title("SM vs memory-controller util (busy samples, SM≥50%)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(COMPARE_PNG, dpi=160)
    plt.close(fig)


class Dmon:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.fh = None

    def start(self) -> None:
        self.fh = self.log_path.open("w")
        self.proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
            stdout=self.fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(1.2)

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        if self.fh:
            self.fh.flush()
            self.fh.close()


async def run_one(engine, prompt: str, sampling: SamplingParams) -> None:
    request_id = str(uuid.uuid4())
    async for _ in engine.generate(prompt, sampling, request_id):
        pass


async def run_decode(engine, prompt: str, n_requests: int = 6) -> float:
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        min_tokens=256,
        ignore_eos=True,
    )
    t0 = time.perf_counter()
    for _ in range(n_requests):
        await run_one(engine, prompt, sampling)
    return time.perf_counter() - t0


async def run_prefill(
    engine, prompt: str, concurrency: int = 32, rounds: int = 8
) -> float:
    sampling = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1)
    t0 = time.perf_counter()
    for _ in range(rounds):
        await asyncio.gather(
            *[run_one(engine, prompt, sampling) for _ in range(concurrency)]
        )
    return time.perf_counter() - t0


async def async_main() -> None:
    print(f"Loading vLLM engine: {MODEL_NAME}")
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL_NAME,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.85,
            max_num_seqs=128,
            enable_prefix_caching=False,
            disable_log_stats=True,
            enable_log_requests=False,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    short = apply_chat_template(tokenizer, SHORT_PROMPT)
    long = long_prompt(tokenizer, 1024)
    print(f"long prompt tokens ≈ {len(tokenizer.encode(long))}")

    print("Warmup...")
    await run_one(engine, short, SamplingParams(temperature=0.0, max_tokens=16))
    await run_one(engine, long, SamplingParams(temperature=0.0, max_tokens=1))

    print("\n=== decode  c=1, 256 tokens × 6 ===")
    dmon = Dmon(DECODE_LOG)
    dmon.start()
    decode_s = await run_decode(engine, short, n_requests=6)
    dmon.stop()
    print(f"decode wall {decode_s:.1f}s  log={DECODE_LOG}")

    await asyncio.sleep(2)

    print("\n=== prefill  c=32, ~1024-token prompt, max_tokens=1 × 8 rounds ===")
    dmon = Dmon(PREFILL_LOG)
    dmon.start()
    prefill_s = await run_prefill(engine, long, concurrency=32, rounds=8)
    dmon.stop()
    print(f"prefill wall {prefill_s:.1f}s  log={PREFILL_LOG}")

    decode_rows = parse_dmon(DECODE_LOG)
    prefill_rows = parse_dmon(PREFILL_LOG)
    decode_stats = summarize(decode_rows)
    prefill_stats = summarize(prefill_rows)

    render_dmon_screenshot(
        DECODE_LOG,
        DECODE_PNG,
        "nvidia-smi dmon -s u     [decode, concurrency=1]",
        decode_stats,
    )
    render_dmon_screenshot(
        PREFILL_LOG,
        PREFILL_PNG,
        "nvidia-smi dmon -s u     [prefill, concurrency=32, max_tokens=1]",
        prefill_stats,
    )
    plot_compare(decode_stats, prefill_stats)

    payload = {
        "model": MODEL_NAME,
        "sampler": "nvidia-smi dmon -s u -d 1",
        "note": (
            "dmon `sm` is SM activity %; `mem` is memory-controller busy %, "
            "a proxy for HBM bandwidth (not GB/s). GeForce does not expose "
            "DRAM throughput counters here."
        ),
        "decode": {
            "concurrency": 1,
            "new_tokens": 256,
            "requests": 6,
            "wall_s": decode_s,
            **decode_stats,
        },
        "prefill": {
            "concurrency": 32,
            "prompt_tokens_approx": 1024,
            "max_tokens": 1,
            "rounds": 8,
            "wall_s": prefill_s,
            **prefill_stats,
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {DECODE_PNG}, {PREFILL_PNG}, {COMPARE_PNG}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        decode_stats = summarize(parse_dmon(DECODE_LOG))
        prefill_stats = summarize(parse_dmon(PREFILL_LOG))
        payload = json.loads(RESULTS_PATH.read_text())
        payload["decode"].update(decode_stats)
        payload["prefill"].update(prefill_stats)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2))
        render_dmon_screenshot(
            DECODE_LOG, DECODE_PNG,
            "nvidia-smi dmon -s u     [decode, concurrency=1]",
            decode_stats,
        )
        render_dmon_screenshot(
            PREFILL_LOG, PREFILL_PNG,
            "nvidia-smi dmon -s u     [prefill, concurrency=32, max_tokens=1]",
            prefill_stats,
        )
        plot_compare(decode_stats, prefill_stats)
        print(f"rewrote {DECODE_PNG}, {PREFILL_PNG}, {COMPARE_PNG}")
        print(json.dumps({"decode": decode_stats, "prefill": prefill_stats}, indent=2))
        return
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
