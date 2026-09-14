# Measured LLM Inference Infrastructure on a Single Ada GPU

This repository records a sequence of measurements of LLM inference
infrastructure on one **RTX 4060 Ti 16GB** (Ada Lovelace, sm_89) with
**Qwen2.5-3B-Instruct**. The work moves from a naive serving baseline, through
bottleneck characterization and compression, to FP8 on the same vLLM protocol,
then to a queue-depth autoscaling simulation and a lightweight model registry.

It is a measured study with scripts, raw JSON/CSV, plots, and engineering
notes. It is not an installable serving library and not a production cluster.

## What This Repository Is

Five experiments isolate one layer of the inference stack at a time:

- how requests are batched
- how weights are compressed
- whether batching and FP8 still compose on the **same** serving protocol
- how replicas are scaled when decode is memory-bandwidth-bound
- how a model version is named, compared, and rolled back

Headline numbers come from this machine. Software stacks differ across labs
and are called out below. Independent speedups must not be multiplied.

## Hardware and Scope

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Ti 16GB × 1, WSL2 |
| Architecture | Ada Lovelace, compute capability **8.9** (native FP8 Tensor Cores) |
| Model | `Qwen/Qwen2.5-3B-Instruct` (3.09B) |
| Lab 1 / Lab 5 | vLLM 0.10.2, torch 2.8.0+cu128 (`vllmlab`); prefix caching **off** |
| Lab 1 baseline / Lab 2 / Lab 4 | torch 2.13.0+cu130, transformers 5.15.0 (`llmlab`) |
| Lab 2 extras | bitsandbytes 0.50.0, torchao 0.18.0 |
| Lab 3 | minikube + KEDA; **simulated** workers, no GPU model |
| Lab 4 | peft 0.20.0, MLflow 3.15.1; toy LoRA, local sqlite |

Cloud T4/A10 GPUs can reproduce memory compression, not the FP8 *speed*
numbers: those cards lack native FP8 Tensor Cores.

Lab 2 FP8 was torchao in-process (never checkpointed). Lab 5 uses vLLM
`--quantization fp8` (E4M3 W8A8, per-tensor). Same numeric format, not a
byte-identical reload. Lab 2 WikiText-2 PPL does not transfer to Lab 5.
`kv_cache_dtype=fp8` was not enabled in Lab 5.

## Results at a Glance

All figures below are taken from committed artifacts. Rounding matches the
JSON; see each lab README for definitions.

### Serving (Lab 1)

Same 32 requests, 128 forced tokens, concurrency 32, BF16, prefix cache off:

| Stack | p95 e2e | QPS | Source |
| --- | ---: | ---: | --- |
| Naive HF `generate()`, FIFO, batch=1 | **154.22 s** | **0.197** | `lab1_serving/baseline_results.json` |
| vLLM continuous batching + paged KV | **5.08 s** | **6.302** | `lab1_serving/vllm_results.json` |

Throughput ratio at this point: **31.9×** (0.197 → 6.302 QPS). Single-request
generate time is ~5 s on both stacks; most of the e2e gap is queueing, not a
faster model.

Latency split on the same vLLM engine (`lab1_serving/ttft_tpot_results.json`):

| concurrency | TTFT p50 | TPOT p50 |
| ---: | ---: | ---: |
| 1 | **36.1 ms** | **33.16 ms** |
| 32 | **263.5 ms** | **35.41 ms** |

TTFT rises **7.3×**. TPOT rises **+6.8%**. Almost all of the ~5 s e2e at
c=32 is steady decode.

Device-side `nvidia-smi dmon` (`lab1_serving/bound_probe_results.json`):

| Phase | busy SM% | busy mem% |
| --- | ---: | ---: |
| decode (c=1) | 95 | **93** |
| prefill (c=32, `max_tokens=1`) | 99 | **49** |

SM% is near full in both phases. Memory-controller busy% is the contrast.

### Compression (Lab 2) — bare HuggingFace, not vLLM

From `lab2_compression/summary.json` (weight memory is
`torch.cuda.memory_allocated()` after load/quantize):

| Config | Weight memory | vs BF16 | decode compile (tok/s) | prefill compile (tok/s) | WikiText-2 PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | **5.748 GiB** | 1.00× | 26.90 | 4534 | **8.4683** |
| INT8 (bitsandbytes) | **3.170 GiB** | **1.81×** | n/a (eager 5.61) | n/a (eager 2984) | 8.5803 (**+1.32%**) |
| FP8 weight-only | 3.171 GiB | 1.81× | **38.39** | 4323 | 8.5054 (+0.44%) |
| FP8 act+weight | 3.170 GiB | 1.81× | 34.37 | **6882** | 8.5353 (+0.79%) |

INT8 and FP8 save similar weight memory. Only native FP8 plus `torch.compile`
is faster (prefill **1.52×** vs compiled BF16). INT8 decode is **4.4× slower**
than eager BF16. Structured pruning without recovery training collapsed
perplexity (10% MLP sparsity → PPL 10,551) and *increased* allocated memory
because `torch.nn.utils.prune` keeps a mask (`lab2_compression/results_prune.json`).

### Integration (Lab 5) — FP8 on Lab 1's vLLM protocol

`lab5_integration/comparison.json`, c=32, same prompts and 128 forced tokens:

| Stack | p95 e2e | QPS | vs naive | vs BF16+vLLM |
| --- | ---: | ---: | ---: | ---: |
| BF16 + naive HF | 154.22 s | 0.197 | 1.00× | — |
| BF16 + vLLM | 5.08 s | 6.302 | 31.9× | 1.00× |
| FP8 + vLLM | **3.11 s** | **10.289** | **52.1×** | **1.63×** |

**52.1× is a measured three-stack comparison**, not 31.9× multiplied by a
Lab 2 FP8 factor. Lab 2 never entered vLLM. From c=1 to c=64 the FP8/BF16
QPS ratio stays **1.63–1.66×** (`lab5_integration/fp8_vllm_results.json`).

### Autoscaling (Lab 3) — simulated workers

The cluster does **not** serve a GPU model. Workers fake LLM traits: 20 s
startup, concurrency 2, CPU busy 5% of request time. KEDA scales on
**queue depth**, 1→8.

Canonical run (`lab3_autoscaling/logs/timeline.csv`):

- burst at t=16.1 s, queue_depth=198
- HPA desired 1→5 at t=27.0 s; 8 pods created at t=42.1 s; 8 Ready at t=60.9 s
- 200 requests done at t=84.3 s
- per-pod CPU peak **65.6 m** at t=127.4 s (43 s after the queue drained; never
  crossed an 80 m / 80%-of-100 m CPU HPA line)
- scaled back to 1 replica at t=292.0 s

A repeat run peaked at **63.1 m** CPU (`logs_run1/timeline.csv`). That run's
cold-start log is incomplete (only the initial replica).

### Registry (Lab 4) — pointer rollback, not serving

One-epoch LoRA on 32 local JSONL rows, two versions of
`qwen25-3b-lora-sft`. v1: lr 2e-4, `lora_r=8`, `final_eval_loss` 3.5126.
v2: lr 1e-4, `lora_r=16`, `final_eval_loss` 3.8399 (worse). Production was
moved from v2 back to v1 via MLflow stage/alias. That call flips a pointer;
it does not reload weights into a serving replica. A later local re-run of
`rollback --to-version 1` (v1 was already Production) wrote
`lab4_registry/rollback_result.txt` with `rollback_seconds=0.0368`. That is
a metadata-API timing, not the original v2→v1 flip and not a serving reload.

## How the Experiments Connect

```text
Lab 1  Serving baseline
         naive HF generate vs vLLM; TTFT/TPOT; dmon bandwidth
         ↓
Lab 2  Compression
         INT8 / FP8 / prune on bare HuggingFace + torchao
         ↓
Lab 5  FP8 + vLLM integration
         same serving protocol as Lab 1; added after Labs 1–4
         ↓
Lab 3  Queue-depth autoscaling
         simulated workers (no GPU model in the cluster)
         ↓
Lab 4  Registry and rollback
         toy LoRA + MLflow metadata; not wired to serving
```

Lab 5 lives in `lab5_integration/` because it was added after Labs 1–4. The
directory number is historical. Read it after Lab 2: that is where batching
and compression first sit on one protocol.

Lab 1 shows that on this card, most of the serving gain is continuous
batching, not a faster forward pass. Lab 2 shows that 8-bit weights save
memory, but speed requires native FP8 and compiler fusion; INT8 on this
consumer GPU is a slow dequant path. Lab 5 re-measures FP8 where Lab 1
measured BF16. Lab 3 asks which signal should add replicas when the host CPU
stays idle during a memory-bound decode-like wait. Lab 4 asks whether two
on-disk adapters can be told apart without logged commit, data hash, and
metrics.

Labs 1, 2, 4, and 5 run the real 3B model on the 4060 Ti. Lab 3 does not.

## Repository Structure

```text
lab1_serving/         serving baseline, TTFT/TPOT, dmon probe
lab2_compression/     BF16 / INT8 / FP8 / prune on HuggingFace
lab5_integration/     FP8 on the Lab 1 vLLM protocol
lab3_autoscaling/     minikube + KEDA queue-depth demo (simulated workers)
lab4_registry/        MLflow registry, lineage screenshots, toy SFT data
```

Each directory has its own README, scripts, and measurement artifacts.

## Reproduction

There is no single command that reruns the whole study. GPU labs need the
matching conda env and this GPU. Do not run two vLLM engines at once.

```bash
# Lab 1 baseline (llmlab)
conda activate llmlab
python lab1_serving/baseline_bench.py

# Lab 1 / Lab 5 vLLM path (vllmlab); one engine at a time
conda activate vllmlab
python lab1_serving/vllm_bench.py
python lab1_serving/ttft_tpot_bench.py
python lab1_serving/bound_probe.py
python lab5_integration/fp8_vllm_bench.py

# Lab 2 (llmlab). run_all.sh covers part of the matrix; see that lab README.
conda activate llmlab
bash lab2_compression/run_all.sh
python lab2_compression/prune_demo.py
python lab2_compression/summarize.py
```

Lab 3 needs Docker, minikube, kubectl, and helm; see
`lab3_autoscaling/README.md`. Lab 4 needs MLflow; adapters and `tracking.db`
are gitignored and are recreated by `lab4_registry/run_demo.sh`.

Plot-only flags exist on the vLLM scripts if you only want to redraw figures
from saved JSON.

## Limitations

- One consumer GPU, one 3B instruct model, WSL2.
- Lab 1 and Lab 2 use different PyTorch / serving stacks; their speedups are
  not composable.
- Lab 3 workers are synthetic. The result is about the scaling *signal* and
  a configured 20 s ready delay, not about GPU autoscaling in production.
- Lab 4 demonstrates registry semantics on a toy adapter. Serving never loads
  `models:/qwen25-3b-lora-sft/Production`.
- No multi-GPU, no distributed serving, no distillation, no `kv_cache_dtype=fp8`.
- FP8 throughput will not match on GPUs without native FP8 Tensor Cores.

## License

MIT. Measurements and writeups in this repository are provided under the
license in `LICENSE`. Model weights are from the Qwen authors under their
license and are not redistributed here.
