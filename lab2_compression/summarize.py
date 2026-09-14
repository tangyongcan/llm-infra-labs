"""Merge the per-mode result JSONs into one table plus a summary figure.

Each measurement lives in its own file because each ran in its own process:
  results_<mode>.json                     eager speed + memory
  results_<mode>_ppl.json                 perplexity only
  results_<mode>_compiled_prefill.json    compiled compute-bound throughput
  results_<mode>_compiled_decode1.json    compiled batch-1 decode
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
PLOT_PATH = DIR / "compression_summary.png"
PRUNE_PLOT_PATH = DIR / "pruning_summary.png"
TABLE_PATH = DIR / "summary.json"

MODES = ["bf16", "int8", "fp8-weight", "fp8-dynamic"]
LABELS = {
    "bf16": "BF16\n(baseline)",
    "int8": "INT8\n(bitsandbytes)",
    "fp8-weight": "FP8 weight-only\n(torchao)",
    "fp8-dynamic": "FP8 act+weight\n(torchao)",
}
COLORS = {
    "bf16": "#64748b",
    "int8": "#f59e0b",
    "fp8-weight": "#3b82f6",
    "fp8-dynamic": "#16a34a",
}


def load(name: str) -> dict | None:
    path = DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect() -> dict[str, dict]:
    table: dict[str, dict] = {}
    for mode in MODES:
        eager = load(f"results_{mode}.json") or {}
        ppl = load(f"results_{mode}_ppl.json") or {}
        cprefill = load(f"results_{mode}_compiled_prefill.json") or {}
        cdecode = load(f"results_{mode}_compiled_decode1.json") or {}

        if not eager and not ppl:
            continue

        table[mode] = {
            "backend": eager.get("backend") or ppl.get("backend"),
            "weight_mem_gib": eager.get("weight_mem_gib")
            or ppl.get("weight_mem_gib"),
            "decode1_peak_mem_gib": eager.get("decode1_peak_mem_gib"),
            "decode1_tok_s_eager": eager.get("decode1_tok_s"),
            "decode8_tok_s_eager": eager.get("decode8_tok_s"),
            "prefill_tok_s_eager": eager.get("prefill_tok_s"),
            "decode1_tok_s_compiled": cdecode.get("decode1_tok_s"),
            "prefill_tok_s_compiled": cprefill.get("prefill_tok_s"),
            "perplexity": ppl.get("perplexity"),
            "mean_nll": ppl.get("mean_nll"),
            "ppl_eval_tokens": ppl.get("ppl_eval_tokens"),
            "quantize_s": eager.get("quantize_s"),
        }
    return table


def ratio(value, base):
    if value is None or base in (None, 0):
        return None
    return value / base


def add_ratios(table: dict[str, dict]) -> dict[str, dict]:
    base = table.get("bf16")
    if not base:
        return table
    for mode, row in table.items():
        row["compression_vs_bf16"] = ratio(base["weight_mem_gib"], row["weight_mem_gib"])
        row["speedup_decode1_eager"] = ratio(
            row["decode1_tok_s_eager"], base["decode1_tok_s_eager"]
        )
        row["speedup_decode8_eager"] = ratio(
            row["decode8_tok_s_eager"], base["decode8_tok_s_eager"]
        )
        row["speedup_prefill_eager"] = ratio(
            row["prefill_tok_s_eager"], base["prefill_tok_s_eager"]
        )
        row["speedup_prefill_compiled"] = ratio(
            row["prefill_tok_s_compiled"], base["prefill_tok_s_compiled"]
        )
        row["speedup_decode1_compiled"] = ratio(
            row["decode1_tok_s_compiled"], base["decode1_tok_s_compiled"]
        )
        if row.get("perplexity") and base.get("perplexity"):
            row["ppl_delta_pct"] = (
                (row["perplexity"] - base["perplexity"]) / base["perplexity"] * 100
            )
    return table


def bar_panel(ax, modes, values, title, ylabel, fmt="{:.2f}", baseline=None):
    xs = np.arange(len(modes))
    vals = [0 if v is None else v for v in values]
    ax.bar(
        xs,
        vals,
        color=[COLORS[m] for m in modes],
        width=0.62,
        edgecolor="white",
    )
    for x, v, raw in zip(xs, vals, values):
        if raw is None:
            ax.text(x, 0, "n/a", ha="center", va="bottom", fontsize=8, color="#888")
        else:
            ax.text(
                x,
                v,
                fmt.format(raw),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    if baseline is not None:
        ax.axhline(baseline, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m] for m in modes], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(vals) * 1.22 if max(vals) else 1)


def grouped_panel(ax, modes, eager, compiled, title, ylabel, fmt):
    xs = np.arange(len(modes))
    width = 0.36
    for offset, series, name, alpha in [
        (-width / 2, eager, "eager", 0.45),
        (width / 2, compiled, "torch.compile", 1.0),
    ]:
        vals = [0 if v is None else v for v in series]
        ax.bar(
            xs + offset,
            vals,
            width=width,
            color=[COLORS[m] for m in modes],
            alpha=alpha,
            edgecolor="white",
            label=name,
        )
        for x, v, raw in zip(xs + offset, vals, series):
            if raw is None:
                ax.text(x, 0, "n/a", ha="center", va="bottom", fontsize=7, color="#888")
            else:
                ax.text(x, v, fmt.format(raw), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m] for m in modes], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    allv = [v for v in list(eager) + list(compiled) if v]
    if allv:
        ax.set_ylim(0, max(allv) * 1.25)


def plot(table: dict[str, dict]) -> None:
    modes = [m for m in MODES if m in table]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

    bar_panel(
        axes[0][0],
        modes,
        [table[m]["weight_mem_gib"] for m in modes],
        "Weight memory after load (lower = better)",
        "GiB",
        "{:.2f}",
    )
    grouped_panel(
        axes[0][1],
        modes,
        [table[m]["decode1_tok_s_eager"] for m in modes],
        [table[m]["decode1_tok_s_compiled"] for m in modes],
        "Decode, batch 1 (memory-bandwidth-bound)",
        "tokens/s",
        "{:.1f}",
    )
    grouped_panel(
        axes[1][0],
        modes,
        [table[m]["prefill_tok_s_eager"] for m in modes],
        [table[m]["prefill_tok_s_compiled"] for m in modes],
        "Prefill, 8x512 (compute-bound)\nnative FP8 tensor cores show up here",
        "tokens/s",
        "{:.0f}",
    )
    ppls = [table[m]["perplexity"] for m in modes]
    bar_panel(
        axes[1][1],
        modes,
        ppls,
        "WikiText-2 perplexity (lower = better)",
        "perplexity",
        "{:.3f}",
        baseline=table.get("bf16", {}).get("perplexity"),
    )
    # Perplexity differences are sub-percent; zoom so they stay visible.
    valid = [p for p in ppls if p]
    if valid:
        axes[1][1].set_ylim(min(valid) * 0.985, max(valid) * 1.015)

    fig.suptitle(
        "Lab 2: Qwen2.5-3B-Instruct compression on RTX 4060 Ti (Ada, native FP8)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


def plot_pruning() -> None:
    data = load("results_prune.json")
    if not data:
        return
    rows = data["rows"]
    levels = [r["target_sparsity"] * 100 for r in rows]
    ppl = [r.get("perplexity") for r in rows]
    mem = [r["mem_alloc_gib"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.plot(levels, ppl, marker="o", color="#dc2626", linewidth=2)
    ax.set_yscale("log")
    for x, y in zip(levels, ppl):
        ax.annotate(
            f"{y:,.0f}" if y > 100 else f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xlabel("MLP structured sparsity (%)")
    ax.set_ylabel("WikiText-2 perplexity (log)")
    ax.set_title("One-shot structured pruning destroys quality\n(no fine-tuning)")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    ax.bar(
        [f"{l:.0f}%" for l in levels],
        mem,
        color=["#64748b"] + ["#f59e0b"] * (len(levels) - 1),
        width=0.6,
        edgecolor="white",
    )
    for i, v in enumerate(mem):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xlabel("MLP structured sparsity")
    ax.set_ylabel("allocated GiB")
    ax.set_title("Masking does not save memory\n(weight_orig + mask are extra copies)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(mem) * 1.2)

    fig.tight_layout()
    fig.savefig(PRUNE_PLOT_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    table = add_ratios(collect())
    TABLE_PATH.write_text(json.dumps(table, indent=2))
    plot(table)
    plot_pruning()

    def cell(v, fmt="{:.3f}"):
        return "n/a" if v is None else fmt.format(v)

    print(f"{'mode':14s} {'memGiB':>8s} {'comp':>6s} {'dec1':>8s} {'dec8':>8s} "
          f"{'pre-eag':>9s} {'pre-cmp':>9s} {'dec1-cmp':>9s} {'ppl':>9s} {'dppl%':>7s}")
    for mode, row in table.items():
        print(
            f"{mode:14s} "
            f"{cell(row['weight_mem_gib'], '{:.3f}'):>8s} "
            f"{cell(row.get('compression_vs_bf16'), '{:.2f}x'):>6s} "
            f"{cell(row['decode1_tok_s_eager'], '{:.2f}'):>8s} "
            f"{cell(row['decode8_tok_s_eager'], '{:.1f}'):>8s} "
            f"{cell(row['prefill_tok_s_eager'], '{:.0f}'):>9s} "
            f"{cell(row['prefill_tok_s_compiled'], '{:.0f}'):>9s} "
            f"{cell(row['decode1_tok_s_compiled'], '{:.2f}'):>9s} "
            f"{cell(row['perplexity'], '{:.4f}'):>9s} "
            f"{cell(row.get('ppl_delta_pct'), '{:+.2f}'):>7s}"
        )
    print(f"\nwrote {TABLE_PATH}\nwrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
