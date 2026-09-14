"""LoRA SFT on Qwen2.5-3B-Instruct + MLflow Model Registry.

One-epoch toy fine-tune so there is a real adapter to version, not a dummy
pickle. The experiment is lineage and pointer rollback, not quality: v1 and
v2 differ by hyperparameters on the same data file.

    python train_and_register.py train --run-name v1 --lr 2e-4 --lora-r 8
    python train_and_register.py train --run-name v2 --lr 1e-4 --lora-r 16 --promote
    python train_and_register.py rollback --to-version 1
    python train_and_register.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_DATA = HERE / "data" / "toy_sft_v1.jsonl"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPERIMENT_NAME = "qwen25-lora-sft"
REGISTERED_MODEL = "qwen25-3b-lora-sft"

# File-store tracking is enough for a local registry (MLflow 2.9+ / 3.x).
DEFAULT_TRACKING_URI = f"sqlite:///{HERE / 'tracking.db'}"
DEFAULT_ARTIFACT_ROOT = HERE / "mlartifacts"


def configure_mlflow() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        artifact = (
            DEFAULT_ARTIFACT_ROOT.as_uri() if uri.startswith("sqlite:") else None
        )
        if artifact:
            client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact)
        else:
            client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def data_version(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.name}@{digest}"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SFTDataset:
    def __init__(self, rows: list[dict], tokenizer, max_len: int) -> None:
        self.examples = [self._encode(r, tokenizer, max_len) for r in rows]

    @staticmethod
    def _ids(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _encode(row: dict, tokenizer, max_len: int) -> dict:
        import torch
        prompt_ids = SFTDataset._ids(
            tokenizer,
            [{"role": "user", "content": row["instruction"]}],
            add_generation_prompt=True,
        )
        full_ids = SFTDataset._ids(
            tokenizer,
            [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ],
            add_generation_prompt=False,
        )
        if tokenizer.eos_token_id is not None and (
            not full_ids or full_ids[-1] != tokenizer.eos_token_id
        ):
            full_ids = full_ids + [tokenizer.eos_token_id]
        full_ids = full_ids[:max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        attn = [1] * len(full_ids)
        pad = max_len - len(full_ids)
        if pad > 0:
            pad_id = tokenizer.pad_token_id
            full_ids = full_ids + [pad_id] * pad
            labels = labels + [-100] * pad
            attn = attn + [0] * pad
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def split_rows(rows: list[dict], eval_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    import torch

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(rows), generator=g).tolist()
    eval_n = min(eval_n, max(1, len(rows) // 4))
    eval_idx = set(perm[:eval_n])
    train = [rows[i] for i in range(len(rows)) if i not in eval_idx]
    ev = [rows[i] for i in range(len(rows)) if i in eval_idx]
    return train, ev


def mean_loss(model, loader, device) -> float:
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            total += float(loss.item())
            n += 1
    model.train()
    return total / max(n, 1)


def train_one(args: argparse.Namespace) -> str:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configure_mlflow()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data).resolve()
    rows = load_jsonl(data_path)
    train_rows, eval_rows = split_rows(rows, args.eval_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = SFTDataset(train_rows, tokenizer, args.max_len)
    eval_ds = SFTDataset(eval_rows, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    commit = git_commit()
    dirty = git_dirty()
    dver = data_version(data_path)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_param("base_model", BASE_MODEL)
        mlflow.log_param("data_version", dver)
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("git_commit", commit)
        mlflow.log_param("git_dirty", str(dirty).lower())
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("lora_r", args.lora_r)
        mlflow.log_param("lora_alpha", args.lora_alpha)
        mlflow.log_param("lora_dropout", args.lora_dropout)
        mlflow.log_param("num_epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("max_seq_len", args.max_len)
        mlflow.log_param("train_size", len(train_rows))
        mlflow.log_param("eval_size", len(eval_rows))
        mlflow.log_param("seed", args.seed)
        mlflow.set_tag("mlflow.source.git.commit", commit)
        mlflow.set_tag("data_version", dver)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mlflow.log_param("trainable_params", trainable)
        mlflow.log_param("total_params", total)

        step = 0
        t0 = time.perf_counter()
        model.train()
        for epoch in range(args.epochs):
            running = 0.0
            seen = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                opt.step()
                step += 1
                running += float(loss.item())
                seen += 1
                mlflow.log_metric("train_loss", float(loss.item()), step=step)
            epoch_loss = running / max(seen, 1)
            ev_loss = mean_loss(model, eval_loader, device)
            mlflow.log_metric("epoch_train_loss", epoch_loss, step=epoch + 1)
            mlflow.log_metric("eval_loss", ev_loss, step=epoch + 1)
            print(
                f"epoch {epoch + 1}/{args.epochs}  "
                f"train_loss={epoch_loss:.4f}  eval_loss={ev_loss:.4f}"
            )

        train_s = time.perf_counter() - t0
        mlflow.log_metric("train_seconds", train_s)
        final_eval = mean_loss(model, eval_loader, device)
        mlflow.log_metric("final_eval_loss", final_eval)

        adapter_dir = HERE / "outputs" / f"adapter-{run.info.run_id[:8]}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Metadata stub: registering the adapter dir as an MLflow model.
        # Loading the 3B base inside pyfunc.predict is out of scope for this lab.
        class AdapterPyFunc(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                cfg_path = Path(context.artifacts["adapter"]) / "adapter_config.json"
                self.config = json.loads(cfg_path.read_text())

            def predict(self, context, model_input):
                import pandas as pd

                n = len(model_input)
                return pd.DataFrame(
                    {
                        "peft_type": [self.config.get("peft_type")] * n,
                        "r": [self.config.get("r")] * n,
                        "base_model": [self.config.get("base_model_name_or_path")] * n,
                    }
                )

        mlflow.pyfunc.log_model(
            name="model",
            python_model=AdapterPyFunc(),
            artifacts={"adapter": str(adapter_dir)},
            pip_requirements=["mlflow", "pandas", "peft", "transformers", "torch"],
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL)
        print(
            f"registered {REGISTERED_MODEL} v{mv.version}  "
            f"run={run.info.run_id}  eval_loss={final_eval:.4f}  "
            f"git={commit[:12]} dirty={dirty} data={dver}"
        )
        if args.promote:
            promote_version(mv.version, archive_existing=True)
        return str(mv.version)


def promote_version(version: str | int, archive_existing: bool = True) -> float:
    """Point Production at `version`. Returns wall time of the stage call."""
    configure_mlflow()
    client = MlflowClient()
    # Stage API plus alias so MLflow 3 UI shows the same Production pointer.
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.transition_model_version_stage(
            name=REGISTERED_MODEL,
            version=str(version),
            stage="Production",
            archive_existing_versions=archive_existing,
        )
    stage_s = time.perf_counter() - t0
    client.set_registered_model_alias(
        name=REGISTERED_MODEL,
        alias="production",
        version=str(version),
    )
    print(
        f"{REGISTERED_MODEL} v{version} -> Production  "
        f"(transition_model_version_stage {stage_s:.4f}s)"
    )
    return stage_s


def rollback_to(version: str | int) -> float:
    return promote_version(version, archive_existing=True)


def print_status() -> None:
    configure_mlflow()
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        print(f"no versions of {REGISTERED_MODEL}")
        return
    print(f"{'ver':>4}  {'stage':<12}  {'run':<32}  eval_loss  git_commit      data_version")
    for mv in sorted(versions, key=lambda x: int(x.version)):
        run = client.get_run(mv.run_id)
        p = run.data.params
        metrics = run.data.metrics
        eval_loss = metrics.get("final_eval_loss", float("nan"))
        print(
            f"{mv.version:>4}  {mv.current_stage:<12}  {mv.run_id:<32}  "
            f"{eval_loss:9.4f}  {p.get('git_commit', '?')[:12]:<12}    "
            f"{p.get('data_version', '?')}"
        )
    try:
        prod = client.get_model_version_by_alias(REGISTERED_MODEL, "production")
        print(f"alias @production -> v{prod.version}")
    except Exception:
        print("alias @production -> (unset)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="1-epoch LoRA SFT + register")
    t.add_argument("--run-name", default=None)
    t.add_argument("--data", default=str(DEFAULT_DATA))
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lora-r", type=int, default=8)
    t.add_argument("--lora-alpha", type=int, default=16)
    t.add_argument("--lora-dropout", type=float, default=0.05)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--max-len", type=int, default=384)
    t.add_argument("--eval-size", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--promote", action="store_true", help="mark this version Production")

    r = sub.add_parser("rollback", help="one-command Production rollback")
    r.add_argument("--to-version", required=True, help="registry version to restore")

    pr = sub.add_parser("promote", help="mark an existing version Production")
    pr.add_argument("--version", required=True)

    sub.add_parser("status", help="print registry + lineage table")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train_one(args)
    elif args.cmd == "rollback":
        s = rollback_to(args.to_version)
        print(f"rollback_seconds={s:.4f}")
        print_status()
    elif args.cmd == "promote":
        promote_version(args.version)
        print_status()
    elif args.cmd == "status":
        print_status()


if __name__ == "__main__":
    main()
