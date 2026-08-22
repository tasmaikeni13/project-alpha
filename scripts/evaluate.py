#!/usr/bin/env python3
"""Evaluate one checkpoint at T=0,1,2,4,8 and write token-level artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from project_alpha.data import PackedTokenDataset
from project_alpha.model import AlphaLM, ModelConfig


ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (0, 1, 2, 4, 8)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[AlphaLM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["config"])
    model = AlphaLM(config, with_reasoner=bool(checkpoint["with_reasoner"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    x, y = x[mask], y[mask]
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denom) if denom > 0 else None


def token_features(tokenizer_path: Path, token_ids: np.ndarray) -> dict[str, np.ndarray]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    strings = tokenizer.decode_batch([[int(i)] for i in token_ids], skip_special_tokens=False)
    punctuation = np.asarray([bool(re.search(r"[^\w\s]", s)) for s in strings], dtype=np.float32)
    numeric = np.asarray([bool(re.search(r"\d", s)) for s in strings], dtype=np.float32)
    code_like = np.asarray([bool(re.search(r"[{}<>/=;]|\\\\|::|==", s)) for s in strings], dtype=np.float32)
    return {"is_punctuation": punctuation, "is_numeric": numeric, "is_code_like": code_like, "token_text": np.asarray(strings, dtype=object)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="validation", choices=("validation", "test"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--output-dir", default="results/phase1")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((ROOT / "data" / "processed" / "metadata.json").read_text())
    model, checkpoint = load_model(ROOT / args.checkpoint, device)
    dataset = PackedTokenDataset(ROOT / metadata["token_paths"][args.split], args.sequence_length)
    train_tokens = np.load(ROOT / metadata["token_paths"]["train"], mmap_mode="r")
    frequencies = np.bincount(np.asarray(train_tokens, dtype=np.int64), minlength=model.config.vocab_size)
    n_examples = dataset.n_examples if args.max_examples is None else min(dataset.n_examples, args.max_examples)

    eval_depths = DEPTHS if model.with_reasoner else (0,)
    losses: dict[int, list[np.ndarray]] = {d: [] for d in eval_depths}
    entropies: list[np.ndarray] = []
    token_ids: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    context_lengths: list[np.ndarray] = []
    diagnostics: dict[str, list[np.ndarray]] = {}
    elapsed: dict[int, list[float]] = {d: [] for d in eval_depths}
    with torch.inference_mode():
        for start in range(0, n_examples, args.batch_size):
            indices = list(range(start, min(start + args.batch_size, n_examples)))
            inputs, targets = dataset.batch(indices, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                t0 = time.perf_counter()
                hidden, diag = model.hidden_at_depths(inputs, eval_depths, return_diagnostics=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                stream_elapsed = time.perf_counter() - t0
                for key, value in diag.items():
                    diagnostics.setdefault(key, []).append(value.float().cpu().numpy().reshape(-1))
                for d in eval_depths:
                    t_depth = time.perf_counter()
                    logits = model.logits(hidden[d]).float()
                    flat_targets = targets.reshape(-1)
                    log_probs = F.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
                    token_loss = (-log_probs.gather(1, flat_targets[:, None]).squeeze(1)).cpu().numpy()
                    losses[d].append(token_loss)
                    elapsed[d].append(stream_elapsed if d == 0 else time.perf_counter() - t_depth)
                    if d == 0:
                        entropy = (-(log_probs.exp() * log_probs).sum(dim=-1)).cpu().numpy()
                        entropies.append(entropy)
            target_np = targets.cpu().numpy().reshape(-1)
            token_ids.append(target_np)
            batch_len = targets.shape[1]
            position = np.tile(np.arange(batch_len, dtype=np.int32), targets.shape[0])
            positions.append(position)
            context_lengths.append(position + 1)

    arrays: dict[str, np.ndarray] = {}
    for d in eval_depths:
        arrays[f"loss_{d}"] = np.concatenate(losses[d]).astype(np.float32)
    arrays["entropy_0"] = np.concatenate(entropies).astype(np.float32)
    arrays["token_id"] = np.concatenate(token_ids).astype(np.int64)
    arrays["position"] = np.concatenate(positions).astype(np.int32)
    arrays["context_length"] = np.concatenate(context_lengths).astype(np.int32)
    for key, values in diagnostics.items():
        arrays[key] = np.concatenate(values).astype(np.float32)
    arrays["token_frequency"] = frequencies[arrays["token_id"]].astype(np.int64)
    for left, right in zip(eval_depths, eval_depths[1:]):
        arrays[f"delta_{left}_to_{right}"] = arrays[f"loss_{left}"] - arrays[f"loss_{right}"]
    if model.with_reasoner:
        arrays["delta_0_to_8"] = arrays["loss_0"] - arrays["loss_8"]

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{args.split}_{Path(args.checkpoint).parent.name}.npz"
    np.savez_compressed(out_dir / artifact_name, **arrays)

    best_depth = np.asarray(eval_depths)[np.argmin(np.stack([arrays[f"loss_{d}"] for d in eval_depths]), axis=0)]
    dstar: dict[str, np.ndarray] = {}
    for epsilon in (0.01, 0.03, 0.05, 0.1):
        min_loss = np.min(np.stack([arrays[f"loss_{d}"] for d in eval_depths]), axis=0)
        selected = np.full_like(best_depth, eval_depths[-1], dtype=np.int16)
        for d in eval_depths:
            selected[(selected == eval_depths[-1]) & (arrays[f"loss_{d}"] <= min_loss + epsilon)] = d
        dstar[f"{epsilon:.2f}"] = selected
        arrays[f"dstar_epsilon_{epsilon:.2f}"] = selected
    np.savez_compressed(out_dir / artifact_name, **arrays)

    feature_values = token_features(ROOT / "data" / "processed" / "tokenizer.json", arrays["token_id"])
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "examples": n_examples,
        "target_tokens": int(len(arrays["token_id"])),
        "model_parameters": sum(v.numel() for v in model.parameters()),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "validation_loss": {str(d): float(arrays[f"loss_{d}"].mean()) for d in eval_depths},
        "perplexity": {str(d): float(math.exp(min(20.0, arrays[f"loss_{d}"].mean()))) for d in eval_depths},
        "mean_marginal_gain": {f"{left}_to_{right}": float(arrays[f"delta_{left}_to_{right}"].mean()) for left, right in zip(eval_depths, eval_depths[1:])},
        "marginal_gain_std": {f"{left}_to_{right}": float(arrays[f"delta_{left}_to_{right}"].std()) for left, right in zip(eval_depths, eval_depths[1:])},
        "helped_fraction_0_to_8": float(np.mean(arrays["delta_0_to_8"] > 1e-6)) if model.with_reasoner else None,
        "harmed_fraction_0_to_8": float(np.mean(arrays["delta_0_to_8"] < -1e-6)) if model.with_reasoner else None,
        "zero_fraction_0_to_8": float(np.mean(np.abs(arrays["delta_0_to_8"]) <= 1e-6)) if model.with_reasoner else None,
        "best_depth_histogram": {str(d): int(np.sum(best_depth == d)) for d in eval_depths},
        "dstar_histograms": {eps: {str(d): int(np.sum(values == d)) for d in eval_depths} for eps, values in dstar.items()},
        "correlation_dstar_0.05": {
            "entropy_0": pearson(dstar["0.05"], arrays["entropy_0"]),
            "log_token_frequency": pearson(dstar["0.05"], np.log1p(arrays["token_frequency"])),
            "position": pearson(dstar["0.05"], arrays["position"]),
            "context_length": pearson(dstar["0.05"], arrays["context_length"]),
            "is_punctuation": pearson(dstar["0.05"], feature_values["is_punctuation"]),
            "is_numeric": pearson(dstar["0.05"], feature_values["is_numeric"]),
            "is_code_like": pearson(dstar["0.05"], feature_values["is_code_like"]),
        } if model.with_reasoner else {},
        "logit_seconds_per_batch_approx": {str(d): float(np.mean(elapsed[d])) for d in eval_depths},
        "token_feature_rates": {key: float(value.mean()) for key, value in feature_values.items() if key != "token_text"},
        "artifact": str((out_dir / artifact_name).relative_to(ROOT)),
    }

    tokenizer = __import__("tokenizers").Tokenizer.from_file(str(ROOT / "data" / "processed" / "tokenizer.json"))
    examples: dict[str, list[dict]] = {}
    transitions = [f"{left}_to_{right}" for left, right in zip(eval_depths, eval_depths[1:])]
    if model.with_reasoner:
        transitions.append("0_to_8")
    for transition in transitions:
        gain = arrays[f"delta_{transition}"]
        order = np.argsort(gain)
        chosen = list(order[:5]) + list(order[len(order) // 2 - 2 : len(order) // 2 + 3]) + list(order[-5:])
        examples[transition] = [
            {"index": int(i), "token_id": int(arrays["token_id"][i]), "token": tokenizer.decode([int(arrays["token_id"][i])], skip_special_tokens=False), "gain": float(gain[i]), "position": int(arrays["position"][i])}
            for i in chosen
        ]
    (out_dir / f"{args.split}_{Path(args.checkpoint).parent.name}_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n")
    (out_dir / f"{args.split}_{Path(args.checkpoint).parent.name}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
