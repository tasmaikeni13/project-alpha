#!/usr/bin/env python3
"""Train the fixed 32k BPE tokenizer and pack WikiText-103 reproducibly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def parquet_paths(split: str) -> list[Path]:
    if split == "train":
        return [RAW / "train-00000-of-00002.parquet", RAW / "train-00001-of-00002.parquet"]
    return [RAW / f"{split}-00000-of-00001.parquet"]


def text_batches(split: str, batch_size: int = 4096) -> Iterator[list[str]]:
    for path in parquet_paths(split):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            yield batch.column("text").to_pylist()


def text_iterator(split: str) -> Iterator[str]:
    for batch in text_batches(split):
        yield from batch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=32_000,
        min_frequency=2,
        special_tokens=SPECIAL,
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator("train"), trainer=trainer)
    return tokenizer


def encode_batches(tokenizer: Tokenizer, split: str) -> Iterator[np.ndarray]:
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise RuntimeError("tokenizer did not create <eos>")
    for batch in text_batches(split):
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        yield from (np.asarray(enc.ids + [eos], dtype=np.uint32) for enc in encodings)


def pack_split(tokenizer: Tokenizer, split: str) -> tuple[Path, int]:
    count = sum(int(len(ids)) for ids in encode_batches(tokenizer, split))
    out_path = OUT / f"{split}_tokens.npy"
    mmap = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint32, shape=(count,))
    cursor = 0
    for ids in encode_batches(tokenizer, split):
        mmap[cursor : cursor + len(ids)] = ids
        cursor += len(ids)
    mmap.flush()
    del mmap
    return out_path, count


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer_path = OUT / "tokenizer.json"
    tokenizer = make_tokenizer()
    tokenizer.save(str(tokenizer_path))
    counts: dict[str, int] = {}
    token_paths: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path, count = pack_split(tokenizer, split)
        token_paths[split] = str(path.relative_to(ROOT))
        counts[split] = count

    raw_manifest = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for split in ("train", "validation", "test")
        for path in parquet_paths(split)
    }
    metadata = {
        "dataset": "Salesforce/wikitext",
        "configuration": "wikitext-103-raw-v1",
        "source_revision": "main (retrieved 2026-08-22)",
        "source_files": raw_manifest,
        "tokenizer_type": "ByteLevel BPE",
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": sha256(tokenizer_path),
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL},
        "token_counts": counts,
        "token_paths": token_paths,
        "packing": "concatenate each parquet text row followed by <eos>; fixed contiguous windows",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
