"""Controlled task generators for the Phase-0 complexity audit.

The generators return symbolic records instead of model tensors.  Each task
has an explicit dependency depth and can be rendered/tokenized independently.
The fixed-length pointer task pads every instance to the same number of
records so length cannot reveal the required sequential depth.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable


@dataclass(frozen=True)
class TaskSample:
    task: str
    input_text: str
    target: str
    dependency_depth: int
    input_length: int
    metadata: dict


def _pairs(mapping: dict[int, int]) -> str:
    return " ".join(f"K{k} V{v}" for k, v in mapping.items())


def direct_associative_lookup(rng: random.Random, depth: int = 1, n_keys: int = 16) -> TaskSample:
    mapping = {k: rng.randrange(10_000) for k in range(n_keys)}
    key = rng.randrange(n_keys)
    text = f"LOOKUP {_pairs(mapping)} QUERY K{key} ANSWER"
    return TaskSample("direct_lookup", text, f"V{mapping[key]}", 1, len(text.split()), {"key": key})


def multi_query_associative_recall(rng: random.Random, depth: int = 1, n_keys: int = 24, queries: int = 4) -> TaskSample:
    mapping = {k: rng.randrange(10_000) for k in range(n_keys)}
    keys = rng.sample(list(mapping), queries)
    text = f"RECALL {_pairs(mapping)} QUERY " + " ".join(f"K{k}" for k in keys) + " ANSWER"
    target = " ".join(f"V{mapping[k]}" for k in keys)
    return TaskSample("multi_query_recall", text, target, queries, len(text.split()), {"keys": keys})


def pointer_chasing(rng: random.Random, dependency_depth: int = 4, n_nodes: int = 64) -> TaskSample:
    nxt = list(range(n_nodes))
    rng.shuffle(nxt)
    start = rng.randrange(n_nodes)
    current = start
    for _ in range(dependency_depth):
        current = nxt[current]
    text = f"POINTER START N{start} " + " ".join(f"N{i} N{nxt[i]}" for i in range(n_nodes)) + " ANSWER"
    return TaskSample("pointer_chasing", text, f"N{current}", dependency_depth, len(text.split()), {"start": start})


def adaptive_pointer_chasing(rng: random.Random, dependency_depth: int = 4, n_nodes: int = 64) -> TaskSample:
    table = {i: rng.randrange(n_nodes) for i in range(n_nodes)}
    start = rng.randrange(n_nodes)
    current = start
    for _ in range(dependency_depth):
        # The next address is a value-dependent transform, not a fixed edge.
        current = (table[current] + current * 3 + 1) % n_nodes
    text = f"ADAPTIVE START N{start} " + " ".join(f"N{i} V{table[i]}" for i in range(n_nodes)) + " ANSWER"
    return TaskSample("adaptive_pointer_chasing", text, f"N{current}", dependency_depth, len(text.split()), {"start": start})


def state_tracking(rng: random.Random, dependency_depth: int = 8) -> TaskSample:
    state = rng.randrange(100)
    initial = state
    ops: list[str] = []
    for _ in range(dependency_depth):
        delta = rng.randrange(-9, 10)
        state += delta
        ops.append(f"{'ADD' if delta >= 0 else 'SUB'} {abs(delta)}")
    text = f"STATE {initial} " + " ".join(ops) + " QUERY ANSWER"
    return TaskSample("state_tracking", text, str(state), dependency_depth, len(text.split()), {})


def iterated_function_composition(rng: random.Random, dependency_depth: int = 8, modulus: int = 97) -> TaskSample:
    funcs = [(rng.randrange(modulus), rng.randrange(modulus)) for _ in range(4)]
    x = rng.randrange(modulus)
    value = x
    chosen: list[int] = []
    for _ in range(dependency_depth):
        f = rng.randrange(len(funcs))
        chosen.append(f)
        a, b = funcs[f]
        value = (a * value + b) % modulus
    text = f"COMPOSE MOD {modulus} X {x} " + " ".join(
        f"F{i} {a} {b}" for i, (a, b) in enumerate(funcs)
    ) + " APPLY " + " ".join(f"F{i}" for i in chosen) + " ANSWER"
    return TaskSample("iterated_function", text, str(value), dependency_depth, len(text.split()), {"functions": chosen})


def distractor_controlled_retrieval(rng: random.Random, dependency_depth: int = 1, n_records: int = 64) -> TaskSample:
    target_key = rng.randrange(n_records)
    target_value = rng.randrange(10_000)
    records = [(i, rng.randrange(10_000)) for i in range(n_records)]
    records[target_key] = (target_key, target_value)
    rng.shuffle(records)
    text = "DISTRACTOR " + " ".join(f"K{k} V{v}" for k, v in records) + f" QUERY K{target_key} ANSWER"
    return TaskSample("distractor_retrieval", text, f"V{target_value}", 1, len(text.split()), {"key": target_key})


def fixed_length_pointer_task(
    rng: random.Random,
    dependency_depth: int,
    max_dependency_depth: int = 32,
    n_nodes: int = 64,
) -> TaskSample:
    """Pointer task with identical rendered length for every dependency depth."""
    nxt = list(range(n_nodes))
    rng.shuffle(nxt)
    start = rng.randrange(n_nodes)
    current = start
    for _ in range(dependency_depth):
        current = nxt[current]
    # A fixed number of inert records and fixed-width control fields keep the
    # sequence length constant across train and OOD depth tiers.
    records = " ".join(f"N{i:02d} N{nxt[i]:02d}" for i in range(n_nodes))
    hop_markers = " ".join("STEP" for _ in range(dependency_depth))
    padding = " ".join("FILL" for _ in range(max_dependency_depth - dependency_depth))
    # The number of STEP markers specifies the computation to perform, but
    # every example has the same total number of marker/filler slots.
    text = f"FIXED START N{start:02d} HOPS {hop_markers} {padding} {records} ANSWER"
    return TaskSample("fixed_length_pointer", text, f"N{current:02d}", dependency_depth, len(text.split()), {"start": start})


TASK_GENERATORS: dict[str, Callable[..., TaskSample]] = {
    "direct_lookup": direct_associative_lookup,
    "multi_query_recall": multi_query_associative_recall,
    "pointer_chasing": pointer_chasing,
    "adaptive_pointer_chasing": adaptive_pointer_chasing,
    "state_tracking": state_tracking,
    "iterated_function": iterated_function_composition,
    "distractor_retrieval": distractor_controlled_retrieval,
    "fixed_length_pointer": fixed_length_pointer_task,
}


def make_suite(seed: int = 0, n_per_tier: int = 64) -> list[TaskSample]:
    rng = random.Random(seed)
    train_depths = (1, 2, 4, 8)
    ood_depths = (12, 16, 24, 32)
    samples: list[TaskSample] = []
    for depth in train_depths + ood_depths:
        tier = "train" if depth in train_depths else "ood"
        for _ in range(n_per_tier):
            sample = fixed_length_pointer_task(rng, depth)
            samples.append(
                TaskSample(sample.task, sample.input_text, sample.target, sample.dependency_depth, sample.input_length, {**sample.metadata, "tier": tier})
            )
    for name, generator in TASK_GENERATORS.items():
        if name == "fixed_length_pointer":
            continue
        for _ in range(n_per_tier):
            if name in {"direct_lookup", "multi_query_recall", "distractor_retrieval"}:
                samples.append(generator(rng))
            else:
                depth = rng.choice(train_depths)
                samples.append(generator(rng, dependency_depth=depth))
    return samples
