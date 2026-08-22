# Controlled sequential-dependency task suite

The executable generators are in [`src/project_alpha/tasks.py`](../src/project_alpha/tasks.py)
and the stability check is [`scripts/run_task_suite.py`](../scripts/run_task_suite.py).
A sample contains symbolic input, exact target, dependency depth, rendered
length, and generator metadata. This makes true difficulty available for
analysis without exposing future losses to a controller.

## Tasks

| Task | Controlled dependency | Confound control |
|---|---|---|
| Direct associative lookup | one key/value read | fixed-size table; query key sampled uniformly |
| Multi-query associative recall | multiple independent reads | table size held fixed; query count recorded |
| Pointer chasing | follow a random address map for (D) hops | node-table size held fixed |
| Adaptive pointer chasing | next address is a value-dependent transform of the previous address | fixed node-table size; no length-coded path |
| State tracking | apply signed updates to a scalar state for (D) steps | operation widths sampled independently |
| Iterated function composition | apply random modular affine functions for (D) steps | modulus and function library fixed |
| Distractor-controlled retrieval | retrieve one record among distractors | record count and rendered length fixed |
| Fixed-length pointer | follow a pointer chain for (D) hops with `STEP`/`FILL` slots | depths 1,2,4,8,12,16,24,32 render exactly the same length |

The main train/OOD split is:

```text
train D = {1, 2, 4, 8}
OOD   D = {12, 16, 24, 32}
```

The fixed-length generator makes the computation marker count different while
keeping total length identical. This tests whether a model responds to
sequential structure rather than merely sequence length. Balanced variants
should randomize superficial markers before claiming algorithmic
generalization.

## Stability gate

The initial generator audit used seed 1729 and eight samples per depth.
[`results/phase0_task_stability.json`](../results/phase0_task_stability.json)
records `status: pass`, all eight task constructors, and length 165 for every
fixed-length depth. The suite does not claim neural-model accuracy; it verifies
that target generation is correct and planned OOD tiers do not change renderer
length.

## Required measurements in the adaptive phase

For every task and depth, report exact accuracy, loss, allocated depth, oracle
(d^*_{\epsilon}), and matched-compute comparisons against fixed depth, early
depth prediction, ACT/PonderNet, ANIRA-style online halting, and a
representation-drift/KL exit. Balance frequency, position, input length, and
lexical markers. Never label an allocation “reasoning” from accuracy or
perplexity alone.
