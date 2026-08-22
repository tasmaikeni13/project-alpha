# Phase-1 execution plan

Phase 1 establishes the latent-compute response curve before learned halting.
It does not alter the Phase-0 hypothesis. The Phase-0 audit narrows the future
claim because ANIRA and the 2026 Looped-Mamba paper cover much of the broad
idea; that reason is documented in [`novelty_risks.md`](novelty_risks.md).

## Environment and portability

The target is one AMD Instinct MI300X VF (`gfx942`) with approximately 192 GiB
visible HBM, ROCm 7.0.2/7.14 runtime, PyTorch `2.10.0+rocm7.0`, and Triton
`3.6.0`. Complete command capture is in
[`runs/environment.json`](../runs/environment.json). BF16 autocast is used,
optimizer state remains FP32, and the official Mamba-3 Triton SISO path is not
used until its forward and gradient behavior passes on this target.

## Model and data

- ByteLevel BPE, vocabulary exactly 32,000, trained only on WikiText-103
  training text.
- Raw `Salesforce/wikitext`, `wikitext-103-raw-v1` parquet files, packed into
  contiguous 512-token examples with `<eos>` between text rows.
- Mamba-3 SISO reference stream: width 256, six blocks, state 64, head 64.
- One shared recurrent cross-attention/SwiGLU reasoner; fixed Mamba context;
  tied embedding/output head; maximum evaluated depth 8.
- Plain seven-block Mamba-3 baseline, with measured parameter count printed
  rather than inferred from formulas.

## Optimization and reproducibility

AdamW uses betas `(0.9, 0.95)`, weight decay `0.1`, gradient clipping `1.0`,
BF16 autocast, 1.5% warmup, and cosine decay. The short pilot evaluates
`2e-4, 3e-4, 5e-4, 6e-4` on the same batches; the selected LR is stored in
`runs/lr_pilot.json`. Every checkpoint stores model/optimizer/scheduler state,
Python/NumPy/PyTorch/CUDA RNG states, the epoch permutation, and batch cursor
so data order is resumable.

The initial run is two complete passes over the training token stream. A third
pass is allowed only if validation materially improves and there is no severe
overfit. Checkpoints are written every 500 optimizer steps.

## Evaluation and profiling

The same reasoner checkpoint is evaluated at (T=0,1,2,4,8). The artifact
stores per target token: `loss_0`, `loss_1`, `loss_2`, `loss_4`, `loss_8`,
`entropy_0`, `token_id`, `position`, `token_frequency`, `context_length`, and
hidden/step norms. It additionally stores adjacent marginal gains, multiple
(d^*_{\epsilon}) arrays, example records, correlations, and summary JSON. The
profiling script times stream, context projections, reasoning, MLP, output
projection, recurrence overhead, end-to-end depth, peak HBM, and tokens/sec
under synchronized timing, with an optional `torch.profiler` trace.

## Gates

GO to adaptive routing only if training is finite, the reasoner is not
catastrophically worse, marginal gains are heterogeneous, a subset benefits,
not every token prefers maximum depth, and throughput is reproducible. A
failure is retained as a result and should redirect the project to the
narrowest supported question; no adaptive halting code belongs in this phase.
