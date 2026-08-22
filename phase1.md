# PHASE 1 — 10–15M WIKITEXT-103 FULL PRETRAINING
## Establish the latent-compute response curve before adaptive routing

You are continuing Project Alpha after Phase 0.

Read all Phase-0 documents before modifying code.

Do not change the Phase-0 scientific hypothesis without documenting the reason.

---

# OBJECTIVE

Build a clean 10–15 million parameter causal language model based on Mamba-3 plus a SMALL weight-tied latent reasoning cell.

Fully pretrain it on WikiText-103 for approximately 2–3 complete passes over the training corpus.

The purpose is NOT SOTA WikiText perplexity.

The questions are:

1. Does extra latent computation improve language prediction at all?
2. Is its benefit heterogeneous across tokens/examples?
3. Does the benefit saturate at different depths?
4. Can we obtain hindsight "required depth" targets?
5. Is the implementation fast and numerically stable enough to support later experiments?

DO NOT add learned adaptive halting yet.

---

# ENVIRONMENT

First detect:
- OS
- GPU
- VRAM
- CUDA vs ROCm
- PyTorch version
- Triton version
- compiler toolchain

Record everything in:

    runs/environment.json

If running locally on an RTX 3050 6GB:
- use Linux/WSL2 if necessary for the required libraries;
- prioritize correctness and memory efficiency;
- use gradient accumulation;
- sequence length may begin at 512 and move to 1024 if practical.

If running on MI300X:
- use BF16;
- benchmark the latest supported ROCm PyTorch stack;
- do not assume Mamba-3 fast kernels work correctly until unit tested.

---

# SOFTWARE STACK

Use:

Python + PyTorch for orchestration.

Do NOT use pure C++.

For performance:

1. PyTorch fused operations / hipBLASLt / cuBLAS underneath torch.matmul;
2. FlashAttention-2 or optimized PyTorch SDPA for the reasoning cell;
3. official Mamba-3 implementation if correct on target hardware;
4. Triton for missing/fused kernels;
5. C++/CUDA or C++/HIP ONLY for a profiler-proven bottleneck.

Before writing a custom kernel:
- create a reference;
- unit test output and gradients;
- benchmark it;
- profile wall-clock contribution.

Never optimize an operation because it "looks expensive."

---

# MODEL TARGET

Target measured trainable parameters:

    10M <= params <= 15M

Do not trust rough formulas; print exact count.

Suggested starting point, adjust as needed:

- vocab: 32k BPE
- d_model ~256
- 6–8 Mamba-3 stream blocks
- d_state 64 or 128
- headdim 64
- tied input/output embeddings
- RMSNorm
- SwiGLU where needed
- dropout 0 unless overfitting becomes severe

Use SISO Mamba-3 first if MIMO materially complicates portability.
MIMO can be introduced only after correctness is established.

---

# LATENT REASONING CELL

After causal stream encoding H, define a token state:

    z_i^(0) = h_i

Use ONE weight-tied recurrent reasoning cell R_theta.

Preferred first implementation:

    q_i^(r) = RMSNorm(z_i^(r))
    a_i^(r) = causal_cross_attention(
        query=q_i^(r),
        keys=H_<=i,
        values=H_<=i
    )
    u_i^(r) = z_i^(r) + W_o a_i^(r)
    z_i^(r+1) = u_i^(r) + SwiGLU(RMSNorm(u_i^(r)))

Keys/values from H are fixed across recurrence.

This is important:
extra recurrence = extra reasoning over the already encoded causal context,
not a complete re-encoding of the sequence.

Use FlashAttention/SDPA where possible.

The same output head must work at every depth.

---

# TRAINING RECURRENCE

Train the shared reasoner with variable depth.

Start with:

    T_train in {1, 2, 4}

sampled per batch.

Optionally include T=8 with low probability if stable and affordable.

Use deep supervision:

    L = weighted_mean_r CE(logits^(r), targets)

Do not make all auxiliary depths equally dominant without testing.
The final sampled depth should carry the main objective.

Also train a plain parameter-matched Mamba-3 baseline with NO recurrent reasoner.

---

# TOKENIZER/DATA

Download and preprocess WikiText-103 reproducibly.

Train or use a fixed 32k tokenizer based only on training text.

Record:
- tokenizer type
- vocabulary
- checksum
- corpus revision/checksum
- number of resulting train/validation/test tokens

Do not assume "103M words/tokens" equals our tokenizer count.

Pack causal sequences efficiently.

Initial sequence length:
    512

Try:
    1024

only if throughput remains acceptable.

---

# OPTIMIZER

Use AdamW.

Do not test Muon/SOAP here.
That would confound architecture research.

Starting range:
- beta1 = 0.9
- beta2 = 0.95
- weight decay = 0.1
- BF16 autocast when supported
- FP32 optimizer state
- grad clip = 1.0

Run a short LR range pilot rather than blindly copying a large-model LR.

Likely candidate LRs:
    2e-4, 3e-4, 5e-4, 6e-4

Select from short controlled runs.

Warm up 1–2% of total steps.
Cosine decay afterward.

---

# FULL PRETRAINING

Perform 2 full epochs initially.

Run a third epoch if:
- validation is still improving materially,
- no severe overfitting,
- compute budget allows.

Checkpoint frequently enough to recover from failure.

Every run must be resumable bit-for-bit in data order.

---

# CRITICAL EVALUATION

At validation time evaluate the SAME checkpoint at:

    T = 0, 1, 2, 4, 8

where T=0 means no recurrent reasoning cell if architecturally possible.

For every target token store:

    loss_0
    loss_1
    loss_2
    loss_4
    loss_8
    entropy_0
    token_id
    position
    token_frequency
    context_length
    hidden diagnostics

Compute:

    Delta_i(r1 -> r2) = loss_i(r1) - loss_i(r2)

and define hindsight oracle depth:

    d*_epsilon(i)

for multiple epsilon values.

Suggested:
    epsilon in {0.01, 0.03, 0.05, 0.1} nats

---

# ANALYSES

Produce:

1. validation loss vs recurrence depth;
2. latency vs recurrence depth;
3. token-level distribution of marginal gain;
4. percentage of tokens harmed by extra recurrence;
5. percentage helped;
6. depth saturation distribution;
7. correlation of d* with:
   - token entropy,
   - token frequency,
   - position,
   - context length,
   - punctuation/code/numeric tokens if detectable;
8. examples with:
   - high marginal value;
   - zero marginal value;
   - negative marginal value.

Important:
do NOT infer "reasoning" merely because recurrence improves perplexity.

---

# PERFORMANCE PROFILING

Benchmark separately:

- Mamba stream
- QKV projection
- reasoning attention
- reasoning MLP
- recurrence loop overhead
- output projection
- optimizer
- dataloader

Use synchronized timing.

Measure:
- tokens/sec
- examples/sec
- peak VRAM/HBM
- TFLOP estimate if practical
- kernel launch overhead
- percentage time by component

Use torch.profiler initially.
On ROCm also use appropriate ROCm profiling tools if available.

---

# CUSTOM KERNEL POLICY

Do not implement a HIP/CUDA kernel during this phase unless profiling proves it is necessary.

If a custom operation is needed:

Reference:
    PyTorch

First optimization:
    Triton

Only if needed:
    C++/HIP on AMD
    C++/CUDA on NVIDIA

Write forward + backward tests:
- FP32 numerical reference
- BF16 tolerances
- gradient check on small tensors
- randomized shapes
- odd/non-power-of-two shapes

---

# REQUIRED BASELINES

At minimum:

A. plain Mamba-3, parameter matched;
B. Mamba-3 + recurrent reasoner, T=1;
C. same checkpoint evaluated at T=2,4,8.

---

# PHASE-1 GO/NO-GO CRITERIA

GO only if:

1. training is numerically stable;
2. reasoner can be trained without catastrophic perplexity degradation;
3. marginal value of recurrence is meaningfully heterogeneous;
4. some subset of tokens benefits from extra depth;
5. not every token simply prefers maximum T;
6. throughput is measurable and reproducible.

A particularly encouraging result is:

    substantial variance in d*_epsilon across tokens

while simple features like token frequency/entropy do not perfectly explain it.

Write final report:

    research/phase1_results.md

and save machine-readable measurements in:

    results/phase1/

Do not implement adaptive halting until this analysis is complete.
