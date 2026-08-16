# Alpha Architecture

Status: architecture specification, version 0.2  
Formalization: [`AlphaArchitecture/Theorems.lean`](AlphaArchitecture/Theorems.lean)

## 1. Purpose

Alpha is a reasoning-first autoregressive architecture for long-context language modeling. It is designed around three separate jobs:

1. **Stream modeling:** efficiently compress the regular, predictable parts of a sequence.
2. **Exact recall:** preserve and retrieve particular past records without forcing them through a fixed-size recurrent state.
3. **Adaptive reasoning:** spend a variable number of latent reasoning steps on a small working set before producing an answer.

The separation is essential. A fixed-size state can be efficient and stable, but it cannot losslessly represent every possible history as context length grows. Conversely, dense global attention preserves token-level access but materializes a quadratic number of query-key interactions. Alpha therefore combines a recurrent state, local attention, and context-dependent sparse external memory.

Alpha does **not** claim that memory, compute, and exact recall can all be constant for unbounded context. Exact arbitrary retention requires storage that grows with the retained information. Alpha keeps the storage when exactness matters and makes access sparse.

## 2. System overview

```text
tokens
  │
embedding
  │
  ├── repeated stream blocks ──────────────────────────────┐
  │     ├─ damped phase-state mixer                        │
  │     ├─ causal sliding-window attention                 │
  │     ├─ sparse episodic-memory read on scheduled blocks │
  │     └─ gated fusion + feed-forward network             │
  │                                                        │
  └──────────────────────► workspace builder ◄─────────────┘
                                      │
                           adaptive reasoning loop
                          proposer → critic → update
                                      ▲
                         trusted, fixed step budget
                                      │
                              output projection
```

There are two distinct recurrences:

- **Sequence recurrence** moves left-to-right through tokens and is deliberately compressive.
- **Reasoning recurrence** iterates over a bounded workspace. It adds computation depth without asking one state to remember the entire sequence.

## 3. Stream block

Let the input to layer `ℓ` be `H ∈ ℝ^(n×d)`. A block contains three mixers followed by a feed-forward network.

### 3.1 Damped phase-state mixer

Normalize each token first:

`u_t = RMSNorm(h_t)`.

For every state head and phase channel, maintain a pair

`s_t = (q_t, p_t) ∈ ℝ²`.

Token-conditioned projections produce:

- damping `a_t ∈ (a_min, 1)`;
- rotation angle `θ_t ∈ ℝ`;
- interpolation gate `λ_t ∈ (0, 1)`;
- input and output projections `B_t` and `C_t`.

A direct parameterization is

`a_t = a_min + (1 - a_min) sigmoid(raw_a_t)`.

Define the rotation

```text
R(θ) = [[ cos θ, -sin θ ],
        [ sin θ,  cos θ ]].
```

The state update is

```text
g_t = (1 - λ_t) Δ_t B_(t-1) u_(t-1) + λ_t Δ_t B_t u_t
s_t = a_t R(θ_t) s_(t-1) + g_t
y_t = C_t s_t + D u_t
```

For `t = 0`, use a learned start input in place of `B_(t-1)u_(t-1)`. `B_t` and `C_t` may use a low-rank multi-input, multi-output projection so multiple value lanes share the same state update. The recurrence is inspired by the complex state and exponential-trapezoidal input treatment in [Mamba-3](https://arxiv.org/abs/2603.15569), while its two-dimensional phase interpretation supplies Alpha's energy structure.

The unforced phase energy is

`E(q, p) = q² + p²`.

Rotation preserves this energy, and damping changes it to `a_t² E`. Thus `0 ≤ a_t ≤ 1` makes each unforced phase pair non-expansive. Input forcing, residual branches, and feed-forward layers may add energy; the energy theorem does not apply to the whole network without additional bounds.

Each step has the affine form

`s_t = A_t s_(t-1) + b_t`,

where `A_t` and `b_t` depend on input tokens but not on the evolving state. Time-varying affine transitions compose associatively, so prefill can use an exact tree-based prefix scan. Autoregressive decoding applies the same recurrence one token at a time.

### 3.2 Local exact interaction

The local branch uses causal sliding-window attention over the most recent `W` positions:

`local_t = Attention(q_t, K_[t-W:t], V_[t-W:t])`.

This branch handles syntax, short-range copying, and precise recent interactions. Its attention-pair count is at most `nW`, rather than `n²`. Local attention is not the long-term memory mechanism: once an item leaves the window, exact access must come from episodic memory.

### 3.3 Context-dependent episodic memory

An episodic record contains at least

```text
(record_id, key, exact_value, position, layer_id, salience)
```

The value stored for an exact record is not quantized. Summaries may be quantized, but summaries are routing aids and never replace an exact record when lossless recall is required.

Memory has two operating modes:

- **Archival mode:** append all designated records to tiered storage. Storage grows linearly with the number of records; arbitrary retained records are not evicted.
- **Bounded mode:** keep at most `C` records using salience-aware eviction. This bounds memory but explicitly gives up lossless arbitrary recall.

Salience can start with the magnitude of the state-write residual, following the idea that information poorly absorbed by the compressor deserves exact storage. This is a heuristic, not a formal guarantee.

For a query `q`, a context-dependent router returns at most `K` record identifiers:

`S(q, context, memory) ⊆ record_ids`, with `|S| ≤ K`.

The router must depend on content; a fixed sliding or dilated pattern is insufficient for context-conditioned long-range lookup. A first implementation may use locality-sensitive hashing with exact record gathering, following the SSM plus context-dependent sparse-attention construction in [HAX](https://arxiv.org/abs/2507.00449). A learned hierarchical index is also compatible, provided it preserves the `K`-record output contract.

After routing, Alpha gathers the exact keys and values and performs ordinary softmax attention only over those records. Recent local records and a small set of persistent anchor records may be unioned with the routed set.

The exact-memory guarantee is conditional: if the requested record exists and the router selects it, the exact value bypasses the recurrent fallback. No theorem in this repository proves that a learned router will select the correct record.

### 3.4 Mixer fusion

For normalized token `u_t`, compute available branches:

```text
r_t = phase-state output
l_t = local-attention output
e_t = episodic-memory output, or zero on non-memory blocks
```

Use a token-wise gate

`(γ_r, γ_l, γ_e) = softmax(W_gate u_t)`

and fuse them through

```text
h'_t = h_t + W_out(γ_r r_t + γ_l l_t + γ_e e_t)
h_next_t = h'_t + SwiGLU(RMSNorm(h'_t)).
```

Every block contains the phase-state and local branches. Episodic reads are enabled on a configurable subset of blocks, initially every fourth block. All three paths remain independently ablatable.

## 4. Adaptive reasoning core

The reasoning core operates after stream encoding and before the final output projection. It never rescans the complete context on every reasoning step.

### 4.1 Workspace construction

Build `R` workspace slots from:

- the current local window;
- the final recurrent summaries;
- the top episodic records selected for the current problem;
- learned scratch slots.

`R` is fixed and much smaller than context length. Slot zero is the answer/readout slot; the other slots carry evidence and scratch state. The workspace builder may use dense attention because it attends only over this bounded set.

### 4.2 Iterative cognition

Let `Z_0 ∈ ℝ^(R×d)` be the initial workspace. A weight-tied reasoning block performs

```text
proposal_t = Proposer(Z_t, retrieved_memory)
critique_t = Critic(Z_t, proposal_t)
Z_(t+1) = Z_t + Update(Z_t, proposal_t, critique_t)
candidate_t = OutputHead(Z_(t+1))
```

The loop may issue new sparse memory queries, each subject to the same top-`K` contract. Before the first update, a task policy or learned complexity head requests an integer budget `T_request ∈ [1, T_max]`. The trusted controller computes

`T = min(T_request, T_max)`,

freezes `T`, and executes exactly `T` updates. No proposal, critic, candidate score, confidence estimate, convergence estimate, or memory result can change that counter after the loop begins. A bad complexity estimate can allocate too little useful computation, but it cannot prevent termination or exceed `T_max`.

Alpha does not require a verifier for halting. An optional untrusted assessor may score completed candidates, but it has no control edge into the step counter. The Lean declarations `trustedStepCount_le_cap` and `advisoryRun_steps_independent` formalize the cap and this noninterference boundary. They do not prove that the requested budget is adequate or that a candidate is correct.

This design follows the evidence that recurrent latent depth can scale test-time reasoning without emitting a token for every internal step ([recurrent-depth reasoning](https://arxiv.org/abs/2502.05171)). Alpha adds an explicit workspace, external-memory access, criticism, and verifier-independent bounded computation.

## 5. Alpha-125M reference configuration

The first implementation should use the following shape. It is a research baseline, not an asserted optimum.

| Component | Initial value |
| --- | ---: |
| Model width `d` | 768 |
| Stream blocks `L` | 12 |
| Feed-forward width | 1792 |
| Local-attention width | 256: 8 heads × 32 |
| Local window `W` | 512 |
| Phase-state width | 512: 8 heads × 64 real channels |
| Phase channels per head | 64 |
| Minimum damping `a_min` | 0.5 |
| Episodic-memory blocks | 4, 8, 12 |
| Router key/query width | 256 |
| Sparse records per query `K` | 64 |
| Workspace slots `R` | 64 |
| Maximum reasoning steps `T_max` | 8 |
| Vocabulary | 32,000 |
| Training context | 2,048 tokens |

Use tied input and output token embeddings, parameter-free rotary positions in the local branch, bias-free matrix projections except where stated below, and no learned absolute-position table. Share the episodic router/interface across memory-enabled blocks and tie the reasoning-cell weights across all reasoning steps.

The parameter contract is:

| Parameter group | Count |
| --- | ---: |
| 32,000 × 768 tied token embedding/output table | 24,576,000 |
| 12 stream blocks × 7,494,424 | 89,933,088 |
| Shared episodic query/key/value/output maps | 1,572,864 |
| One tied reasoning cell | 8,849,664 |
| 64 learned workspace slots | 49,152 |
| 8-way pre-loop complexity head, including bias | 6,152 |
| Final RMSNorm | 768 |
| **Total trainable parameters** | **124,987,688** |

Each stream-block count consists of a 1,985,560-parameter phase branch, 786,432-parameter width-256 local attention, 592,128-parameter gated fusion, 4,128,768-parameter SwiGLU, and two 768-weight RMSNorms. The phase parameter projection emits element-wise `B_t` and `C_t` lanes plus one `a_t`, `θ_t`, and `λ_t` scalar per state head. The tied reasoner consists of full-width self-attention, full-width cross-attention to retrieved records, a width-1792 SwiGLU, and three RMSNorms. Recurrent states and episodic records are runtime state, not trainable parameters.

## 6. Execution and storage complexity

Suppressing projection dimensions, let:

- `n` be context length;
- `W` be local window size;
- `K` be retrieved records per query;
- `M` be the number of episodic-memory blocks;
- `I(n)` be the router/index cost per query;
- `R` be workspace size and `T` reasoning iterations.

The attention pairs actually materialized by the stream stack are bounded by

`L n W + M n K`.

For fixed `L`, `W`, `M`, and `K`, this is linear in `n`. Router construction and lookup are additional. With a suitable hashing or hierarchical implementation, the intended total retrieval cost is near-linear; that systems claim is not proved by Lean.

The reasoning core costs approximately

`O(T(R² + RK))`,

which depends on workspace size rather than full context length.

Inference storage is:

- constant-size recurrent state per stream block;
- `O(LWd)` local KV storage;
- `O(Mnd_mem)` exact archival storage, or `O(MCd_mem)` in bounded mode;
- `O(Rd)` reasoning workspace.

Linear archival storage is the explicit price of retaining arbitrary exact records. It can be tiered across GPU, CPU, and persistent storage while a small active cache remains on GPU. The recent [HOLA](https://arxiv.org/abs/2607.02303) preprint provides early empirical support for pairing a recurrent compressor with a bounded exact cache, but Alpha does not treat those preliminary results as a proof.

## 7. Machine-checked guarantees

All declarations below compile with Lean 4 and mathlib. They are in [`AlphaArchitecture/Theorems.lean`](AlphaArchitecture/Theorems.lean).

| Lean declaration | What it establishes | Required assumptions |
| --- | --- | --- |
| `AffineStep.compose_assoc` | Affine state transitions compose associatively. | Semiring/module algebra. |
| `AffineStep.summarize_correct` | A composed scan summary equals sequential execution exactly. | Exact algebra; not floating-point arithmetic. |
| `phaseEnergy_rotate` | A two-dimensional phase rotation preserves `q²+p²`. | Real arithmetic. |
| `phaseEnergy_damped` | Damping scales phase energy by `a²`. | Real arithmetic. |
| `phaseEnergy_damped_le` | An unforced phase pair is energy non-increasing. | `0 ≤ a ≤ 1`. |
| `portPower_le_supply` | A lossless interconnection plus nonnegative dissipation is passive relative to external supply. | Zero internal power from `J`; nonnegative power in `R`. |
| `norm_iterateState_le` | Initial-state influence is bounded by `ρⁿ‖x₀‖`. | The state transition is `ρ`-contractive and `ρ ≥ 0`. |
| `norm_drivenState_le` | A driven contractive state obeys a recursive norm envelope. | Contractive transition and bounded inputs. |
| `inputEnvelope_le` | The accumulated input envelope is at most `U/(1-ρ)`. | `0 ≤ ρ < 1` and `U ≥ 0`. |
| `norm_drivenState_le_geometric` | Combined fading-initial-state and bounded-input estimate. | Same conditions as above. |
| `memoryWrite_read_same` | Reading a key immediately after writing returns the exact value. | Decidable key equality. |
| `hybridRead_exact_hit` | An exact-memory hit overrides the lossy fallback. | The queried record is present. |
| `exact_recall_requires_state_capacity` | Lossless recall of every length-`n` history requires at least `|V|ⁿ` distinguishable states. | Finite token and state spaces. |
| `no_exact_recall_of_small_state` | A smaller fixed state cannot encode and decode every history exactly. | Finite token and state spaces. |
| `selectedPairs_le` | At most `N K` sparse query-key pairs are materialized for `N` queries. | Every query selects at most `K` keys. |
| `selectedPairs_le_dense` | Sparse pair count is no greater than dense `n²` when `K ≤ n`. | Per-query selection cap. |
| `trustedStepCount_le_cap` | The reasoning controller never executes more than its immutable hard cap. | None beyond natural-number ordering. |
| `trustedStepCount_eq_requested` | A requested budget inside the cap is executed exactly. | `requested ≤ hardCap`. |
| `advisoryRun_steps_independent` | Replacing an arbitrary verifier/advisory cannot change the executed step count. | None; the advisory may have any type or value. |

Two consequences matter most:

1. **Stability and memory are different properties.** Contraction provides a useful state bound, but it also makes old-state influence decay exponentially. The external exact store is therefore architectural, not optional decoration.
2. **Sparse access does not remove storage requirements.** The formal lower bound rules out exact arbitrary recall from a fixed-size state. Alpha reduces query-key interaction count while retaining the records that exact recall needs.

## 8. Claims deliberately left open

The formalization does not prove:

- that Alpha learns language or beats a Transformer;
- that the router retrieves every relevant record;
- that bounded memory avoids forgetting;
- that the proposer or critic is correct, or that the complexity head requests enough steps;
- that more reasoning iterations monotonically improve an answer;
- end-to-end stability of residual, feed-forward, attention, and memory branches;
- numerical equivalence of sequential and parallel scans in finite precision;
- a wall-clock speedup on any particular device.

These are empirical questions for later controlled experiments. The architecture should be evaluated against matched dense-Transformer, pure recurrent, Mamba-3, and sparse-attention baselines. Recent theory and experiments support hybrids on recall and length generalization, but do not establish Alpha's performance in advance ([hybrid expressivity-efficiency analysis](https://arxiv.org/abs/2603.08859)).

## 9. Minimum implementation contract

An implementation counts as Alpha only if it satisfies all of the following:

1. The stream state uses token-conditioned damped phase transitions with `0 < a_t < 1`.
2. Prefill and decoding implement the same affine recurrence.
3. Local attention is causal and bounded by `W`.
4. Long-term retrieval is context-dependent and returns at most `K` exact records per query.
5. The implementation states whether memory is archival or bounded; bounded mode must not claim lossless arbitrary recall.
6. The reasoning loop operates on a bounded workspace rather than rescanning the full context.
7. The requested reasoning budget is clamped and frozen before the loop; no internal verifier, critic, confidence, convergence, or candidate signal may alter it.
8. Recurrent, local, episodic-memory, and reasoning components can each be disabled for ablation.

Everything else—router family, storage backend, MIMO rank, layer schedule, training curriculum, and optional post-run assessment—may evolve without changing the core architecture.

## 10. Training Alpha-125M on 1B FineWeb-Edu tokens

This is the canonical first experiment. It tests whether the components train and contribute; it is not enough data to establish scaling or Transformer superiority. One billion tokens is only about eight tokens per parameter. The [Chinchilla scaling study](https://arxiv.org/abs/2203.15556) is a warning that a 125M model may benefit from substantially more data, so any result from this run must be labeled a pilot.

### 10.1 Data and tokenizer

Use the official [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset with configuration `sample-10BT`. The official card describes that configuration as an approximately 10-billion-GPT-2-token random sample; pin its exact repository revision and record the dataset license and Common Crawl terms with the run. FineWeb-Edu itself was introduced and evaluated in the [FineWeb paper](https://arxiv.org/abs/2406.17557).

1. Split by document before tokenization. Hash the immutable document `id`, or `(dump, url)` if `id` is unavailable, with a recorded seed. Reserve 0.5% of documents for validation and never admit them to tokenizer fitting or training.
2. Train a 32,000-entry byte-level BPE on training documents only. The count includes `<bos>`, `<eod>`, and `<pad>`; byte fallback removes the need for `<unk>`. Save the tokenizer model and its checksum.
3. Shuffle training documents deterministically, insert `<eod>`, and tokenize until exactly 1,000,000,000 non-padding training tokens have been consumed under this tokenizer. The dataset's published GPT-2-token count is not Alpha's token count.
4. Pack to 2,048 tokens with segment IDs. At every document boundary, reset phase states, local KV visibility, position indices, and episodic memory. Mask padding and do not count it toward the token budget.
5. Materialize a fixed 10-million-token validation set from reserved documents. Report loss in nats/token and perplexity using the same tokenizer and reset rules.

The data loader must checkpoint its document order, document offset, packed-token offset, and number of non-padding tokens already consumed. Resuming must reproduce the next batch exactly.

### 10.2 Causal objectives

The main objective is next-token cross-entropy on every non-padding stream position. To train the more expensive reasoning path without constructing a workspace for all 2,048 positions, sample one causal anchor per packed sequence that has at least 512 tokens since the most recent document reset; skip this auxiliary example if none exists. The workspace at that anchor may read only its prefix. Run the tied cell for all depths `1…8` and predict the anchor's next token from the answer slot at every depth.

Use

`L = L_stream + 0.25 L_reason + 0.05 L_route + 0.01 L_budget`.

- `L_stream` is mean causal cross-entropy over all ordinary token targets.
- `L_reason` is the mean anchor cross-entropy over depths `1…8`. Deep supervision prevents later depths from being the only useful ones.
- `L_route` trains the sparse index to reproduce the exact top-`K` dot-product set for that one anchor. Compute a stop-gradient teacher over only the causal prefix, add any missed teacher identifiers to the training candidate set, and minimize multi-label cross-entropy on the student's router logits. This adds `O(n)` teacher scores per sequence rather than an `n²` matrix. Language loss still determines which learned keys become useful.
- `L_budget` trains the pre-loop complexity head. With the eight recorded anchor losses, label the smallest depth whose loss is within `0.05` nats of the best depth, stop gradient through that label, and use ordinary eight-class cross-entropy. Begin this term after 100 million tokens, once depth losses contain a useful signal.

At inference, the complexity head reads `Z_0`, requests a value in `1…8`, and the trusted controller freezes that count. The training-only hindsight label is not an inference-time verifier. For controlled evaluation, override the head and report results at fixed `T = 1, 2, 4, 8` as well.

Use hard top-`K` retrieval in the forward pass and softmax only over the gathered records. During the first 100 million tokens, add seeded exploration noise to router scores before selection and anneal it to zero; otherwise an initially bad hard router may never explore useful records.

### 10.3 Optimizer and schedule

Use AdamW for the first study. A newer or more elaborate optimizer would confound an architecture comparison; AdamW is well understood, and the published GPT-3 125M baseline used a maximum learning rate of `6×10⁻⁴`, a 2,048-token context, and a 0.5M-token batch ([GPT-3, Table 2.1](https://arxiv.org/abs/2005.14165)). Alpha should start more conservatively because of its new recurrent and routing paths:

| Setting | Value |
| --- | ---: |
| Optimizer | AdamW |
| Maximum learning rate | `3×10⁻⁴` |
| Adam betas | `(0.9, 0.95)` |
| Adam epsilon | `1×10⁻⁸` |
| Weight decay | `0.1` |
| Global gradient-norm clip | `1.0` |
| Precision | bfloat16 projections; float32 scan/state reductions, softmax reductions, loss, and optimizer state |
| Dropout | `0.0` |
| Global batch | 131,072 non-padding tokens |
| Warmup | first 200 optimizer steps |
| Decay | cosine to `3×10⁻⁵` at exactly 1B consumed tokens |

Apply weight decay to rank-two weight matrices. Exclude biases, RMSNorm weights, damping/frequency scalars, learned fusion gates, and workspace slots. Divide summed loss by the true number of non-padding targets before gradient accumulation.

With length 2,048, the global batch is 64 sequences. It can be formed with any data-parallel microbatch and accumulation combination satisfying

`devices × sequences_per_device × accumulation_steps = 64`.

For example, eight devices with two sequences per device use four accumulation steps. The run contains 7,629 full optimizer steps plus one final partial step of 51,712 tokens. Drive warmup, decay, logging, and stopping from the cumulative non-padding token count so that the partial step is handled correctly.

### 10.4 Initialization and numerical rules

- Initialize ordinary linear and embedding weights from `Normal(0, 0.02)`. Scale residual-output projections by `1 / sqrt(2L)` and initialize the recurrent-reasoning update gate to `0.1`.
- Initialize phase-head half-lives logarithmically from 16 to 4,096 tokens, using `a = 2^(-1/half_life)` and the inverse of Alpha's bounded damping parameterization. Initialize phase periods logarithmically from 4 to 8,192 tokens with `θ = 2π/period`. Both remain learnable.
- Initialize `λ_t` around `0.5`. Compute trigonometric values, affine scan products, recurrent accumulations, attention softmax normalizers, and state-norm diagnostics in float32 even when surrounding projections use bfloat16.
- Keep phase states and exact-memory records separate per packed segment. Detach neither recurrent state nor selected memory values inside a 2,048-token training segment.
- Use pre-norm residual blocks. Reject a step and restore the last checkpoint on any NaN/Inf; do not silently zero invalid gradients.

### 10.5 Checkpoints, monitoring, and acceptance

Checkpoint every 50 million consumed tokens and at the end. Save model, AdamW state, scheduler state, all random-number-generator states, tokenizer checksum, data cursor, index/hash seeds, and exact token count. Evaluate a fixed 1-million-token validation slice every 25 million tokens and the full 10-million-token validation set at the end.

Log total and component losses, learning rate, gradient norm, tokens/second, peak memory, phase-state RMS/max, the distribution of damping values, router candidate recall against the anchor teacher, selected-record age, memory-gate usage, predicted step-budget histogram, and validation loss at fixed `T = 1, 2, 4, 8`.

Before committing the full budget, the implementation must pass parameter-count, causal-leakage, document-reset, top-`K` cap, checkpoint-resume, and sequential-versus-scan equivalence tests. The final report must include matched ablations with the recurrent, local, episodic, and reasoning paths individually disabled, plus a parameter- and token-matched Transformer baseline. A successful 1B-token run means stable optimization and measurable component utility; it does not by itself validate Alpha's larger performance claim.

### 10.6 Low-budget validation ladder

Do not begin with a billion-parameter comparison. Validate Alpha in progressively more expensive gates and scale only when the previous gate succeeds.

| Stage | Model size | Purpose |
| --- | ---: | --- |
| 1 | 1–5M parameters | Verify recurrence, memory reads, routing, segment resets, and trusted halting. |
| 2 | 10–30M parameters | Test long-range mechanisms on controlled synthetic tasks. |
| 3 | 50–125M parameters | Compare Alpha with matched Transformer, SSM, and simpler hybrid baselines. |
| 4 | 125M on 1B tokens | Run the full architecture pilot only after the earlier gates pass. |

At stages 1–3, keep parameter count, tokenizer, token budget, optimizer, and hardware fixed across these baselines:

- a local-attention Transformer;
- a pure phase-state SSM;
- an SSM plus local attention;
- Alpha without episodic memory;
- Alpha without reasoning recurrence;
- full Alpha.

Use cheap synthetic tasks to isolate the architectural claims: delayed copy and associative recall beyond the local window, multiple-key retrieval with distractors, state tracking, long-range induction, and fixed-budget reasoning. Measure exact recall accuracy, validation loss, memory usage, FLOPs/token, throughput, and performance as context length grows. Then use a 10M–100M-token FineWeb-Edu slice for natural-language validation before committing to the full 1B-token pilot.

The Lean proofs establish structural contracts, not learned performance. A small positive result is evidence that a mechanism is worth scaling, not proof of frontier capability. If Alpha does not beat its matched ablations on recall, loss, or efficiency at small scale, stop rather than assuming a larger model will fix the architecture.
