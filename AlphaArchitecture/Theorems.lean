import Mathlib

/-!
# Alpha architecture: machine-checked core

This file formalizes only structural guarantees used by the architecture.
It intentionally makes no claims about learnability, language-model quality,
router accuracy, verifier calibration, or reasoning benchmark performance.
-/

namespace Alpha

open scoped InnerProductSpace

/-! ## Affine scans -/

/-- One time-varying affine state transition `x ↦ A x + b`. -/
structure AffineStep (R V : Type*) [Semiring R] [AddCommMonoid V] [Module R V] where
  linear : V →ₗ[R] V
  bias : V

namespace AffineStep

variable {R V : Type*} [Semiring R] [AddCommMonoid V] [Module R V]

/-- Evaluate an affine transition. -/
def apply (step : AffineStep R V) (x : V) : V :=
  step.linear x + step.bias

/-- The identity affine transition. -/
def identity : AffineStep R V where
  linear := LinearMap.id
  bias := 0

/-- `compose outer inner` applies `inner` first and `outer` second. -/
def compose (outer inner : AffineStep R V) : AffineStep R V where
  linear := outer.linear.comp inner.linear
  bias := outer.linear inner.bias + outer.bias

theorem eq_of_fields {a b : AffineStep R V}
    (linear_eq : a.linear = b.linear) (bias_eq : a.bias = b.bias) : a = b := by
  cases a
  cases b
  simp_all

@[simp] theorem apply_identity (x : V) : apply (identity : AffineStep R V) x = x := by
  simp [apply, identity]

theorem apply_compose (outer inner : AffineStep R V) (x : V) :
    apply (compose outer inner) x = apply outer (apply inner x) := by
  simp [apply, compose, add_assoc]

/-- Affine transition composition is associative, which is the algebraic fact
needed by a tree-based parallel prefix scan. -/
theorem compose_assoc (a b c : AffineStep R V) :
    compose (compose a b) c = compose a (compose b c) := by
  apply eq_of_fields
  · apply LinearMap.ext
    intro x
    rfl
  · simp [compose, add_assoc]

@[simp] theorem identity_compose (a : AffineStep R V) :
    compose identity a = a := by
  apply eq_of_fields
  · apply LinearMap.ext
    intro x
    rfl
  · simp [compose, identity]

@[simp] theorem compose_identity (a : AffineStep R V) :
    compose a identity = a := by
  apply eq_of_fields
  · apply LinearMap.ext
    intro x
    rfl
  · simp [compose, identity]

/-- Sequential evaluation of a list of transitions. -/
def run : List (AffineStep R V) → V → V
  | [], x => x
  | step :: steps, x => run steps (apply step x)

/-- A single affine summary of a list of transitions. -/
def summarize : List (AffineStep R V) → AffineStep R V
  | [] => identity
  | step :: steps => compose (summarize steps) step

/-- The composed scan summary is exactly equivalent to sequential execution. -/
theorem summarize_correct (steps : List (AffineStep R V)) (x : V) :
    apply (summarize steps) x = run steps x := by
  induction steps generalizing x with
  | nil => simp [summarize, run]
  | cons step steps ih =>
      rw [summarize, apply_compose, ih]
      rfl

end AffineStep

/-! ## Damped phase rotations -/

/-- Squared Euclidean energy of one real representation of a complex state. -/
def phaseEnergy (z : ℝ × ℝ) : ℝ :=
  z.1 ^ 2 + z.2 ^ 2

/-- A two-dimensional rotation. Input-dependent angles model a complex-valued
state update without requiring complex tensors. -/
noncomputable def phaseRotate (θ : ℝ) (z : ℝ × ℝ) : ℝ × ℝ :=
  (Real.cos θ * z.1 - Real.sin θ * z.2,
   Real.sin θ * z.1 + Real.cos θ * z.2)

/-- Rotation followed by scalar damping. -/
noncomputable def dampedPhaseRotate (a θ : ℝ) (z : ℝ × ℝ) : ℝ × ℝ :=
  let rotated := phaseRotate θ z
  (a * rotated.1, a * rotated.2)

theorem phaseEnergy_nonneg (z : ℝ × ℝ) : 0 ≤ phaseEnergy z := by
  dsimp [phaseEnergy]
  positivity

/-- Rotations preserve phase energy exactly. -/
theorem phaseEnergy_rotate (θ : ℝ) (z : ℝ × ℝ) :
    phaseEnergy (phaseRotate θ z) = phaseEnergy z := by
  dsimp [phaseEnergy, phaseRotate]
  nlinarith [Real.sin_sq_add_cos_sq θ]

/-- Damping scales phase energy by the square of the damping coefficient. -/
theorem phaseEnergy_damped (a θ : ℝ) (z : ℝ × ℝ) :
    phaseEnergy (dampedPhaseRotate a θ z) = a ^ 2 * phaseEnergy z := by
  calc
    phaseEnergy (dampedPhaseRotate a θ z) =
        a ^ 2 * phaseEnergy (phaseRotate θ z) := by
          simp [phaseEnergy, dampedPhaseRotate]
          ring
    _ = a ^ 2 * phaseEnergy z := by rw [phaseEnergy_rotate]

/-- A damping coefficient in `[0, 1]` makes the unforced phase transition
energy non-increasing. -/
theorem phaseEnergy_damped_le (a θ : ℝ) (z : ℝ × ℝ)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    phaseEnergy (dampedPhaseRotate a θ z) ≤ phaseEnergy z := by
  rw [phaseEnergy_damped]
  have haSq : a ^ 2 ≤ 1 := by nlinarith
  calc
    a ^ 2 * phaseEnergy z ≤ 1 * phaseEnergy z :=
      mul_le_mul_of_nonneg_right haSq (phaseEnergy_nonneg z)
    _ = phaseEnergy z := one_mul _

/-! ## Conditional passivity -/

section Passivity

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Instantaneous power for a lossless interconnection `J`, dissipative map
`R`, state-gradient `x`, and external port input `u`. -/
def portPower (J R : V →ₗ[ℝ] V) (x u : V) : ℝ :=
  ⟪x, J x - R x + u⟫_ℝ

/-- If `J` contributes zero power and `R` contributes nonnegative
dissipation, internal dynamics cannot produce energy. -/
theorem portPower_le_supply (J R : V →ₗ[ℝ] V) (x u : V)
    (hJ : ⟪x, J x⟫_ℝ = 0)
    (hR : 0 ≤ ⟪x, R x⟫_ℝ) :
    portPower J R x u ≤ ⟪x, u⟫_ℝ := by
  simp only [portPower, inner_add_right, inner_sub_right, hJ]
  linarith

end Passivity

/-! ## Fading-state and driven-state bounds -/

section Stability

variable {V : Type*} [NormedAddCommGroup V]

/-- Repeated application of a state transition. -/
def iterateState (step : V → V) : ℕ → V → V
  | 0, x => x
  | n + 1, x => step (iterateState step n x)

/-- A contractive state transition exponentially suppresses its initial state.
This is both a stability guarantee and a precise statement of fading memory. -/
theorem norm_iterateState_le (step : V → V) (ρ : ℝ) (n : ℕ) (x : V)
    (hρ : 0 ≤ ρ)
    (hstep : ∀ y, ‖step y‖ ≤ ρ * ‖y‖) :
    ‖iterateState step n x‖ ≤ ρ ^ n * ‖x‖ := by
  induction n with
  | zero => simp [iterateState]
  | succ n ih =>
      calc
        ‖iterateState step (n + 1) x‖ = ‖step (iterateState step n x)‖ := by rfl
        _ ≤ ρ * ‖iterateState step n x‖ := hstep _
        _ ≤ ρ * (ρ ^ n * ‖x‖) := mul_le_mul_of_nonneg_left ih hρ
        _ = ρ ^ (n + 1) * ‖x‖ := by ring

/-- Driven state evolution `x_{t+1} = step(x_t) + input_t`. -/
def drivenState (step : V → V) (input : ℕ → V) : ℕ → V → V
  | 0, x => x
  | n + 1, x => step (drivenState step input n x) + input n

/-- Scalar envelope obeying the same contractive recurrence as the norm bound. -/
def inputEnvelope (ρ U : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => ρ * inputEnvelope ρ U n + U

/-- Norm bound for a contractive recurrent state with uniformly bounded input. -/
theorem norm_drivenState_le (step : V → V) (input : ℕ → V)
    (ρ U : ℝ) (n : ℕ) (x : V)
    (hρ : 0 ≤ ρ)
    (hstep : ∀ y, ‖step y‖ ≤ ρ * ‖y‖)
    (hinput : ∀ i, ‖input i‖ ≤ U) :
    ‖drivenState step input n x‖ ≤
      ρ ^ n * ‖x‖ + inputEnvelope ρ U n := by
  induction n with
  | zero => simp [drivenState, inputEnvelope]
  | succ n ih =>
      calc
        ‖drivenState step input (n + 1) x‖ =
            ‖step (drivenState step input n x) + input n‖ := by rfl
        _ ≤ ‖step (drivenState step input n x)‖ + ‖input n‖ := norm_add_le _ _
        _ ≤ ρ * ‖drivenState step input n x‖ + U :=
          add_le_add (hstep _) (hinput n)
        _ ≤ ρ * (ρ ^ n * ‖x‖ + inputEnvelope ρ U n) + U :=
          add_le_add (mul_le_mul_of_nonneg_left ih hρ) (le_refl U)
        _ = ρ ^ (n + 1) * ‖x‖ + inputEnvelope ρ U (n + 1) := by
          simp only [inputEnvelope]
          ring

/-- With `0 ≤ ρ < 1`, the accumulated input contribution is bounded by the
geometric-series ceiling `U / (1 - ρ)`. -/
theorem inputEnvelope_le (ρ U : ℝ) (n : ℕ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hU : 0 ≤ U) :
    inputEnvelope ρ U n ≤ U / (1 - ρ) := by
  induction n with
  | zero =>
      simp only [inputEnvelope]
      exact div_nonneg hU (by linarith)
  | succ n ih =>
      rw [inputEnvelope]
      calc
        ρ * inputEnvelope ρ U n + U ≤ ρ * (U / (1 - ρ)) + U :=
          add_le_add (mul_le_mul_of_nonneg_left ih hρ0) (le_refl U)
        _ = U / (1 - ρ) := by
          field_simp [ne_of_gt (by linarith : 0 < 1 - ρ)]
          ring

/-- Combined bounded-input bounded-state estimate. -/
theorem norm_drivenState_le_geometric (step : V → V) (input : ℕ → V)
    (ρ U : ℝ) (n : ℕ) (x : V)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hU : 0 ≤ U)
    (hstep : ∀ y, ‖step y‖ ≤ ρ * ‖y‖)
    (hinput : ∀ i, ‖input i‖ ≤ U) :
    ‖drivenState step input n x‖ ≤ ρ ^ n * ‖x‖ + U / (1 - ρ) := by
  exact (norm_drivenState_le step input ρ U n x hρ0 hstep hinput).trans
    (add_le_add (le_refl (ρ ^ n * ‖x‖))
      (inputEnvelope_le ρ U n hρ0 hρ1 hU))

end Stability

/-! ## Exact external memory and its unavoidable capacity -/

/-- An abstract exact key-value memory. Implementations may use a hash table,
tree, ANN index plus exact records, or tiered CPU/GPU storage. -/
abbrev ExactMemory (Key Value : Type*) := Key → Option Value

/-- Functional write into exact memory. -/
def memoryWrite {Key Value : Type*} [DecidableEq Key]
    (memory : ExactMemory Key Value) (key : Key) (value : Value) :
    ExactMemory Key Value :=
  fun query => if query = key then some value else memory query

@[simp] theorem memoryWrite_read_same {Key Value : Type*} [DecidableEq Key]
    (memory : ExactMemory Key Value) (key : Key) (value : Value) :
    memoryWrite memory key value key = some value := by
  simp [memoryWrite]

@[simp] theorem memoryWrite_read_other {Key Value : Type*} [DecidableEq Key]
    (memory : ExactMemory Key Value) (key other : Key) (value : Value)
    (h : other ≠ key) :
    memoryWrite memory key value other = memory other := by
  simp [memoryWrite, h]

/-- Exact memory is consulted before the lossy recurrent fallback. -/
def hybridRead {Key Value : Type*}
    (memory : ExactMemory Key Value) (fallback : Key → Value) (key : Key) : Value :=
  (memory key).getD (fallback key)

/-- An exact-memory hit is returned independently of the recurrent fallback. -/
theorem hybridRead_exact_hit {Key Value : Type*}
    (memory : ExactMemory Key Value) (fallback : Key → Value)
    (key : Key) (value : Value) (hit : memory key = some value) :
    hybridRead memory fallback key = value := by
  simp [hybridRead, hit]

/-- Exact recall of every length-`n` token history requires at least as many
distinguishable states as there are histories. -/
theorem exact_recall_requires_state_capacity
    {Token State : Type*} [Fintype Token] [Fintype State] (n : ℕ)
    (encode : (Fin n → Token) → State)
    (decode : State → Fin n → Token)
    (exact : ∀ history, decode (encode history) = history) :
    Fintype.card Token ^ n ≤ Fintype.card State := by
  have injective : Function.Injective encode := by
    intro x y hxy
    calc
      x = decode (encode x) := (exact x).symm
      _ = decode (encode y) := congrArg decode hxy
      _ = y := exact y
  have cardBound := Fintype.card_le_of_injective encode injective
  simpa using cardBound

/-- Consequently, a state space smaller than the history space cannot provide
lossless arbitrary recall. -/
theorem no_exact_recall_of_small_state
    {Token State : Type*} [Fintype Token] [Fintype State] (n : ℕ)
    (small : Fintype.card State < Fintype.card Token ^ n) :
    ¬ ∃ (encode : (Fin n → Token) → State)
        (decode : State → Fin n → Token),
        ∀ history, decode (encode history) = history := by
  rintro ⟨encode, decode, exact⟩
  exact (not_le_of_gt small)
    (exact_recall_requires_state_capacity n encode decode exact)

/-! ## Sparse retrieval budget -/

/-- Total query-key pairs materialized by a sparse selector. -/
def selectedPairs {Query Key : Type*} [Fintype Query]
    (selected : Query → Finset Key) : ℕ :=
  ∑ query, (selected query).card

/-- A per-query selection cap gives a global linear pair bound. The theorem is
agnostic to how the context-dependent selector is implemented. -/
theorem selectedPairs_le {Query Key : Type*} [Fintype Query]
    (selected : Query → Finset Key) (budget : ℕ)
    (bounded : ∀ query, (selected query).card ≤ budget) :
    selectedPairs selected ≤ Fintype.card Query * budget := by
  classical
  unfold selectedPairs
  calc
    (∑ query, (selected query).card) ≤ ∑ _query : Query, budget := by
      exact Finset.sum_le_sum fun query _ => bounded query
    _ = Fintype.card Query * budget := by simp

/-- For `n` sequence positions and fixed top-`k`, at most `n*k` attention
pairs are materialized. -/
theorem selectedPairs_fin_le {Key : Type*}
    (n budget : ℕ) (selected : Fin n → Finset Key)
    (bounded : ∀ query, (selected query).card ≤ budget) :
    selectedPairs selected ≤ n * budget := by
  simpa using selectedPairs_le selected budget bounded

/-- When `k ≤ n`, sparse pair materialization is no larger than dense
`n × n` materialization. -/
theorem selectedPairs_le_dense (n budget : ℕ)
    (selected : Fin n → Finset (Fin n))
    (bounded : ∀ query, (selected query).card ≤ budget)
    (budget_le : budget ≤ n) :
    selectedPairs selected ≤ n * n := by
  exact (selectedPairs_fin_le n budget selected bounded).trans
    (Nat.mul_le_mul_left n budget_le)

/-! ## Verifier-isolated bounded reasoning -/

/-- The trusted controller executes a requested number of reasoning steps,
clamped to an immutable hard cap. No verifier output is an input. -/
def trustedStepCount (requested hardCap : ℕ) : ℕ :=
  min requested hardCap

/-- The trusted controller can never execute more than its hard cap. -/
theorem trustedStepCount_le_cap (requested hardCap : ℕ) :
    trustedStepCount requested hardCap ≤ hardCap := by
  exact min_le_right requested hardCap

/-- A request already inside the cap is executed exactly. -/
theorem trustedStepCount_eq_requested (requested hardCap : ℕ)
    (withinCap : requested ≤ hardCap) :
    trustedStepCount requested hardCap = requested := by
  exact min_eq_left withinCap

/-- Metadata for a run in which an untrusted advisory may select or score an
answer, but the trusted controller owns the step count. -/
structure AdvisoryRun (Candidate : Type*) where
  steps : ℕ
  selected : Candidate

/-- Attach an arbitrary advisory result to a run without giving it a control
path into the halting schedule. -/
def advisoryRun {Verifier Candidate : Type*}
    (requested hardCap : ℕ) (select : Verifier → Candidate)
    (verifier : Verifier) : AdvisoryRun Candidate where
  steps := trustedStepCount requested hardCap
  selected := select verifier

/-- Changing an arbitrary verifier can change the selected candidate, but it
cannot change when the trusted controller halts. -/
theorem advisoryRun_steps_independent {Verifier Candidate : Type*}
    (requested hardCap : ℕ) (select : Verifier → Candidate)
    (verifier₁ verifier₂ : Verifier) :
    (advisoryRun requested hardCap select verifier₁).steps =
      (advisoryRun requested hardCap select verifier₂).steps := by
  rfl

end Alpha
