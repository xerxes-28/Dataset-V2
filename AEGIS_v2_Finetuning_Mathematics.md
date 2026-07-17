

## HOW TO READ THIS DOCUMENT

This is not a tutorial. This is a **mental model system**.
Read it the way you'd read a circuit schematic — not line by line,
but layer by layer. First understand the shape. Then the details.
Then the math. Then the intuition behind the math.

Each section answers exactly three questions:
  1. WHAT is this?
  2. WHY does it exist?
  3. HOW do you choose it?

---

## PART 0 — THE BIG PICTURE FIRST

Before any math. You need to see the whole system.

```
PRE-TRAINING                    FINE-TUNING
────────────                    ────────────
Giant model learns              You redirect that
to predict next token           knowledge to your domain
from internet-scale data

  "I know language              "Now apply that to
   and general facts"            embedded systems only"

  Costs $50M+                   Costs $30-100
  Takes months                  Takes hours-days
  Anthropic does this           YOU do this
```

Fine-tuning is not teaching from scratch.
It's steering a ship that's already moving.
The ship (base model) already knows physics, math, code, language.
You're just changing its destination.

```
WHAT CHANGES DURING FINE-TUNING:

  Base Model Weights (14B params)
  ┌─────────────────────────────┐
  │  W₁  W₂  W₃  ... W_14B    │  ← These encode all general knowledge
  └─────────────────────────────┘
           │
           │ QLoRA adds small matrices ON TOP
           │
  ┌─────────────────────────────┐
  │  W₁  W₂  W₃  ... W_14B    │  ← Original: FROZEN (unchanged)
  │  +A₁B₁ +A₂B₂ ...          │  ← LoRA adapters: TRAINABLE
  └─────────────────────────────┘

You train the A and B matrices.
NOT the original weights.
That's the whole trick.
```

---

## PART 1 — THE MATHEMATICS OF NEURAL NETWORKS

### 1.1 What a weight actually is

A neural network is a function:
```
  f(x) = output
```
Where x is your input text (as numbers) and output is the next token prediction.

Internally it's just matrix multiplication + nonlinear functions:

```
  Layer 1:  h₁ = σ(W₁x + b₁)
  Layer 2:  h₂ = σ(W₂h₁ + b₂)
  ...
  Layer N:  output = softmax(W_N h_{N-1} + b_N)
```

Where:
  W = weight matrix (what gets trained)
  b = bias vector (also trained)
  σ = activation function (adds nonlinearity)
  softmax = converts raw scores to probabilities over vocabulary

For a 14B parameter model:
  14,000,000,000 individual W and b values
  All these together = the model's "knowledge"

### 1.2 What training actually does

Training minimises a LOSS FUNCTION.

For language models the loss is **Cross-Entropy Loss**:

```
  L = -Σ y_true × log(y_pred)
```

In plain language:
  y_true = the actual next token (1 for correct, 0 for wrong)
  y_pred = what the model predicted (probability 0 to 1)
  log    = punishes confident wrong answers more

```
EXAMPLE:
  Correct next token: "timeout"
  Model says:
    "timeout"  → probability 0.70  → loss = -log(0.70) = 0.357
    "timeout"  → probability 0.10  → loss = -log(0.10) = 2.303
    "timeout"  → probability 0.01  → loss = -log(0.01) = 4.605

  Lower probability for correct answer = higher loss
  Training pushes loss DOWN → model becomes more confident
  about correct answers
```

### 1.3 How weights update — Gradient Descent

After computing loss, we compute gradients:

```
  ∂L/∂W = "how much does loss change if I nudge W slightly?"
```

Then update:
```
  W_new = W_old - η × ∂L/∂W
```

Where η = learning rate (how big the step is)

```
VISUAL:

  Loss
   │  ╲
   │   ╲        Current position ●
   │    ╲      ↙ gradient points uphill
   │     ╲    ↙
   │      ╲  ↙
   │       ╲↙
   │        ● ← Minimum (what we want)
   └──────────────── Weights

  We always step in the direction that reduces loss.
  Learning rate = how big each step is.
```

---

## PART 2 — LORA: THE CORE MATHEMATICAL TRICK

### 2.1 Why not just train all 14B parameters?

```
  14B parameters × 4 bytes each = 56 GB just to STORE the model
  Training requires:
    Model weights:    56 GB
    Gradients:        56 GB (one per parameter)
    Optimizer states: 112 GB (Adam needs 2× gradients)
    Activations:      20-50 GB (intermediate computations)
    ──────────────────────────
    TOTAL:            ~280 GB

  A100 40GB has 40GB VRAM.
  Full fine-tuning is IMPOSSIBLE on consumer hardware.
```

### 2.2 The LoRA insight

Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

Key observation:
```
  When a model adapts to a new task, the change in weights
  has LOW INTRINSIC RANK.

  Meaning: the update matrix ΔW can be approximated by
  the product of two small matrices.
```

Mathematically:
```
  Normal fine-tuning:
    W_new = W_old + ΔW
    ΔW is shape [d_out × d_in]  e.g. [4096 × 4096] = 16M params

  LoRA approximation:
    ΔW ≈ B × A
    A is shape [r × d_in]       e.g. [64 × 4096]  = 262K params
    B is shape [d_out × r]      e.g. [4096 × 64]  = 262K params
    r = rank (small number like 16, 32, 64)

  Parameters saved:
    Full:  16,000,000 parameters per layer
    LoRA:  524,288 parameters per layer
    Ratio: 30× fewer parameters to train!
```

### 2.3 Forward pass with LoRA

During inference:
```
  output = W_old × x  +  (B × A) × x × (α/r)

  Where:
    W_old × x      = original model's contribution (frozen)
    B × A × x      = LoRA adapter's contribution (trained)
    α/r            = scaling factor
```

Visual:
```
  Input x
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
  [W_old]                        [A]
  (frozen)                    [r × d_in]
    │                              │
    │                              ▼
    │                            [B]
    │                         [d_out × r]
    │                              │
    ▼                              ▼ × (α/r)
  output_base    +          output_lora
    │                              │
    └──────────────┬───────────────┘
                   ▼
              final output
```

### 2.4 QLoRA — Quantisation + LoRA

QLoRA adds one more trick: **quantise the frozen base weights**.

```
  Normal float32: 4 bytes per parameter
  float16/bfloat16: 2 bytes per parameter
  int8: 1 byte per parameter
  int4 (QLoRA): 0.5 bytes per parameter

  14B model in float32: 56 GB
  14B model in int4:     7 GB  ← fits on A100 40GB easily!

  QLoRA = frozen base in int4 + LoRA adapters in bfloat16
```

Memory calculation for your setup:
```
  DeepSeek-R1-Distill-14B:
    Base model (int4):       14B × 0.5 bytes = 7 GB
    LoRA adapters (bf16):    ~0.5 GB
    Gradients (bf16):        ~1 GB
    Optimizer states (8bit): ~2 GB
    Activations:             ~8 GB
    ──────────────────────
    TOTAL:                   ~18.5 GB  ✓ fits on A100 40GB
```

---

## PART 3 — EVERY HYPERPARAMETER, THE MATH, AND HOW TO CHOOSE

### 3.1 LEARNING RATE (η) — The Most Critical Parameter

**Definition:** How large a step we take in weight space per gradient update.

**The math:**
```
  W_new = W_old - η × ∂L/∂W

  If η = 2e-4:
    W_new = W_old - 0.0002 × gradient
```

**The landscape:**
```
  Loss
   │
   │  ╲         ╱
   │   ╲       ╱
   │    ╲     ╱
   │     ╲   ╱
   │      ╲ ╱
   │       V  ← minimum
   └────────────

  η too HIGH:
    Steps are huge → overshoots minimum → loss oscillates or explodes
    
  η too LOW:
    Steps tiny → takes forever → gets stuck in local minimum
    
  η JUST RIGHT:
    Descends smoothly → reaches minimum efficiently
```

**How to choose:**
```
  BASE RULE:  η_LoRA = η_full × √(r/d)
  
  For full fine-tune of 14B: η ≈ 1e-5 to 5e-5
  For QLoRA (rank 64, d=4096): multiply by √(64/4096) = √(0.015) ≈ 0.125
  
  So: η_LoRA ≈ 1e-5 × (1/0.125) = 8e-5 ... but practitioners use 2e-4
  
  Practical ranges:
    SFT QLoRA:   1e-4  to  3e-4   (use 2e-4)
    DPO:         5e-6  to  1e-4   (use 5e-5)
    Full FT:     1e-5  to  5e-5   (not for you)
```

**Warning signs:**
```
  Loss goes UP  → η too high, lower it by 5×
  Loss barely moves → η too low, raise it by 3×
  Loss spikes then recovers → η slightly high, lower by 2×
  Loss is smooth and descending → perfect
```

---

### 3.2 LEARNING RATE SCHEDULER

**Definition:** A function that changes η over the course of training.

**Why?** Starting and ending with the same step size is suboptimal:
- Beginning: weights are far from optimum → big steps okay
- End: weights are near optimum → need tiny steps to land precisely

**Cosine Annealing (what you use):**

```
  η(t) = η_min + 0.5 × (η_max - η_min) × (1 + cos(πt/T))

  Where:
    t = current step
    T = total steps
    η_min = 0 (usually)
    η_max = your learning rate (2e-4)

  Looks like this:
  
  η
  2e-4 │╲
       │  ╲
       │    ╲
       │      ╲
       │        ╲___
  0    └──────────────── training steps
       start         end
```

**Warmup:**
```
  Instead of starting at η_max immediately,
  ramp up linearly for first warmup_ratio × total_steps:
  
  η
  2e-4 │    ╭──╮
       │   ╱    ╲
       │  ╱      ╲
       │ ╱        ╲___
  0    └──────────────── steps
       ←warmup→
  
  warmup_ratio = 0.03 means first 3% of steps are warmup
  For 10,000 total steps: warmup = first 300 steps
  
  WHY WARMUP?
  At step 0, weights are random-ish for LoRA (A=random, B=zero)
  High LR immediately = unstable updates
  Warmup gives model time to orient before big steps
```

---

### 3.3 LORA RANK (r)

**Definition:** The inner dimension of the LoRA matrices A and B.

**The math:**
```
  A: [r × d_in]
  B: [d_out × r]
  
  ΔW = B × A: [d_out × d_in]
  
  Parameters per layer = r × d_in + d_out × r = r(d_in + d_out)
  
  For d_in = d_out = 4096 (typical transformer layer):
    r=8:   8 × 8192  = 65,536 params
    r=16:  16 × 8192 = 131,072 params
    r=32:  32 × 8192 = 262,144 params
    r=64:  64 × 8192 = 524,288 params  ← AEGIS v2
    r=128: 128 × 8192 = 1,048,576 params
```

**What rank controls:**
```
  RANK = capacity to change

  Low rank (r=8):
    Few parameters → can only make small adjustments
    Good for: style transfer, minor behavior changes
    Risk: underfitting (model can't learn domain well enough)

  High rank (r=64-128):
    Many parameters → can make large adjustments
    Good for: domain specialisation (your case)
    Risk: overfitting, catastrophic forgetting (if LR too high)

  For AEGIS v2 (heavily domain-specific, 480K samples):
    r=64 is correct
    r=32 would work but slower domain capture
    r=128 risks forgetting base knowledge
```

**How to choose:**
```
  General rule:
    Small dataset (<10K), light task:   r = 8-16
    Medium dataset, moderate task:      r = 32
    Large dataset, heavy domain shift:  r = 64   ← YOU
    Massive dataset, full specialise:   r = 128
```

---

### 3.4 LORA ALPHA (α)

**Definition:** Scaling factor applied to LoRA output.

**The math:**
```
  output = W_old × x + (α/r) × B × A × x

  The (α/r) term scales how much influence the adapter has.
  
  If α = r:       scaling = 1.0  (neutral)
  If α = 2r:      scaling = 2.0  (adapter has double influence)
  If α = r/2:     scaling = 0.5  (adapter has half influence)
```

**What alpha controls:**
```
  Think of it as: "how loudly does LoRA speak
                   relative to the original model?"

  α/r ratio:
    = 1.0  →  balanced
    > 1.0  →  LoRA dominates (faster learning, more forgetting risk)
    < 1.0  →  base model dominates (slower, safer)
```

**How to choose:**
```
  Common pattern: α = 2r
  
  For AEGIS v2: r=64, α=128
    Effective scale = 128/64 = 2.0
    
  This means LoRA updates are 2× amplified.
  Works because your domain is very different from base training.
  
  Conservative alternative: α = r (α=64)
  Aggressive alternative:   α = 4r (α=256, risk of forgetting)
  
  Rule: if loss is unstable → lower α
        if model not learning domain → raise α
```

---

### 3.5 BATCH SIZE

**Definition:** How many training samples are processed before updating weights.

**The math:**
```
  Gradient for one sample:
    g_i = ∂L_i/∂W

  Gradient for batch of N samples:
    g_batch = (1/N) × Σ g_i    (average over batch)

  Weight update:
    W_new = W_old - η × g_batch
```

**Why batch size matters:**
```
  Batch size = 1 (stochastic):
    Very noisy gradient → erratic updates → unstable training
    But: fast iteration, sometimes escapes local minima

  Batch size = full dataset:
    Perfect gradient → stable → but needs entire dataset in memory
    Impossible for large models

  Batch size = 16-32 (mini-batch):
    Good estimate of true gradient + regularisation effect
    Fits in memory → this is what everyone uses

  For AEGIS v2 on A100 (40GB):
    per_device_train_batch_size = 2
    (Limited by GPU memory at seq_len 8192)
```

**Effective batch size:**
```
  Effective batch = per_device_batch × gradient_accumulation_steps × num_GPUs

  AEGIS v2:
    per_device_batch:          2
    gradient_accumulation:     8
    num_GPUs:                  1
    ───────────────────────────
    Effective batch:           16

  What gradient accumulation does:
    Instead of updating weights every step,
    accumulate gradients for 8 steps, THEN update.
    
    Step 1: compute g₁, accumulate
    Step 2: compute g₂, accumulate
    ...
    Step 8: compute g₈, accumulate → update weights with (g₁+...+g₈)/8
    
    Effect: same as batch_size=16, but only needs memory for batch=2
```

---

### 3.6 GRADIENT ACCUMULATION STEPS

**Definition:** Number of forward passes before one weight update.

**Why:**
```
  Larger effective batch = more stable training
  But more batch = more memory
  
  Gradient accumulation lets you fake a large batch
  without the memory cost.
  
  Memory used = per_device_batch_size (small)
  Training stability = effective_batch_size (large)
  Time cost = slightly slower (but acceptable)
```

**How to choose:**
```
  Target effective batch size: 16-32 for LLM fine-tuning
  
  Formula:
    grad_accum = target_effective_batch / (per_device_batch × num_gpus)
  
  AEGIS v2:
    grad_accum = 16 / (2 × 1) = 8  ✓
  
  If you had 2 GPUs:
    grad_accum = 16 / (2 × 2) = 4
```

---

### 3.7 NUMBER OF EPOCHS

**Definition:** How many times the model sees the entire dataset.

**The math:**
```
  total_steps = (dataset_size / effective_batch_size) × num_epochs
  
  AEGIS v2:
    dataset_size:        480,000 samples
    effective_batch:     16
    num_epochs:          2
    
    total_steps = (480,000 / 16) × 2 = 60,000 steps
```

**How to choose:**
```
  Too few epochs:
    Model hasn't seen enough data → underfitting
    Loss still going down at end → should train more

  Too many epochs:
    Model memorises training data → overfitting
    Loses ability to generalise → eval loss goes UP while train loss goes DOWN

  For AEGIS v2 (480K samples):
    2 epochs = reasonable
    
    Why not 3? At 480K samples, 2 passes = 960K gradient updates.
    That's sufficient for the domain shift we want.
    3 epochs risks overfitting to our specific sample phrasings.
    
  Signal to train more: eval loss still decreasing at final checkpoint
  Signal to stop early: eval loss starts increasing (overfitting)
```

---

### 3.8 WEIGHT DECAY

**Definition:** L2 regularisation — penalises large weights.

**The math:**
```
  Without weight decay:
    W_new = W_old - η × ∂L/∂W

  With weight decay (λ=0.01):
    W_new = W_old - η × (∂L/∂W + λ × W_old)
            = W_old × (1 - η×λ) - η × ∂L/∂W
    
  The (1 - η×λ) term shrinks weights toward zero each step.
  
  AEGIS v2:
    η = 2e-4, λ = 0.01
    Shrink factor per step = 1 - (2e-4 × 0.01) = 1 - 2e-6 = 0.999998
    Very mild shrinkage — prevents weights from growing unboundedly
```

**Why it helps:**
```
  Large weights = model is very confident about specific patterns
  Weight decay = prevents overconfidence → better generalisation
  λ = 0.01 is standard → mild regularisation, rarely needs changing
```

---

### 3.9 SEQUENCE LENGTH

**Definition:** Maximum number of tokens the model processes at once.

**Memory cost:**
```
  Memory for attention ∝ sequence_length²
  
  seq_len = 2048: attention matrix = 2048² = 4M elements
  seq_len = 4096: attention matrix = 4096² = 16M elements  (4× more)
  seq_len = 8192: attention matrix = 8192² = 64M elements  (16× more)
  
  This is WHY long context = expensive.
```

**How to choose:**
```
  Rule: set to length that covers 95th percentile of your samples
  
  For AEGIS v2:
    Short embedded Q&A:          ~512 tokens
    Code generation:             ~1024-2048 tokens
    Multi-turn debug sessions:   ~3000-5000 tokens
    <think> + long answer:       ~4000-6000 tokens
    
    8192 covers everything safely ✓
    
  Tradeoff: longer = more memory = smaller batch size
```

---

### 3.10 OPTIMIZER — AdamW 8-bit

**Definition:** Algorithm that determines HOW gradients update weights.

**Standard Adam math:**
```
  At each step t, for each parameter w:
  
  1. Compute gradient:      g_t = ∂L/∂w
  
  2. Update first moment (mean of gradients):
     m_t = β₁ × m_{t-1} + (1-β₁) × g_t
     
  3. Update second moment (variance of gradients):
     v_t = β₂ × v_{t-1} + (1-β₂) × g_t²
     
  4. Bias correction:
     m̂_t = m_t / (1 - β₁ᵗ)
     v̂_t = v_t / (1 - β₂ᵗ)
     
  5. Update weight:
     w_t = w_{t-1} - η × m̂_t / (√v̂_t + ε)
  
  Where:
    β₁ = 0.9    (how much to remember past gradients)
    β₂ = 0.999  (how much to remember past gradient variance)
    ε  = 1e-8   (prevents division by zero)
```

**Why Adam over SGD:**
```
  SGD: same η for every parameter
  Adam: automatically adjusts η per parameter
  
  Parameters that rarely update → get larger effective η
  Parameters that update constantly → get smaller effective η
  
  Result: faster convergence, especially for sparse data
```

**8-bit Adam:**
```
  Normal Adam: stores m_t and v_t in float32 (4 bytes each)
  8-bit Adam:  quantises m_t and v_t to int8 (1 byte each)
  
  Memory saved per parameter: (4+4) - (1+1) = 6 bytes
  
  For 14B model optimizer states:
    Normal: 14B × 8 bytes = 112 GB
    8-bit:  14B × 2 bytes = 28 GB  ← 4× smaller!
    
  With LoRA only training small subset:
    ~0.1B params trained
    8-bit optimizer: 0.1B × 2 bytes = 0.2 GB  (negligible)
```

---

### 3.11 PRECISION — bfloat16

**Definition:** Number format used for computations during training.

```
  float32:  1 sign + 8 exponent + 23 mantissa = 32 bits
  float16:  1 sign + 5 exponent + 10 mantissa = 16 bits
  bfloat16: 1 sign + 8 exponent + 7 mantissa  = 16 bits

  Why bfloat16 over float16?
    bfloat16 has the same RANGE as float32 (8 exponent bits)
    but half the PRECISION (7 vs 23 mantissa bits)
    
    float16 has limited range → overflow/underflow common in training
    bfloat16 has float32 range → numerically stable
    
    A100 GPUs are optimised for bfloat16 → faster than float32
```

---

## PART 4 — CHOOSING THE BEST BASE MODEL

### 4.1 The Decision Framework

```
                        START HERE
                            │
              ┌─────────────▼─────────────┐
              │ Does the task need         │
              │ step-by-step reasoning?    │
              └─────────────┬─────────────┘
                   │               │
                  YES              NO
                   │               │
         ┌─────────▼──┐    ┌───────▼────────┐
         │ R1-Distill │    │ Qwen2.5-Coder  │
         │ series     │    │ or Llama 3.3   │
         └─────────┬──┘    └───────┬────────┘
                   │               │
         ┌─────────▼──┐    ┌───────▼────────┐
         │ How much   │    │ How much RAM?  │
         │ VRAM?      │    └───────┬────────┘
         └─────────┬──┘           │
                   │          A100 40GB
              A100 40GB            │
                   │         14B or 32B
            14B (r=64)        (your call)
            AEGIS v2 ✓
```

### 4.2 Why DeepSeek-R1-Distill-Qwen-14B specifically

```
CRITERION                R1-Distill-14B    Qwen2.5-Coder-14B    Llama-3.3-8B
─────────────────────────────────────────────────────────────────────────────
Native <think> traces    ✅ YES            ❌ No                ❌ No
Reasoning depth          ●●●●●             ●●●●                 ●●●
Code quality             ●●●●●             ●●●●●                ●●●●
Fits A100 40GB (QLoRA)   ✅ YES            ✅ YES               ✅ YES (full)
Architecture stability   ✅ Qwen2.5 base   ✅ Qwen2.5 base      ✅ Llama
Embedded code training   ●●●●●             ●●●●●                ●●●●
Research paper value     HIGH (newer)      MEDIUM               MEDIUM
─────────────────────────────────────────────────────────────────────────────
VERDICT                  ✅ BEST           2nd choice           Backup
```

**The core reason:**
```
  R1-Distill was trained on outputs from DeepSeek-R1 (671B).
  It has internalised the pattern:
  
    <think>
    Let me reason through this hardware problem...
    STM32 I2C timeout usually means...
    </think>
    
    The answer is...
  
  You are NOT teaching it how to reason.
  You are teaching it WHAT to reason about.
  That's 10× easier than teaching reasoning from scratch.
```

---

## PART 5 — SFT vs DPO: THE TWO TRAINING PHASES

### 5.1 SFT — Supervised Fine-Tuning

**What it is:** Show the model (question, correct answer) pairs. Train it to predict the answer.

**The math:**
```
  For each training sample (x, y):
    x = input tokens (system + user message)
    y = output tokens (assistant response)
  
  Loss = -Σ log P(y_t | y_<t, x)
  
  Translation: for each token y_t in the answer,
  maximise the log-probability of predicting it correctly
  given all previous tokens.
  
  This is just next-token prediction on your domain data.
```

**What SFT produces:**
```
  "A model that knows your domain and can answer correctly"
  
  But: it doesn't know WHICH of multiple correct answers is BETTER.
  Two answers might both be technically correct but one is:
    - Better explained
    - Has safer code
    - Cites sources
    - Handles edge cases
    
  SFT can't distinguish these. DPO can.
```

### 5.2 DPO — Direct Preference Optimization

**What it is:** Show the model (question, good_answer, bad_answer) pairs. Train it to prefer good over bad.

**The math:**
```
  DPO loss:
  
  L_DPO = -E[(x, y_w, y_l)] [log σ(β × log(π_θ(y_w|x)/π_ref(y_w|x))
                                    - β × log(π_θ(y_l|x)/π_ref(y_l|x)))]

  Breaking this down:
    π_θ     = your model being trained
    π_ref   = reference model (SFT checkpoint, frozen)
    y_w     = "winner" / preferred response
    y_l     = "loser" / rejected response
    β       = temperature controlling deviation from reference
    σ       = sigmoid function
    
  In plain language:
    Increase probability of y_w (good answer)
    Decrease probability of y_l (bad answer)
    BUT: don't deviate too far from the reference model (controlled by β)
    
  β = 0.1 means:
    Don't change MORE than 10% divergence from reference
    Prevents the model from collapsing to degenerate outputs
```

**DPO learning rate (5e-5) is lower than SFT (2e-4) because:**
```
  DPO is a fine adjustment, not a domain shift.
  At DPO stage, model already knows the domain.
  We're only polishing response quality.
  Large steps would undo SFT learning.
  
  5e-5 = careful, precise adjustment
  2e-4 would be like using a sledgehammer to tune a screw
```

---

## PART 6 — WORKED EXAMPLE WITH REAL NUMBERS

### Full calculation for AEGIS v2

**Given:**
```
  Model:          DeepSeek-R1-Distill-Qwen-14B
  Dataset:        480,000 samples
  Hardware:       A100 40GB
  Goal:           Domain specialisation in embedded/drone/robotics
```

**Step 1: Choose rank**
```
  Large domain shift, 480K samples → r = 64
```

**Step 2: Choose alpha**
```
  α = 2r = 128
  Effective LoRA scale = α/r = 128/64 = 2.0
```

**Step 3: Calculate trainable parameters**
```
  Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  
  For 14B model (40 transformer layers, d_model=5120):
    Attention: d_in=d_out=5120 for q,k,v,o
    FFN:       gate/up have d_in=5120, d_out=13824
               down has d_in=13824, d_out=5120
    
    Per attention layer:
      4 matrices × 2 × r × 5120 = 4 × 2 × 64 × 5120 = 2,621,440
    
    Per FFN layer:
      gate+up: 2 × 64 × (5120+13824) = 2,424,832
      down:    2 × 64 × (13824+5120) = 2,424,832
    
    Per layer total: ~7,471,104
    40 layers:       ~298,844,160
    
    Trainable params: ~300M out of 14B = 2.1% of model
```

**Step 4: Choose learning rate**
```
  QLoRA SFT standard: 2e-4
  
  Verify: not too high for rank 64
  Rule of thumb: η × √r should be < 0.01
  2e-4 × √64 = 2e-4 × 8 = 0.0016 < 0.01 ✓
```

**Step 5: Calculate batch and steps**
```
  per_device_batch:     2  (limited by A100 memory at seq_len 8192)
  grad_accumulation:    8
  effective_batch:      16
  
  steps per epoch = 480,000 / 16 = 30,000
  total steps (2 epochs) = 60,000
  
  warmup steps = 0.03 × 60,000 = 1,800 steps
```

**Step 6: Memory verification**
```
  Base model (int4):        14B × 0.5B = 7.0 GB
  LoRA params (bf16):       300M × 2B  = 0.6 GB
  Gradients (bf16):         300M × 2B  = 0.6 GB
  Optimizer (8-bit Adam):   300M × 2B  = 0.6 GB
  Activations (est):                     8.0 GB
  KV cache (seq 8192):                   4.0 GB
  ─────────────────────────────────────────────
  TOTAL:                                20.8 GB  ✓ (A100 has 40GB)
  Headroom:                             19.2 GB  (comfortable)
```

**Final config:**
```python
LORA_CONFIG = {
    "r":           64,
    "lora_alpha":  128,    # α/r = 2.0
    "lora_dropout": 0.05,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
}

TRAINING_CONFIG = {
    "learning_rate":                    2e-4,
    "lr_scheduler_type":               "cosine",
    "warmup_ratio":                     0.03,
    "per_device_train_batch_size":      2,
    "gradient_accumulation_steps":      8,      # effective batch = 16
    "num_train_epochs":                 2,
    "optim":                           "adamw_8bit",
    "bf16":                             True,
    "weight_decay":                     0.01,
    "max_seq_length":                   8192,
}
```

---

## PART 7 — THE LOSS CURVE: READING TRAINING HEALTH

This is your real-time feedback system on W&B:

```
HEALTHY TRAINING:
  
  Loss
  2.5 │╲
      │  ╲
  1.5 │    ╲──╮
      │        ╲___
  0.8 │             ╲____
  0.5 │                  ╲___
      └─────────────────────── steps
      
  Both train and eval loss decrease smoothly
  Eval loss slightly higher than train (normal)
  Both converge → stop here


OVERFITTING:
  
  Loss
      │    train loss
  1.5 │╲     eval loss
      │  ╲  ╱╲        ← eval goes UP while train goes DOWN
  1.0 │   ╲╱  ╲
  0.8 │         ╲___
      │ ←────────── eval starts rising = STOP HERE
      └─────────────────── steps
      
  Fix: reduce epochs, add dropout, reduce rank


LEARNING RATE TOO HIGH:
  
  Loss
  2.5 │ ╱╲  ╱╲
      │╱  ╲╱  ╲  ← spiky, unstable
  1.5 │        ╱╲
      │
      └─────────────── steps
      
  Fix: lower LR by 5×, increase warmup


LEARNING RATE TOO LOW:
  
  Loss
  2.5 │╲
      │  ╲____________________
  2.0 │                       → barely moving
      │
      └─────────────── steps
      
  Fix: raise LR by 3×
```

---

## PART 8 — TERMINOLOGY MASTER GLOSSARY

```
TERM                DEFINITION
──────────────────────────────────────────────────────────────────────
Parameter           A single trainable number in the model (weight or bias)
Weight              A parameter in a weight matrix W
Gradient            ∂L/∂W — how much loss changes if weight changes by 1
Backpropagation     Algorithm to compute gradients for all parameters
Epoch               One complete pass through the entire training dataset
Step                One weight update (after processing one effective batch)
Loss                Measure of how wrong the model is (lower = better)
Overfitting         Model memorises training data, fails on new data
Underfitting        Model hasn't learned enough (loss still high)
Regularisation      Techniques to prevent overfitting (dropout, weight decay)
Quantisation        Reducing precision of numbers (float32→int4) to save memory
LoRA                Low-Rank Adaptation — efficient fine-tuning technique
QLoRA               LoRA applied to a quantised (4-bit) base model
Rank (r)            Inner dimension of LoRA matrices — controls capacity
Alpha (α)           LoRA scaling factor — controls adapter influence
Adapter             Small trainable module added on top of frozen model
Frozen weights      Base model weights that don't change during fine-tuning
Trainable weights   LoRA adapter weights that DO change during fine-tuning
Batch               Group of samples processed together before weight update
Mini-batch          Small batch (16-32 samples) — standard in practice
Gradient accumulation   Accumulating gradients over multiple steps before updating
Effective batch     per_device_batch × grad_accum × num_gpus
SFT                 Supervised Fine-Tuning — learn from (input, output) pairs
DPO                 Direct Preference Optimization — learn from (good, bad) pairs
RLHF                Reinforcement Learning from Human Feedback (older alternative to DPO)
ChatML              Chat Markup Language — format for conversational training data
JSONL               JSON Lines — one JSON object per line in a text file
Logits              Raw unnormalised scores before softmax
Softmax             Converts logits to probabilities (sums to 1)
Cross-entropy loss  Standard LM loss: -log(P(correct_token))
Perplexity          e^(average_cross_entropy_loss) — model confusion measure
KV cache            Cached key-value pairs for efficient attention computation
Flash Attention     Memory-efficient attention algorithm — required at long seq_len
bfloat16            16-bit float with float32 range — use on A100
Cosine annealing    LR scheduler that follows cosine curve from η_max to 0
Warmup              Gradual LR increase at training start — prevents instability
AdamW               Adam optimiser + weight decay — standard for LLM training
8-bit Adam          Quantised Adam — 4× memory reduction for optimizer states
VRAM                GPU memory (Video RAM) — where model lives during training
Catastrophic forgetting  Model loses general knowledge while learning domain
PEFT                Parameter-Efficient Fine-Tuning — umbrella term (LoRA is one)
Merge               Combining LoRA weights permanently into base model for inference
vLLM                High-throughput inference engine — faster than HuggingFace
Qdrant              Vector database — stores embeddings for RAG
BGE-M3              BAAI General Embedding — embedding model for RAG
BM25                Classic keyword search algorithm (Okapi BM25)
Embedding           Dense vector representation of text (1024 numbers)
Chunk               Small piece of a larger document for RAG ingestion
Context window      Maximum tokens model can process at once (8192 for AEGIS v2)
Hallucination       Model generates confident but incorrect information
RAG                 Retrieval-Augmented Generation — ground answers in real docs
Inference           Running a trained model to generate responses (not training)
Checkpoint          Saved model state at a specific training step
```

---

## PART 9 — THE END PRODUCT

After running through the full pipeline, what do you actually have?

```
WHAT EXISTS AFTER TRAINING:

./aegis_v2_lora/
  ├── adapter_config.json     ← LoRA configuration
  ├── adapter_model.bin       ← The actual trained LoRA weights (~600MB)
  └── tokenizer files

./aegis_v2_merged/
  ├── config.json
  ├── model-00001-of-00006.safetensors
  ├── model-00002-of-00006.safetensors
  ...  (total ~28GB)
  └── tokenizer files         ← Full merged model, ready for vLLM

./aegis_rag_db/
  ├── collection/             ← Qdrant vector database
  └── bm25_corpus.json        ← BM25 keyword index

WHAT THE MERGED MODEL CAN DO:

  Input:  "How do I configure DMA for UART on STM32F4 with HAL?"
  
  Output: <think>
          User wants DMA-based UART. Key considerations:
          1. DMA stream selection (STM32F4 DMA mux fixed)
          2. UART_TX usually DMA2 Stream7 Channel4
          3. Need circular mode for continuous TX
          4. HAL_UART_Transmit_DMA vs LL driver
          Errata: STM32F4 DMA has known issue with burst transfers...
          </think>
          
          Here is complete UART DMA configuration...
          [working, compiler-verified C code]
          [citation: RM0090 p.301, DMA chapter]

  This answer:
  ✓ Reasoned through before responding
  ✓ Retrieved from actual STM32F4 reference manual
  ✓ Code verified by arm-none-eabi-gcc
  ✓ Cited the source page
  ✓ Aware of real errata
  
  Claude cannot do all five of these simultaneously.
  AEGIS v2 can.
```

---

## PART 10 — QUICK DECISION CARD

Photocopy this. Put it next to you when training.

```
┌─────────────────────────────────────────────────────────────┐
│                  AEGIS v2 TRAINING CARD                     │
├─────────────────┬───────────────────────────────────────────┤
│ Loss spiking    │ LR too high → divide by 5                 │
│ Loss not moving │ LR too low → multiply by 3                │
│ Eval loss rises │ Overfitting → reduce epochs or rank       │
│ OOM error       │ Reduce batch size or seq_len              │
│ Training slow   │ Enable packing=True                       │
│ Poor quality    │ Check dataset quality first (always)      │
├─────────────────┼───────────────────────────────────────────┤
│ SFT LR          │ 2e-4                                      │
│ DPO LR          │ 5e-5                                      │
│ Rank            │ 64                                        │
│ Alpha           │ 128                                       │
│ Effective batch │ 16 (2 × 8 accum)                          │
│ Epochs SFT      │ 2                                         │
│ Epochs DPO      │ 1                                         │
│ Scheduler       │ cosine                                    │
│ Warmup          │ 0.03                                      │
│ Optimizer       │ adamw_8bit                                │
│ Precision       │ bfloat16                                  │
│ Seq length      │ 8192                                      │
└─────────────────┴───────────────────────────────────────────┘
```

---
