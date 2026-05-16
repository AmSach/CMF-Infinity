# CMF Infinity Architecture

CMF Infinity is the scalable Continuous Meaning Field architecture family. The goal is to keep the language interface autoregressive while moving the internal computation into an adaptive continuous latent flow.

## Naming

Model names use parameter count in billions:

```text
CMF Infinity <params-in-billions>B
```

Examples:

- `CMF Infinity 0.00037B`: 372k parameters.
- `CMF Infinity 0.203B`: about 203M tensor parameters.
- `CMF Infinity 7B`: future 7B-parameter target.

## Core Stack

```text
tokens
  -> embeddings
  -> dilated causal context encoder
  -> continuous vector field
  -> adaptive latent integration
  -> normalized latent states
  -> token logits
```

The core model components are:

- Dilated CNN context landscape: avoids quadratic attention cost for context construction.
- Vector field: predicts latent movement direction from current state, context, memory anchors, time features, and optional goal vector.
- Adaptive solver: allocates more steps when field curvature is high and fewer steps when the trajectory is stable.
- Deliberative latent refinement: `DeliberativeContinuousMeaningField` performs multiple vector-field refinement passes with a learned halting signal.
- Decoder: maps latent path states back to vocabulary logits.

## Deliberative CMF

The deliberative model is the first in-model reasoning upgrade beyond the original solver loop:

```text
tokens
  -> dilated context landscape
  -> initial latent states
  -> repeated vector-field refinement
  -> learned halt/ponder score
  -> logits
```

The new model class is `cmf.model.DeliberativeContinuousMeaningField`, and the 120M-class preset is `infinity-reasoning-0.12b`.

Important properties:

- It keeps the efficient dilated-CNN context constructor.
- It refines every latent token state across multiple thinking passes.
- It exposes `thinking_steps` and `halt_mean` in forward outputs.
- It can use adaptive inference-time halting when `adaptive_thinking=True`.
- It remains checkpoint/package compatible through `model_type="deliberative_cmf"`.

This is an architecture mechanism, not proof of intelligence by itself. It must be trained long enough for the halt head and refinement passes to learn useful computation.

## Infinity Runtime Layer

The runtime now has a generation-time reasoning scaffold in `cmf/infinity_runtime.py`:

```text
user task
  -> optional evidence retrieval
  -> grounded prompt
  -> repeated candidate generation
  -> verifier scoring by model log probability
  -> evidence/consensus reranking
  -> convergence or budget stop
```

This layer is deliberately external to the neural field core. It lets CMF spend more compute on hard prompts, use local evidence for factual grounding, and keep a traceable record of why a response was selected.

The shipped implementation includes:

- `EvidenceMemory`: dependency-free lexical retrieval over a local text file.
- `DeliberationConfig`: bounded or open-ended thinking policy.
- `deliberative_generate_text`: candidate generation, verifier scoring, evidence alignment, consensus scoring, and convergence stopping.
- CLI exposure in `scripts/cmf_infinity_generate.py` and `scripts/cmf_infinity_chat.py`.

Open-ended thinking is supported only when the caller provides an external stop condition such as `--max-deliberation-seconds`. This keeps the architecture capable of unbounded loops in principle while preventing accidental runaway compute in normal use.

## What “Infinity” Means

`Infinity` is an architectural intent: the model can spend variable thinking steps inside the latent field. Production runs still use hard limits:

- `min_solver_steps`
- `max_solver_steps`
- `curvature_threshold`
- generation-time deliberation budgets or wall-clock stop conditions

This prevents runaway compute and makes benchmarks reproducible. Any paper or README must describe Infinity as adaptive bounded compute, not literal infinite computation.

The adaptive solver now first estimates curvature, selects a step count, and then integrates across exactly one latent-time interval with `dt = 1 / selected_steps`. Earlier prototype behavior could take extra steps with the minimum-step `dt`, which accidentally changed the integration horizon.

## C++/CUDA Plan

The speed-critical target is to move these pieces into compiled code:

1. Precomputed Euler/RK integration: already scaffolded in `cpp/`.
2. Adaptive step controller: Python reference is implemented; C++/CUDA parity is the next native-runtime target.
3. Vector-field fused MLP/gate: CUDA kernel or TorchInductor/AOT target.
4. Dilated context encoder: use fused Conv1d paths first; custom kernels only if profiling proves it is the bottleneck.
5. Inference runner: C++ host loop that owns tokenization, batching, checkpoint loading, and compiled CMF kernels.

Current reality on this host:

- CUDA is visible through PyTorch.
- The C++ extension does not build because MSVC, `nvcc`, and `ninja` are unavailable.
- Python/PyTorch remains the reference path.

## Scaling Path

Large-corpus training should use `scripts/train_large_scale.py`:

```powershell
python scripts\train_large_scale.py --device cuda --tokenizer byte --seq-len 1024 --micro-batch-size 4 --grad-accum 8 --steps 1000
```

For streaming web-scale data:

```powershell
python scripts\train_large_scale.py --device cuda --dataset <hf-dataset-id> --tokenizer gpt2 --seq-len 1024 --amp
```

The script streams text, accumulates gradients, resumes from training checkpoints, and can emit an inference package with config/tokenizer metadata.

## Evaluation Rule

CMF Infinity should only claim superiority when all of these are true:

- Parameters are matched within the configured tolerance.
- Held-out loss is equal or better.
- Reasoning/factuality tasks are equal or better.
- Throughput is equal or better.
- Energy is measured and equal or better.
- The benchmark corpus and prompts are not training-set memorization only.

AGI or frontier-level claims require external, broad, contamination-aware evaluation. No current record satisfies that bar.
