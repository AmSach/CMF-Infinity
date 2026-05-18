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

## Scaling & Pretraining Path

CMF Infinity models leverage a highly optimized distributed pretraining architecture on consumer and multi-GPU cluster hardware (e.g., dual Tesla T4s, RTX 4090s):

### 1. The High-Density Hybrid AGI Mixture
Rather than training on a single web-scraped dataset, CMF Infinity pretrains on a hyper-dense, mathematically diverse **6-source AGI Mixture Recipe** interleaved round-robin by an asynchronous multithreaded queue mixer:
- **FineWeb-Edu (35% mix)**: Pristine educational web scrape.
- **Cosmopedia v2 (25% mix)**: Synthetic textbook and course streams.
- **Stack-Edu-Dedup (15% mix)**: Deduplicated code repositories.
- **OpenWebMath (10% mix)**: Structured mathematical LaTeX files.
- **Proof-Pile-2 (10% mix)**: Scientific and rigorous proofs.
- **Qwen-Math-CoT (5% mix)**: Chain-of-Thought (think) traces.

### 2. Multi-GPU Distributed Data Parallel (DDP)
To maximize throughput and avoid PCIe interconnect bottle-necks common to layer-by-layer sharding (FSDP), CMF Infinity utilizes **PyTorch Distributed Data Parallel (DDP)** combined with:
- **High Tensor-Core Saturation**: Large micro-batch size (`32` sequences of length `512` per GPU).
- **Asynchronous Background Preloading**: A dedicated background CPU thread inside `cached_lm_batches_from_shards` fetches and loads the next token shard in RAM while the GPUs are actively computing forward/backward passes.
- **Dynamic Checkpoint Restoring**: Auto-stripping of DDP (`module.`) and compiler (`_orig_mod.`) prefixes enables seamless zero-downtime resuming.

### 3. Execution Commands
To run the high-speed pretraining with DDP, 200 Billion token budget, and active disk space preservation on Kaggle or a cluster:
```powershell
python kaggle/train_120m_kaggle.py
```

## Adaptive Disk Space Protection (Flow Control)

Large-scale training with background tokenization and downloading (using `prepare_hybrid_datasets.py` or `prepare_hf_token_parallel.py`) runs download tasks in parallel with training. To prevent disk overflows when writing massive quantities of `.pt` shards, the downloader pipelines support filesystem-level adaptive flow control:

- **Usage**: Pass `--max-ahead <N>` (e.g., `--max-ahead 5`) to the downloader/tokenizer scripts.
- **Mechanism**: The downloader monitors the output directory for the oldest shard index currently present (`min_active`). If the active tokenization shard index is `max_ahead` shards ahead of `min_active`, the downloader pauses, applying backpressure all the way to the streaming Hugging Face queues.
- **Dynamic Cleanup**: Combined with `--delete-consumed-shards` on the trainer, consumed shards are dynamically deleted from disk, automatically releasing the downloader's pause block and maintaining a tightly bounded local disk footprint.

## Evaluation Rule

CMF Infinity should only claim superiority when all of these are true:

- Parameters are matched within the configured tolerance.
- Held-out loss is equal or better.
- Reasoning/factuality tasks are equal or better.
- Throughput is equal or better.
- Energy is measured and equal or better.
- The benchmark corpus and prompts are not training-set memorization only.

AGI or frontier-level claims require external, broad, contamination-aware evaluation. No current record satisfies that bar.
