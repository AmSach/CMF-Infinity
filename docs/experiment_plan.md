# CMF Experiment Plan

## Phase 0: Sanity Checks

Goal: prove that the implementation learns at all.

- Dataset: repeated toy strings and tiny character corpora.
- Metric: loss decreases within a few hundred steps.
- Models: CMF only.
- Success: training loss falls and generation reflects the training distribution.

## Phase 1: Small Language Modeling

Goal: compare quality and speed on reproducible small tasks.

- Dataset: Tiny Shakespeare, WikiText-2.
- Metrics: validation perplexity, tokens/sec, peak GPU memory.
- Baselines: small Transformer decoder, TCN, CMF.
- Ablations: solver steps per token, dilation depth, tied embeddings.

## Phase 2: Long-Context Efficiency

Goal: test whether CMF's convolutional field construction gives useful scaling.

- Dataset: long contiguous text chunks.
- Context lengths: 512, 1024, 2048, 4096.
- Metrics: memory scaling, throughput scaling, perplexity change.
- Baselines: Transformer with equivalent parameter budget, Mamba if available.

## Phase 3: Solver Runtime

Goal: justify the C++/CUDA section of the paper.

- Compare PyTorch Euler, C++ CPU Euler, CUDA Euler.
- Measure solver-only latency and full model latency.
- Report speedup by batch size, hidden dimension, and solver steps.
- Add custom backward before claiming training speedups from CUDA.

## Phase 4: Trajectory Analysis

Goal: make the "meaning flow" claim inspectable.

- Visualize latent states with PCA or UMAP.
- Measure trajectory curvature by text type.
- Compare nearest-token paths against sampled text.
- Test whether similar prompts begin in nearby regions and diverge smoothly.

## Phase 5: Robustness & Adaptive Flow

Goal: move from synthetic byte-level facts to real language modeling.

- Dataset: WikiText-2 or Tiny Shakespeare.
- Metrics: Multi-seed perplexity, adaptive step efficiency, 2048-context throughput.
- Success: CMF matches Transformer quality while maintaining a >20% efficiency lead across 3 seeds.

## Reporting Rules

- Do not claim "beats Transformers" unless the same parameter budget, data budget, tokenizer, and evaluation protocol are used.
- Separate solver speed from full-model speed.
- Report both best and median runs across seeds.
- Include failed ablations; they are useful for a new architecture.

