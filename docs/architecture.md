# CMF Architecture

## System Overview

```mermaid
flowchart LR
    A["Token IDs"] --> B["Embedding Table"]
    B --> C["Dilated Context Encoder"]
    C --> D["Context Landscape C"]
    D --> E["Vector Field F_theta"]
    E --> F["ODE Solver"]
    F --> G["Latent Trajectory z(t)"]
    G --> H["Vocabulary Decoder"]
    H --> I["Token Logits"]
```

CMF keeps the normal token input and output interface, but changes the internal generation mechanism from direct next-token projection to latent motion through a learned field.

## Components

### Token Embedding

Input token IDs are mapped to vectors:

```text
e_t = E[x_t]
```

The embedding matrix can be tied to the output decoder to encourage geometry alignment between continuous states and token embeddings.

### Dilated Context Encoder

The context encoder is a stack of residual temporal convolution blocks. Dilation expands the receptive field without an attention matrix:

```text
dilations = 1, 2, 4, 8, ...
```

For language modeling, the prototype uses causal convolution. For fixed prompt encoding, a non-causal variant can be used.

### Vector Field

The field receives the current latent state, the local context, and continuous solver time:

```text
v = F_theta(z, c, tau)
```

The output `v` is a direction in semantic space.

### ODE Solver

The solver advances the latent state:

```text
z_next = z + dt * v
```

The reference implementation uses PyTorch. The extension scaffold provides a C++/CUDA path for integrating precomputed velocity tensors.

### Decoder

The decoder maps trajectory points to vocabulary logits:

```text
logits = z_norm @ E.T
```

The first implementation uses tied weights by default.

## Training Loop

```mermaid
sequenceDiagram
    participant Data
    participant Python
    participant Model
    participant Solver
    participant Optimizer

    Data->>Python: micro-batch of 8 sequences
    Python->>Model: input IDs and labels
    Model->>Solver: integrate latent path
    Solver-->>Model: latent states
    Model-->>Python: loss
    Python->>Python: loss / 32
    Python->>Optimizer: backward accumulation
    Python->>Optimizer: step after 32 micro-batches
```

## Memory Strategy

Default settings:

```text
micro_batch_size = 8
gradient_accumulation_steps = 32
effective_batch_size = 256
```

This trades wall-clock overhead for lower peak memory. Moving the inner solver to C++/CUDA is intended to recover some of that overhead.

## Inference Runtime

The model can now be run through the Infinity runtime layer:

```mermaid
flowchart LR
    A["User task"] --> B["EvidenceMemory retrieval"]
    B --> C["Grounded prompt"]
    C --> D["Candidate generation"]
    D --> E["Model verifier score"]
    E --> F["Evidence and consensus rerank"]
    F --> G["Best response plus trace"]
```

This layer lives in `cmf/infinity_runtime.py`. It is meant to make reasoning compute explicit: callers can spend more candidate-generation steps on hard tasks, attach a local knowledge file, and inspect the selected answer trace. Open-ended deliberation requires a wall-clock stop condition.

## Near-Term Engineering Milestones

1. Establish PyTorch reference correctness.
2. Benchmark Python solver overhead.
3. Build and test C++/CUDA Euler integration.
4. Add custom backward or composite autograd support for the fast path.
5. Fuse vector-field evaluation with integration.
6. Add native parity for adaptive step selection and deliberation telemetry.
7. Compare against Transformer, Mamba, and TCN baselines.
