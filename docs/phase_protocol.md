# Phase Protocol

This document defines the local gates used by `scripts/run_phases.py`.

The phase harness is a reproducibility and regression tool. It is not an AGI benchmark.

## Phase 0: Sanity Learning

Gate:

- Loss is finite.
- Final evaluation loss improves strongly on a repeated toy pattern.
- Generation contains the expected toy-corpus words.
- A checkpoint is written to `records/checkpoints/phase0_cmf.pt`.

## Phase 1: Small LM Baselines

Gate:

- CMF, TCN, and Transformer tiny LM runs complete.
- Losses are finite and non-regressing.
- Optional Mamba support is recorded if `mamba_ssm` is installed.

## Phase 2: Long-Context Efficiency Smoke

Gate:

- CMF, TCN, and Transformer forward passes complete across context lengths.
- Throughput fields are finite.

## Phase 3: Solver Runtime

Gate:

- PyTorch fallback Euler integration matches the `torch.cumsum` reference.
- Optional `cmf_cuda` extension must match the fallback if it is importable.
- Toolchain availability is recorded.

## Phase 4: Trajectory Analysis

Gate:

- Latent trajectory extraction completes.
- PCA, speed, curvature, and prompt distance metrics are finite.
- CSV and SVG artifacts are written.

## Phase 5: Robustness & Adaptive Flow

Gate:

- Multi-seed validation loss variance is below 0.05.
- 2048-token context smoke benchmark completes.
- Adaptive solver reduces total steps by at least 10%.

## Phase 6: Mechanism Smoke Tests

Gate:

- Memory-anchor parameters exist.
- Goal vectors change logits.
- Adaptive solver metadata is returned.

This phase does not prove reasoning accuracy.

## Phase 7: Subword Scaling & Knowledge Memorization

Gate:

- `SimpleBPETokenizer` trains on the local knowledge corpus.
- Loss reduces by more than 50%.

This phase measures memorization/convergence, not open-world knowledge.

## Phase 8: Goal-Steering Smoke Test

Gate:

- Different goal vectors can be trained to produce different output distributions.

This phase does not prove agentic reasoning.

## Phase 9: Parameter-Matched Fair Comparison

Gate:

- CMF and GPT-like baseline are parameter-matched within tolerance.
- Held-out loss and throughput are finite.
- Superiority is recorded separately and never assumed by the pass flag.

## Phase 10: Multimodal and Fake-Quantization Smoke

Gate:

- Spatial encoder emits the expected sequence shape.
- TorchScript fused-step smoke compiles.
- Fake-quantized CMF and GPT forward passes remain finite.

This phase does not prove a production int8 engine or C++ inference runtime.

## Handoff

After every phase, the runner writes:

- `records/phase_N.json`
- `records/phase_N.md`
- `records/HANDOFF.md`
- `records/environment.json`

Continue with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
python scripts\run_phases.py --phase all --device auto
```
