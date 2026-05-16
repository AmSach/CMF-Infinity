# Current Claims: CMF Infinity

Status date: **May 16, 2026**.

CMF Infinity is the project name for the adaptive-step Continuous Meaning Field family. The naming format is:

```text
CMF Infinity <parameter-count-in-billions>B
```

Examples:

- `CMF Infinity 0.00037B`: the 372k-parameter fast CMF used in the latest matched toy benchmark.
- `CMF Infinity 0.203B`: the existing GPT-2-tokenizer legacy checkpoint with about 203M tensor parameters.

`Infinity` means adaptive latent-flow compute with a configurable solver budget. In implementation it is always bounded by `max_solver_steps` for memory, cost, and reproducibility. It is not a claim of unbounded compute or AGI.

## Verified Now

- Tests: `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp_post_120m` passes `29 passed`.
- Strict checkpoint packaging exists in `cmf/checkpointing.py`.
- Raw state dicts no longer masquerade as valid deployment packages.
- Chat scripts now fail closed on missing or incompatible weights.
- Scalable streaming training exists in `scripts/train_large_scale.py` for local text or Hugging Face streaming datasets.
- Matched CMF-vs-Transformer benchmark records are written to `records/quality_efficiency/latest.json`.
- Evidence-grounded deliberative generation exists in `cmf/infinity_runtime.py` and is exposed through `scripts/cmf_infinity_generate.py` and `scripts/cmf_infinity_chat.py`.
- The adaptive solver now selects a bounded step count first, then integrates over exactly one latent-time interval.
- A 120M-class preset and gate exist: `infinity-0.12b` and `scripts/run_120m_gate.py`.
- A deliberative 120M-class preset exists: `infinity-reasoning-0.12b`, backed by `DeliberativeContinuousMeaningField`.

## Latest 120M-Class Gate

Command:

```powershell
python scripts\run_120m_gate.py --device cuda --amp --steps 12 --seq-len 64 --batch-size 1 --eval-batches 2 --bench-iterations 3 --prompt-limit 8 --max-new-tokens 12
```

Scope: smoke training/evaluation on a tiny local corpus. This proves the package can instantiate, train briefly, save, load, prompt, and benchmark. It does not prove deployment quality.

| Metric | Result |
| --- | ---: |
| Parameters | 119,523,840 |
| Eval loss | 99.79 -> 1.86 |
| Prompt accuracy | 0.0 |
| Candidate accuracy | 0.125 |
| Forward throughput | 2,721 tok/s |
| Peak VRAM | 1,906 MB |
| Package | `records/checkpoints/cmf_120m_gate.package.pt` |

Verdict: the 120M package is local-smoke-deployable, but not world-deployable.

## Latest Deliberative 120M Gate

Command:

```powershell
python scripts\run_120m_gate.py --preset infinity-reasoning-0.12b --device cuda --amp --steps 2 --seq-len 32 --batch-size 1 --eval-batches 1 --bench-iterations 1 --prompt-limit 3 --max-new-tokens 6 --thinking-steps 2 --package-out records\checkpoints\cmf_reasoning_120m_gate.package.pt
```

Scope: architecture smoke only, capped to 2 thinking passes for the local 6 GB GPU.

| Metric | Result |
| --- | ---: |
| Parameters | 120,753,921 |
| Eval loss | 98.67 -> 98.67 |
| Prompt accuracy | 0.0 |
| Candidate accuracy | 0.0 |
| Forward throughput | 1,107 tok/s |
| Peak VRAM | 986 MB |
| Package | `records/checkpoints/cmf_reasoning_120m_gate.package.pt` |

Verdict: the deliberative architecture instantiates, trains, saves, loads, and benchmarks, but it has not yet learned useful reasoning in this tiny run.

## Latest Matched GPU Benchmark

Command:

```powershell
python scripts\run_quality_efficiency.py --device cuda --steps 30 --seq-len 96 --batch-size 8 --bench-iterations 20
```

Scope: small synthetic language/reasoning benchmark. This is an architecture smoke benchmark, not a frontier intelligence benchmark.

| Metric | CMF Infinity 0.00037B | Matched Transformer | Result |
| --- | ---: | ---: | --- |
| Parameters | 372,000 | 368,280 | Matched within 1.01% |
| Eval loss | 0.2180 | 1.3330 | CMF better |
| Prompt accuracy | 40% | 0% | CMF better |
| Candidate accuracy | 44% | 16% | CMF better |
| Throughput | 132,862 tok/s | 84,906 tok/s | CMF 1.56x faster |
| Peak train VRAM | 30.3 MB | 44.2 MB | CMF 0.68x |
| Train energy/token | 0.000480 J | 0.000547 J | CMF 0.88x |
| Forward energy/token | 0.0000711 J | 0.0000280 J | Transformer better |

Claim gate result:

- `beats_transformer`: true for this small matched synthetic benchmark.
- `landslide_10x_energy`: false.
- AGI/frontier-level reasoning: not demonstrated.

## C++/CUDA Status

The repository contains a C++/CUDA extension scaffold:

- `cpp/cmf_extension.cpp`
- `cpp/cmf_cuda_kernel.cu`
- `cmf/fast_integrator.py`

The extension is not currently built on this machine. `python setup.py build_ext --inplace` fails because Microsoft Visual C++ Build Tools are missing, and `records/cpp_extension_status.json` also reports no `nvcc` or `ninja`.

Current compiled-kernel claim: source exists and fallback correctness is tested; compiled acceleration is not active on this host.

## Not Yet Proven

- True AGI.
- Frontier-model factuality or reasoning.
- 10x lower power usage.
- Full C++ inference/training runtime.
- Large-corpus scaling on FineWeb-Edu-scale data.
- Frontier-level agentic reasoning from the deliberation scaffold alone.

The next honest milestone is a longer benchmark on real held-out text with the compiled extension active and energy measured over a longer steady-state window.
