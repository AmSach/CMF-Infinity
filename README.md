# Continuous Meaning Field / CMF Infinity

Continuous Meaning Field (CMF) is a research prototype for modeling language as a continuous trajectory through a learned semantic vector field.

Instead of treating generation only as "predict the next token", CMF learns a vector field that moves a latent state through meaning space. A dilated convolutional encoder reads the prompt into a contextual landscape, an ODE-style solver traces a latent path through that landscape, and a decoder maps the visited points back to vocabulary logits.

This repository contains:

- A paper draft in [paper/continuous_meaning_field.md](paper/continuous_meaning_field.md).
- A BibTeX bibliography in [paper/references.bib](paper/references.bib).
- Architecture notes in [docs/architecture.md](docs/architecture.md).
- CMF Infinity architecture notes in [docs/cmf_infinity_architecture.md](docs/cmf_infinity_architecture.md).
- An experiment plan in [docs/experiment_plan.md](docs/experiment_plan.md).
- A PyTorch prototype in [cmf/](cmf/).
- A C++/CUDA extension scaffold in [cpp/](cpp/).

## Core Idea

Normal autoregressive language models build text as a discrete sequence of token decisions. CMF keeps the token interface, but inserts a continuous latent process between prompt encoding and vocabulary decoding:

```text
tokens -> embeddings -> dilated CNN context -> vector field solver -> latent path -> logits -> tokens
```

The model learns:

```text
dz / dt = F_theta(z, c, t)
```

where `z` is the current semantic state, `c` is prompt context from a dilated CNN, and `F_theta` predicts the direction of movement through the latent field.

## Why Dilated CNNs

Dilated temporal convolutions give the model a wide receptive field with predictable compute and no quadratic attention matrix. They are not a drop-in replacement for attention in every setting, but they are a strong candidate for efficient prompt-conditioned field construction.

## Why C++ and CUDA

Python remains the orchestration layer: datasets, batching, checkpoints, and experiment configuration.

C++/CUDA is reserved for the repetitive inner loop: integrating thousands of latent trajectories across many solver steps. The included extension scaffold exposes an `euler_integrate` operator for precomputed velocity fields. The pure PyTorch path remains the default training path because it keeps autograd simple while the custom backward pass is developed.

On the current machine, the extension source exists but is not built because Microsoft Visual C++ Build Tools, `nvcc`, and `ninja` are not available. See `records/cpp_extension_status.json`.

## CMF Infinity Naming

Adaptive-step CMF models are named:

```text
CMF Infinity <parameter-count-in-billions>B
```

`Infinity` means bounded adaptive latent thinking steps, controlled by `max_solver_steps`; it is not a claim of literal infinite compute or AGI.

## Quick Start

Install the Python package in editable mode:

```powershell
python -m pip install -e .
```

Run the toy training script:

```powershell
python scripts/train_toy.py
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
```

Run the phase research harness:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python scripts/run_phases.py --phase all --device auto
```

The runner writes durable handoff records to [records/HANDOFF.md](records/HANDOFF.md) and phase outputs to `records/phase_*.json`.

Run the matched GPU quality/efficiency comparison:

```powershell
python scripts\run_quality_efficiency.py --device cuda --steps 30 --seq-len 96 --batch-size 8 --bench-iterations 20
```

Current best claims are summarized in [docs/current_claims.md](docs/current_claims.md).

Run scalable streaming training:

```powershell
python scripts\train_large_scale.py --device cuda --tokenizer byte --seq-len 1024 --micro-batch-size 4 --grad-accum 8 --steps 1000 --amp
```

Continue training from an existing CMF package:

```powershell
python scripts\train_large_scale.py --device cuda --init-package records\checkpoints\cmf_120m_gate.package.pt --token-cache records\data\smollm_fineweb_edu_gpt2_2m.pt --seq-len 128 --micro-batch-size 1 --grad-accum 16 --steps 1000 --amp --tf32
```

Run the local RTX 4050 Chinchilla-scaling launcher:

```powershell
.\train_rtx4050_chinchilla.bat
```

By default this targets `20 * parameters` tokens for `infinity-0.12b`, about
2.39B tokens, writes sharded dataset caches under
`records\data\chinchilla_gpt2_120m`, and writes full run logs under
`records\runs\rtx4050_chinchilla_*`. For a short validation run:

```powershell
.\train_rtx4050_chinchilla.bat --target-tokens 100000 --steps 2 --shard-tokens 50000
```

Build a Kaggle-ready source bundle:

```powershell
python scripts\make_kaggle_bundle.py --include-package records\checkpoints\cmf_120m_gate.package.pt --include-token-cache records\data\smollm_fineweb_edu_gpt2_2m.pt
```

Kaggle launchers live in [kaggle/](kaggle/).

Run generation with evidence-grounded deliberation:

```powershell
python scripts\cmf_infinity_generate.py records\checkpoints\large_scale_cmf.package.pt --prompt "Explain the CMF architecture." --knowledge-file docs\cmf_infinity_architecture.md --deliberation-steps 3 --deliberation-candidates 2 --print-trace
```

Use open-ended deliberation only with a wall-clock stop condition:

```powershell
python scripts\cmf_infinity_generate.py records\checkpoints\large_scale_cmf.package.pt --prompt "Reason through this carefully." --open-ended-deliberation --max-deliberation-seconds 30
```

Run the 120M-class smoke gate:

```powershell
python scripts\run_120m_gate.py --device cuda --amp --steps 12 --seq-len 64 --batch-size 1
```

The 120M gate writes a package to `records/checkpoints/cmf_120m_gate.package.pt` and an evaluation report to `records/evals_120m/latest.json`.

Run the deliberative 120M-class smoke gate:

```powershell
python scripts\run_120m_gate.py --preset infinity-reasoning-0.12b --device cuda --amp --steps 2 --seq-len 32 --batch-size 1 --thinking-steps 2 --package-out records\checkpoints\cmf_reasoning_120m_gate.package.pt
```

The deliberative preset uses `DeliberativeContinuousMeaningField`, which performs repeated latent refinement with a learned halting signal.

Build the optional C++/CUDA extension:

```powershell
python setup.py build_ext --inplace
```

If CUDA is not available, `setup.py` builds the CPU extension only.

## Default Training Settings

The prototype defaults to the memory-safe settings described in the project brief:

```text
micro_batch_size = 8
gradient_accumulation_steps = 32
effective_batch_size = 256
```

These settings are in [cmf/config.py](cmf/config.py).

## Claim Discipline

The paper draft intentionally presents CMF as a proposed architecture and research program. It does not claim superiority over Transformers or Mamba until experiments support that claim. The first credible target is to show competitive small-scale language modeling at lower memory or latency for long-context settings.

The latest matched synthetic benchmark shows a small CMF Infinity model beating a matched Transformer on loss, prompt accuracy, candidate accuracy, throughput, peak VRAM, and training energy, but not forward energy. This is not evidence of AGI or frontier-level intelligence.
