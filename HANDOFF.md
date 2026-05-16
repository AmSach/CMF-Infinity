# CMF Infinity Handoff

Last updated: **May 16, 2026, 23:54 IST**.

Goal: build and evaluate the CMF Infinity architecture family, where model names use `CMF Infinity <params-in-billions>B`.

## Current State

- Strict checkpoint packaging is implemented in `cmf/checkpointing.py`.
- Device selection helpers are in `cmf/runtime.py`.
- Scalable streaming batches are in `cmf/scalable_data.py`.
- Large-scale training entrypoint is `scripts/train_large_scale.py`.
- Matched benchmark entrypoint is `scripts/run_quality_efficiency.py`.
- C++/CUDA extension status checker is `scripts/check_cpp_extension.py`.
- Evidence-grounded deliberation runtime is implemented in `cmf/infinity_runtime.py`.
- One-shot and chat inference expose deliberation and local knowledge files through `scripts/cmf_infinity_generate.py` and `scripts/cmf_infinity_chat.py`.
- Scalable training can now continue from a strict inference package via `scripts/train_large_scale.py --init-package <package.pt>`.
- Kaggle training and inference launchers are in `kaggle/`, with a bundle builder at `scripts/make_kaggle_bundle.py`.
- Local RTX 4050 Chinchilla-scaling launcher is `scripts/run_rtx4050_chinchilla.py`, exposed through `train_rtx4050_chinchilla.bat`.
- Sharded HF token-cache preparation is implemented in `scripts/prepare_hf_token_shards.py`, and `scripts/train_large_scale.py` accepts `--token-cache-dir`.
- Adaptive solver semantics in `cmf/solver.py` now select a step count first and integrate across exactly one latent-time interval.
- Goal-conditioning now broadcasts correctly for parallel CMF variants in `cmf/model.py`.
- 120M-class smoke gate is implemented in `scripts/run_120m_gate.py`.
- 120M-class preset `infinity-0.12b` is available in `cmf/presets.py`.
- Deliberative architecture `DeliberativeContinuousMeaningField` is implemented in `cmf/model.py`.
- Deliberative 120M-class preset `infinity-reasoning-0.12b` is available in `cmf/presets.py`.

## Verified Commands

```powershell
python -m compileall cmf scripts tests
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
python scripts\run_quality_efficiency.py --device cuda --steps 30 --seq-len 96 --batch-size 8 --bench-iterations 20
python scripts\check_cpp_extension.py
python scripts\run_120m_gate.py --device cuda --amp --steps 12 --seq-len 64 --batch-size 1 --eval-batches 2 --bench-iterations 3 --prompt-limit 8 --max-new-tokens 12
python scripts\run_120m_gate.py --preset infinity-reasoning-0.12b --device cuda --amp --steps 2 --seq-len 32 --batch-size 1 --eval-batches 1 --bench-iterations 1 --prompt-limit 3 --max-new-tokens 6 --thinking-steps 2 --package-out records\checkpoints\cmf_reasoning_120m_gate.package.pt
python scripts\train_large_scale.py --dry-run --device cpu --init-package records\checkpoints\cmf_120m_gate.package.pt --token-cache records\data\smollm_fineweb_edu_gpt2_100k.pt --seq-len 64
python scripts\make_kaggle_bundle.py --include-package records\checkpoints\cmf_120m_gate.package.pt --include-token-cache records\data\smollm_fineweb_edu_gpt2_2m.pt
python scripts\run_rtx4050_chinchilla.py --dry-run
```

## New Deliberation Commands

```powershell
python scripts\cmf_infinity_generate.py <package.pt> --prompt "Explain CMF." --knowledge-file docs\cmf_infinity_architecture.md --deliberation-steps 3 --deliberation-candidates 2 --print-trace
python scripts\cmf_infinity_chat.py <package.pt> --knowledge-file docs\cmf_infinity_architecture.md --deliberation-steps 3 --deliberation-candidates 2
```

For open-ended thinking, always provide a wall-clock stop:

```powershell
python scripts\cmf_infinity_generate.py <package.pt> --prompt "Think deeply." --open-ended-deliberation --max-deliberation-seconds 30
```

## Latest Results

- Tests: `32 passed` with `.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp_revolution`.
- Matched synthetic benchmark: CMF Infinity 0.00037B beats matched Transformer on loss, prompt accuracy, candidate accuracy, throughput, VRAM, and training energy.
- CMF does not beat Transformer on forward energy/token.
- No AGI/frontier claim is supported.
- Deliberation, evidence retrieval, and goal steering are architecture scaffolds. They are not yet proof of frontier-level reasoning or factuality.

## Latest 120M Gate

- Model: `CMF Infinity 0.11952B`, 119,523,840 parameters.
- Package: `records/checkpoints/cmf_120m_gate.package.pt` (~478 MB).
- Kaggle bundle target: `dist/cmf_kaggle_bundle.zip`.
- Report: `records/evals_120m/latest.json` and `records/evals_120m/latest.md`.
- Device: CUDA on RTX 4050 Laptop GPU, 6 GB VRAM.
- Smoke training: 12 steps, seq_len 64, batch 1, AMP.
- Eval loss: 99.79 -> 1.86 on the tiny local gate corpus.
- Prompt accuracy: 0.0; candidate accuracy: 0.125.
- Forward throughput: ~2,721 tokens/s for the measured tiny batch.
- Verdict: local smoke package is loadable; not world-deployable; do not start 8B training on this machine without distributed infrastructure and stronger scaling evidence.

## Latest Deliberative 120M Gate

- Model: `CMF Infinity 0.12075B`, 120,753,921 parameters.
- Package: `records/checkpoints/cmf_reasoning_120m_gate.package.pt`.
- Report: `records/evals_120m/infinity-reasoning-0.12b_latest.json`.
- Device: CUDA on RTX 4050 Laptop GPU, 6 GB VRAM.
- Smoke training: 2 steps, seq_len 32, batch 1, AMP, thinking steps capped at 2.
- Eval loss: 98.67 -> 98.67.
- Prompt accuracy: 0.0; candidate accuracy: 0.0.
- Forward throughput: ~1,107 tokens/s.
- Verdict: architecture path is implemented and packageable, but it did not learn useful reasoning in this tiny run.

## C++/CUDA Status

The extension source exists in `cpp/`, but `python setup.py build_ext --inplace` currently fails because this host lacks Microsoft Visual C++ Build Tools. `records/cpp_extension_status.json` also reports no `nvcc` or `ninja`.

Next native-runtime step:

1. Install MSVC Build Tools.
2. Install CUDA toolkit with `nvcc`.
3. Install `ninja`.
4. Run `python setup.py build_ext --inplace`.
5. Run `python scripts\check_cpp_extension.py`.

## Important Claim Rule

Do not call the model AGI, frontier-level, or 10x cheaper unless broad external evaluations and energy measurements prove it. Current records support only small-benchmark superiority.
