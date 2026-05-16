# Results Summary

Last updated: **May 16, 2026, 21:41 IST**.

## Environment

- Python: 3.12.7
- PyTorch: 2.5.1+cu121
- GPU visible: NVIDIA GeForce RTX 4050 Laptop GPU
- C++ extension: not built on this host
- Missing native toolchain: MSVC `cl`, `nvcc`, and `ninja`

## Test Status

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
```

Result: `18 passed`.

## Latest Matched Benchmark

Command:

```powershell
python scripts\run_quality_efficiency.py --device cuda --steps 30 --seq-len 96 --batch-size 8 --bench-iterations 20
```

| Metric | CMF Infinity 0.00037B | Matched Transformer |
| --- | ---: | ---: |
| Parameters | 372,000 | 368,280 |
| Eval loss | 0.2180 | 1.3330 |
| Prompt accuracy | 40% | 0% |
| Candidate accuracy | 44% | 16% |
| Throughput | 132,862 tok/s | 84,906 tok/s |
| Peak train VRAM | 30.3 MB | 44.2 MB |
| Train energy/token | 0.000480 J | 0.000547 J |
| Forward energy/token | 0.0000711 J | 0.0000280 J |

Interpretation:

- CMF wins this small matched synthetic benchmark on quality, throughput, VRAM, and training energy.
- CMF does not win forward energy/token.
- This is not a frontier or AGI result.

Full records:

- `records/quality_efficiency/latest.json`
- `records/quality_efficiency/latest.md`
- `records/quality_efficiency/*_prompt_rows.csv`
- `records/quality_efficiency/*_candidate_rows.csv`

## C++/CUDA Status

Command:

```powershell
python scripts\check_cpp_extension.py
```

Result:

- `cmf_cuda` importable: false
- Python reference integration: finite
- `cl`: missing
- `nvcc`: missing
- `ninja`: missing

Full record: `records/cpp_extension_status.json`.

## Current Technical Read

The project is now a stricter CMF Infinity research harness:

- Deployment paths no longer silently use random weights.
- Checkpoints can be packaged with config and tokenizer metadata.
- Scalable streaming training exists.
- Benchmarks require parameter matching.
- Claims docs no longer assert AGI/frontier behavior.

The next real engineering milestone is to install the native toolchain, build `cmf_cuda`, and then profile which kernels actually dominate runtime before moving more of the model into C++/CUDA.
