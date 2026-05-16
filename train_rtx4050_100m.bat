@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1
set HF_HOME=%CD%\records\runtime_cache\huggingface
set HF_DATASETS_CACHE=%CD%\records\runtime_cache\huggingface\datasets
set HUGGINGFACE_HUB_CACHE=%CD%\records\runtime_cache\huggingface\hub
set TRANSFORMERS_CACHE=%CD%\records\runtime_cache\huggingface\hub
set TORCH_HOME=%CD%\records\runtime_cache\torch
set TMP=%CD%\records\runtime_cache\tmp
set TEMP=%CD%\records\runtime_cache\tmp

rem Train CMF Infinity 0.12B for a 100M-token local budget from already-downloaded shards.
rem This skips dataset download and reads records\data\chinchilla_gpt2_120m.
rem Defaults: seq_len=128, micro_batch=4, grad_accum=4 => 2,048 tokens/optimizer step.
rem 100,000,000 tokens => 48,828 optimizer steps without exceeding 100M.

.\.venv\Scripts\python.exe -u scripts\snapshot_token_cache.py ^
  --source-dir records\data\chinchilla_gpt2_120m ^
  --output-dir records\data\chinchilla_gpt2_100m_snapshot ^
  --target-tokens 100000000 ^
  --overwrite

.\.venv\Scripts\python.exe -u scripts\run_rtx4050_chinchilla.py ^
  --phase train ^
  --target-tokens 100000000 ^
  --data-dir records\data\chinchilla_gpt2_100m_snapshot ^
  --seq-len 128 ^
  --micro-batch-size 4 ^
  --grad-accum 4 ^
  --steps 48828 ^
  --save-every 500 ^
  --log-every 1 ^
  %*

pause
