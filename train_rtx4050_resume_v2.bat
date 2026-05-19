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

echo ======================================================================
echo 🌌 CMF-v2 HIGH-SPEED PRETRAINING RESUME ENGINE
echo ======================================================================
echo Resuming pre-training from: checkpoint_latest.pt
echo Target budget: 100,000,000 tokens
echo Target steps: 48,828 (Resuming at step 13,545)
echo ======================================================================

.\.venv\Scripts\python.exe -u scripts\run_rtx4050_chinchilla.py ^
  --phase train ^
  --preset infinity-reasoning-0.12b ^
  --no-init-package ^
  --target-tokens 100000000 ^
  --data-dir records\data\chinchilla_gpt2_100m_snapshot ^
  --seq-len 128 ^
  --micro-batch-size 4 ^
  --grad-accum 4 ^
  --steps 48828 ^
  --save-every 500 ^
  --log-every 1 ^
  --resume-checkpoint checkpoint_latest.pt ^
  --run-dir records\runs\rtx4050_100m_v2_resume ^
  %*

pause
