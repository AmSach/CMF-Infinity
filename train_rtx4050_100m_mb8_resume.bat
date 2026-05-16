@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1

rem Visible realtime resume test: bigger micro-batch on RTX 4050.
rem Resume from the latest saved checkpoint of the current mb4 run.
rem micro_batch=8, grad_accum=2 keeps 2,048 tokens/update.

.\.venv\Scripts\python.exe -u scripts\run_rtx4050_chinchilla.py ^
  --phase train ^
  --target-tokens 100000000 ^
  --data-dir records\data\chinchilla_gpt2_100m_snapshot ^
  --seq-len 128 ^
  --micro-batch-size 8 ^
  --grad-accum 2 ^
  --steps 48828 ^
  --save-every 500 ^
  --log-every 1 ^
  --resume-checkpoint records\runs\rtx4050_100m_mb4_resume_20260517_014238\checkpoints\cmf_120m_train.pt ^
  --run-dir records\runs\rtx4050_100m_mb8_resume ^
  %*

pause
