@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%
set HF_HOME=%CD%\records\runtime_cache\huggingface
set HF_DATASETS_CACHE=%CD%\records\runtime_cache\huggingface\datasets
set HUGGINGFACE_HUB_CACHE=%CD%\records\runtime_cache\huggingface\hub
set TRANSFORMERS_CACHE=%CD%\records\runtime_cache\huggingface\hub
set TORCH_HOME=%CD%\records\runtime_cache\torch
set TMP=%CD%\records\runtime_cache\tmp
set TEMP=%CD%\records\runtime_cache\tmp
.\.venv\Scripts\python.exe scripts\run_rtx4050_chinchilla.py %*
pause
