@echo off
echo ============================================================
echo Starting CMF LoRA Fine-Tuning...
echo ============================================================
call .\.venv\Scripts\activate.bat
python scripts\train_lora_slimpajama_200m.py
pause
