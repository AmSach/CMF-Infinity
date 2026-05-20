@echo off
echo ============================================================
echo Starting FULL PRE-TRAINING from scratch for the 120M Model
echo ============================================================
echo.
echo Architecture: CMF-v2 (FlashAttention + Causal Shifted Loss)
echo Dataset: WikiText-103 (Streaming)
echo.

.\.venv\Scripts\python.exe scripts\train_large_scale.py ^
    --preset infinity-reasoning-0.12b ^
    --dataset wikitext ^
    --dataset-name wikitext-103-raw-v1 ^
    --seq-len 512 ^
    --micro-batch-size 8 ^
    --grad-accum 4 ^
    --steps 25000 ^
    --lr 5e-4 ^
    --amp ^
    --save-every 1000 ^
    --checkpoint records\checkpoints\cmf_120m_fresh_pretrain.pt ^
    --package-out records\checkpoints\cmf_120m_pretrained_latest.package.pt

echo.
echo Training complete! 
pause
