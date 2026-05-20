@echo off
echo Max‑tok: %MAXTOK%
echo ------------------------------------------------------------

.\.venv\Scripts\python.exe scripts\cmf_stream_generate.py ^
    %PACKAGE% ^
    --prompt "%PROMPT%" ^
    --deliberation-steps %STEPS% ^
    --temperature %TEMP% ^
    --top-k %TOPK% ^
    --top-p %TOPP% ^
    --max-new-tokens %MAXTOK% %EXTRA_ARGS%

%EXTRA_ARGS%

endlocal
