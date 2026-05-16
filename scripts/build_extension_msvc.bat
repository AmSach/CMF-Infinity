@echo off
setlocal
set VSDEVCMD=C:\BuildTools\Common7\Tools\VsDevCmd.bat
if not exist "%VSDEVCMD%" (
  echo Missing %VSDEVCMD%
  exit /b 1
)
call "%VSDEVCMD%" -arch=x64
set DISTUTILS_USE_SDK=1
set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" setup.py build_ext --inplace
) else (
  python setup.py build_ext --inplace
)
exit /b %ERRORLEVEL%
