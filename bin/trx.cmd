@echo off
rem trx launcher: uses the environment's own venv once it exists (phase 1), else system Python 3.12.
set "TRX_HOME=%~dp0.."
if exist "%TRX_HOME%\toolchain\python\.venv\Scripts\python.exe" (
  "%TRX_HOME%\toolchain\python\.venv\Scripts\python.exe" -I "%TRX_HOME%\trx.py" %*
) else (
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -I "%TRX_HOME%\trx.py" %*
)
exit /b %ERRORLEVEL%
