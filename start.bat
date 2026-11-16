@echo off
setlocal EnableExtensions
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo 尚未安裝完成，請先執行 install.bat
  pause
  exit /b 1
)

set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONUTF8=1

echo 正在啟動 AI 文字修圖工具...
.venv\Scripts\python.exe app.py
pause
