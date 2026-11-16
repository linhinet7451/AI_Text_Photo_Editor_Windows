@echo off
setlocal EnableExtensions
cd /d %~dp0

echo [1/5] 偵測 Python（優先順序：3.11、3.12、3.13）...
set PYCMD=
py -3.11 -V >nul 2>nul && set PYCMD=py -3.11
if not defined PYCMD (
  py -3.12 -V >nul 2>nul && set PYCMD=py -3.12
)
if not defined PYCMD (
  py -3.13 -V >nul 2>nul && set PYCMD=py -3.13
)
if not defined PYCMD (
  py -3 -V >nul 2>nul && set PYCMD=py -3
  if defined PYCMD echo 警告：只找到未指定版本的 Python（py -3），版本未經驗證，可能與 requirements.txt 鎖定的套件版本不相容。
)
if not defined PYCMD (
  python -V >nul 2>nul && set PYCMD=python
  if defined PYCMD echo 警告：使用 python 指令找到的 Python，版本未經驗證，可能與 requirements.txt 鎖定的套件版本不相容。
)
if not defined PYCMD (
  echo 找不到可用的 Python。請先安裝 Python 3.11、3.12 或 3.13。
  pause
  exit /b 1
)

echo 使用 %PYCMD%

echo [2/5] 建立虛擬環境...
%PYCMD% -m venv .venv
if errorlevel 1 goto :fail

echo [3/5] 升級 pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [4/5] 安裝相依套件（第一次可能需較久）...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [5/5] 安裝完成！
echo 下一步：雙擊 start.bat 啟動工具。
pause
exit /b 0

:fail
echo 安裝失敗，請截圖整個視窗內容。
pause
exit /b 1
