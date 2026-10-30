@echo off
rem 可选服务前台安装与启动；服务名由首个参数提供，模型下载需显式参数。
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
where py >nul 2>nul
if errorlevel 1 (
  python service.py %*
) else (
  py -3 service.py %*
)
if errorlevel 1 pause
endlocal
