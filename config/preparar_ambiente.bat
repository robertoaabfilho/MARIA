@echo off
REM MARIA - prepara o Windows para rodar o MARIA (usado pelos outros .bat; pode ser rodado sozinho)
REM  1) acha um Python 3.10-3.13 (se nao houver, instala o Python 3.12 sozinho via winget)
REM  2) se houver placa NVIDIA, instala o PyTorch com GPU; senao, segue na CPU
REM  3) instala as demais bibliotecas (config\requirements.txt)
REM Deixa as variaveis PY e PYV para quem chamou:  "%PY%" %PYV% script.py
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1

REM ---- 1) Python ----
call :achar_python && goto :tem_python
where winget >nul 2>nul || goto :sem_python
echo Python 3.10-3.13 nao encontrado: instalando o Python 3.12 automaticamente (so na primeira vez^)...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "PYV="
call :versao_ok && goto :tem_python
call :achar_python && goto :tem_python
:sem_python
echo Nao consegui encontrar nem instalar o Python automaticamente.
echo Instale o Python 3.12 pelo site https://www.python.org/downloads/ e rode este arquivo de novo.
exit /b 1

:tem_python
echo Python encontrado:
"%PY%" %PYV% --version

REM ---- 2) PyTorch: com placa NVIDIA, versao com CUDA (GPU); senao, CPU ----
nvidia-smi >nul 2>nul || goto :sem_gpu
"%PY%" %PYV% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && goto :bibliotecas
echo Placa NVIDIA encontrada: instalando o PyTorch com suporte a GPU (so na primeira vez, ~3 GB^)...
"%PY%" %PYV% -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126 && goto :bibliotecas
echo Nao consegui instalar o PyTorch com GPU. Vou seguir com a versao de CPU (funciona, mas mais devagar^).
goto :bibliotecas
:sem_gpu
echo Nenhuma placa NVIDIA encontrada: o MARIA vai rodar na CPU (mais devagar, mas funciona^).

REM ---- 3) demais bibliotecas ----
:bibliotecas
"%PY%" %PYV% -c "import streamlit, torch, torchvision, efficientnet_pytorch, pydicom, pynetdicom, pandas, altair, plotly, sys; sys.exit(0 if tuple(int(x) for x in streamlit.__version__.split('.')[:2]) >= (1, 67) else 1)" >nul 2>nul && exit /b 0
echo Instalando ou atualizando as bibliotecas do MARIA (pode demorar alguns minutos^)...
"%PY%" %PYV% -m pip install -r "%~dp0requirements.txt" || (echo Falha ao instalar as bibliotecas. & exit /b 1)
exit /b 0


REM ---- sub-rotinas ----
:achar_python
set "PY=python" & set "PYV=" & call :versao_ok && exit /b 0
for %%V in (3.13 3.12 3.11 3.10) do (
  set "PY=py" & set "PYV=-%%V" & call :versao_ok && exit /b 0
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
  set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe" & set "PYV=" & call :versao_ok && exit /b 0
)
exit /b 1

:versao_ok
"%PY%" %PYV% -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
exit /b %errorlevel%
