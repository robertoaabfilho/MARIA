@echo off
REM MARIA - no DICOM: recebe estudos do PACS, processa e devolve o resultado (Windows: duplo clique)
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
call "%~dp0preparar_ambiente.bat" || (pause & exit /b 1)
cd /d "%~dp0.."
if not exist "integracao\config_dicom.json" (
  echo Falta integracao\config_dicom.json. Copie integracao\config_dicom.exemplo.json, ajuste AE title,
  echo porta, PACS de destino e AE autorizados, e rode este arquivo de novo.
  pause
  exit /b 1
)
echo No DICOM do MARIA ligado. Deixe esta janela aberta; feche-a para desligar.
"%PY%" %PYV% integracao\no_dicom.py
pause
