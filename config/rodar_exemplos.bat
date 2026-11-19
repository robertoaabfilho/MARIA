@echo off
REM MARIA - roda o modelo na pasta "exemplos" e gera modelo\results.csv (Windows: duplo clique)
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
call "%~dp0preparar_ambiente.bat" || (pause & exit /b 1)
cd /d "%~dp0..\modelo"
echo.
"%PY%" %PYV% maria.py --pasta ..\exemplos --planilha ..\exemplos\planilha_medico.csv
echo.
echo Resultado em: %cd%\results.csv
pause
