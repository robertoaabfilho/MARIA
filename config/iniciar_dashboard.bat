@echo off
REM MARIA - abre o dashboard no navegador (Windows: duplo clique neste arquivo)
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
call "%~dp0preparar_ambiente.bat" || (pause & exit /b 1)
cd /d "%~dp0.."
echo.
echo Abrindo o MARIA no navegador. O endereco aparece abaixo (normalmente http://localhost:8501).
echo Deixe esta janela aberta enquanto usa; feche-a para desligar o MARIA.
"%PY%" %PYV% -m streamlit run maria_dashboard.py --browser.gatherUsageStats false
pause
