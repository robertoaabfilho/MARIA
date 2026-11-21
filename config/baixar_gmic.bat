@echo off
REM MARIA - baixa o GMIC (opcional; melhora a prioridade da mama). Windows: duplo clique neste arquivo.
REM O GMIC (NYU) e licenciado sob a GNU AGPLv3 e nao faz parte do MARIA; ao baixar, voce aceita a licenca dele.
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
call "%~dp0preparar_ambiente.bat" || (pause & exit /b 1)
cd /d "%~dp0.."
"%PY%" %PYV% config\baixar_gmic.py
pause
