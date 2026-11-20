#!/bin/bash
# MARIA - abre o dashboard (Linux/macOS):  ./config/iniciar_dashboard.sh
cd "$(dirname "$0")/.."
./config/preparar_ambiente.sh || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "Abrindo o MARIA no navegador (o endereço aparece abaixo, normalmente http://localhost:8501). Ctrl+C desliga."
python3 -m streamlit run maria_dashboard.py --browser.gatherUsageStats false
