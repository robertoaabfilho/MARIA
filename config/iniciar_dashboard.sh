#!/bin/bash
# MARIA - abre o dashboard (Linux/macOS):  ./iniciar_dashboard.sh
cd "$(dirname "$0")/.."
python3 -c "import streamlit, torch, torchvision, efficientnet_pytorch, pydicom, pandas, altair, sys; sys.exit(0 if tuple(int(x) for x in streamlit.__version__.split('.')[:2]) >= (1, 67) else 1)" 2>/dev/null || python3 -m pip install -r config/requirements.txt
echo "Abrindo o MARIA no navegador (o endereço aparece abaixo, normalmente http://localhost:8501). Ctrl+C desliga."
python3 -m streamlit run maria_dashboard.py --browser.gatherUsageStats false
