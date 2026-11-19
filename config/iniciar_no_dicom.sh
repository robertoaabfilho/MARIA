#!/bin/bash
# MARIA - nó DICOM (Linux/macOS): recebe estudos do PACS, processa e devolve o resultado
cd "$(dirname "$0")/.."
[ -f integracao/config_dicom.json ] || { echo "Falta integracao/config_dicom.json (copie de config_dicom.exemplo.json e ajuste)."; exit 1; }
python3 -c "import pynetdicom" 2>/dev/null || python3 -m pip install -r config/requirements.txt
python3 integracao/no_dicom.py
