#!/bin/bash
# MARIA - nó DICOM (Linux/macOS): recebe estudos do PACS, processa e devolve o resultado
cd "$(dirname "$0")/.."
[ -f integracao/config_dicom.json ] || { echo "Falta integracao/config_dicom.json (copie de config_dicom.exemplo.json e ajuste)."; exit 1; }
./config/preparar_ambiente.sh || exit 1
python3 integracao/no_dicom.py
