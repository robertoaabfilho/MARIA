#!/bin/bash
# MARIA - roda o modelo na pasta "exemplos" e gera modelo/results.csv (Linux/macOS)
cd "$(dirname "$0")/../modelo"
python3 maria.py --pasta ../exemplos --planilha ../exemplos/planilha_medico.csv
echo "Resultado em: $(pwd)/results.csv"
