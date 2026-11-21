#!/bin/bash
# MARIA - baixa o GMIC (opcional; melhora a prioridade da mama). Linux/macOS:  ./config/baixar_gmic.sh
# O GMIC (NYU) é licenciado sob a GNU AGPLv3 e não faz parte do MARIA; ao baixar, você aceita a licença dele.
cd "$(dirname "$0")/.."
./config/preparar_ambiente.sh || exit 1
python3 config/baixar_gmic.py
