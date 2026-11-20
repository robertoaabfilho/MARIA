#!/bin/bash
# MARIA - confere/instala as bibliotecas no Linux/macOS (usado pelos outros .sh; pode ser rodado sozinho).
# Usa o python3 do sistema (3.10 a 3.13, ou mais novo). Sem ambiente virtual.
cd "$(dirname "$0")/.."
if ! command -v python3 >/dev/null; then
  echo "Python 3 não encontrado. Instale o Python 3.10 ou mais novo (ex.: sudo apt install python3 python3-pip)."; exit 1
fi
python3 -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" || {
  echo "Este MARIA precisa do Python 3.10 ou mais novo (encontrado: $(python3 --version))."; exit 1; }
python3 -c "import streamlit, torch, torchvision, efficientnet_pytorch, pydicom, pynetdicom, pandas, altair, plotly, sys; sys.exit(0 if tuple(int(x) for x in streamlit.__version__.split('.')[:2]) >= (1, 67) else 1)" 2>/dev/null && exit 0
python3 -m pip --version >/dev/null 2>&1 || {
  echo "O pip não está instalado. Instale com:  sudo apt install python3-pip   (ou o equivalente da sua distribuição)"; exit 1; }
echo "Instalando ou atualizando as bibliotecas do MARIA (pode demorar alguns minutos; ~3 GB com GPU NVIDIA)..."
saida=$(python3 -m pip install -r config/requirements.txt 2>&1) && exit 0
if echo "$saida" | grep -q "externally-managed-environment"; then
  echo "Esta distribuição (ex.: Ubuntu 23.04+, Debian 12+) não deixa o pip instalar no Python do sistema."
  echo "Instalando só para o seu usuário (em ~/.local), sem mexer nos pacotes do sistema..."
  python3 -m pip install --user --break-system-packages -r config/requirements.txt && exit 0
fi
echo "$saida" | tail -15
echo "Falha ao instalar as bibliotecas. Verifique a internet e o espaço em disco e rode de novo."
exit 1
