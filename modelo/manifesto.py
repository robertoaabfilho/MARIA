"""
Manifesto de versões do MARIA (rastreabilidade — IEC 62304 / ISO 13485).

Registra o SHA-256 e o tamanho de cada arquivo de modelo (.pth/.json) e dos exemplos, e calcula a
VERSÃO DO MODELO: os 12 primeiros caracteres do SHA-256 da lista ordenada de hashes dos modelos.
Qualquer troca de peso (retreino, restauração, cópia corrompida) muda a versão.

    python3 manifesto.py --gerar       (re)gera modelo/MANIFESTO.json — faça isso ao liberar uma versão
    python3 manifesto.py --verificar   confere os arquivos atuais contra o manifesto

O maria.py chama verificar() a cada execução e grava a versão em cada linha do resultado.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
ARQ_MANIFESTO = AQUI / "MANIFESTO.json"
PADROES_MODELO = ["modelo_patologia/*.pth", "modelo_achado/*.pth", "modelo_deteccao/*.pth", "modelo_deteccao/*.json",
                  "modelo_densidade/*.pth", "modelo_densidade/*.json", "calibracao.json"]
# (desempenho.json é documentação: não muda a saída, então não entra na versão do modelo)
PADROES_EXEMPLOS = ["exemplos/**/*.dcm", "exemplos/**/*.png", "exemplos/**/*.csv"]


def sha256(caminho, bloco=1 << 22):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            parte = f.read(bloco)
            if not parte:
                break
            h.update(parte)
    return h.hexdigest()


def listar():
    modelos = sorted({p for padrao in PADROES_MODELO for p in AQUI.glob(padrao) if p.is_file()})
    exemplos = sorted({p for padrao in PADROES_EXEMPLOS for p in RAIZ.glob(padrao) if p.is_file()})
    return modelos, exemplos


def versao_de(hashes_modelos):
    texto = "\n".join(f"{nome}:{h}" for nome, h in sorted(hashes_modelos.items()))
    return hashlib.sha256(texto.encode()).hexdigest()[:12]


def gerar(descricao=""):
    modelos, exemplos = listar()
    reg = {"gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "descricao": descricao, "modelos": {}, "exemplos": {}}
    for p in modelos:
        reg["modelos"][p.relative_to(AQUI).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    for p in exemplos:
        reg["exemplos"][p.relative_to(RAIZ).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    reg["versao_modelo"] = versao_de({k: v["sha256"] for k, v in reg["modelos"].items()})
    ARQ_MANIFESTO.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    return reg


def verificar():
    """-> (versao_atual, versao_do_manifesto, divergencias[list[str]])."""
    modelos, _ = listar()
    atuais = {p.relative_to(AQUI).as_posix(): sha256(p) for p in modelos}
    versao_atual = versao_de(atuais)
    if not ARQ_MANIFESTO.is_file():
        return versao_atual, None, ["MANIFESTO.json não encontrado (versão não registrada)"]
    m = json.loads(ARQ_MANIFESTO.read_text(encoding="utf-8"))
    div = []
    for nome, info in m["modelos"].items():
        if nome not in atuais:
            div.append(f"faltando: {nome}")
        elif atuais[nome] != info["sha256"]:
            div.append(f"alterado: {nome}")
    div += [f"não registrado: {n}" for n in atuais if n not in m["modelos"]]
    return versao_atual, m.get("versao_modelo"), div


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--descricao", default="")
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args()
    if a.gerar:
        r = gerar(a.descricao)
        print(f"✅ MANIFESTO.json: versão do modelo {r['versao_modelo']} ({len(r['modelos'])} arquivos de modelo, "
              f"{len(r['exemplos'])} exemplos)")
    else:
        atual, registrada, div = verificar()
        print(f"versão atual {atual} | registrada {registrada}")
        for d in div:
            print("   ⚠️", d)
        sys.exit(1 if div else 0)
