"""
Registro de auditoria do MARIA (LGPD / segurança da informação): quem fez o quê, quando e em qual máquina.

Arquivo: auditoria/auditoria.jsonl (uma linha JSON por evento, só acréscimo). Cada linha guarda o hash SHA-256
da anterior ("encadeamento"): apagar, alterar ou reordenar linhas quebra a cadeia, e verificar() aponta onde.
O registro NÃO guarda imagens nem diagnósticos; guarda a ação, o usuário e detalhes técnicos (pasta, quantidade
de imagens, versão do modelo, hash do arquivo de resultado).

    python3 auditoria.py --verificar      confere a integridade da cadeia
    python3 auditoria.py --ultimos 20     mostra os últimos eventos
"""
import argparse
import getpass
import hashlib
import json
import platform
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQ_AUDITORIA = RAIZ / "auditoria" / "auditoria.jsonl"
INICIO = "0" * 64


def _hash(evento):
    corpo = {k: v for k, v in evento.items() if k != "hash"}
    return hashlib.sha256(json.dumps(corpo, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _ultimo_hash():
    if not ARQ_AUDITORIA.is_file() or ARQ_AUDITORIA.stat().st_size == 0:
        return INICIO
    with open(ARQ_AUDITORIA, "rb") as f:            # lê só o fim do arquivo
        f.seek(0, 2)
        tam = f.tell()
        f.seek(max(0, tam - 65536))
        linhas = [x for x in f.read().decode("utf-8", errors="replace").splitlines() if x.strip()]
    try:
        return json.loads(linhas[-1])["hash"]
    except Exception:
        return "corrompido"


def registrar(usuario, acao, detalhes=None):
    """Acrescenta um evento. usuario=None usa o usuário do sistema operacional (linha de comando)."""
    ARQ_AUDITORIA.parent.mkdir(parents=True, exist_ok=True)
    evento = {"quando": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "usuario": usuario or f"{getpass.getuser()} (sistema)",
              "acao": acao, "detalhes": detalhes or {}, "maquina": platform.node(), "anterior": _ultimo_hash()}
    evento["hash"] = _hash(evento)
    with open(ARQ_AUDITORIA, "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    return evento


def ler(ultimos=None):
    if not ARQ_AUDITORIA.is_file():
        return []
    linhas = [x for x in ARQ_AUDITORIA.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
    linhas = linhas[-ultimos:] if ultimos else linhas
    out = []
    for x in linhas:
        try:
            out.append(json.loads(x))
        except Exception:
            out.append({"quando": "?", "usuario": "?", "acao": "LINHA ILEGÍVEL", "detalhes": {"texto": x[:200]}})
    return out


def verificar():
    """-> (íntegro: bool, eventos: int, problema: str|None)."""
    anterior = INICIO
    eventos = ler()
    for i, ev in enumerate(eventos, 1):
        if ev.get("acao") == "LINHA ILEGÍVEL":
            return False, len(eventos), f"linha {i} ilegível"
        if ev.get("anterior") != anterior:
            return False, len(eventos), f"linha {i}: encadeamento quebrado (linha apagada ou fora de ordem)"
        if _hash(ev) != ev.get("hash"):
            return False, len(eventos), f"linha {i}: conteúdo alterado"
        anterior = ev["hash"]
    return True, len(eventos), None


def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for parte in iter(lambda: f.read(1 << 20), b""):
            h.update(parte)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--ultimos", type=int, default=0)
    a = ap.parse_args()
    if a.ultimos:
        for ev in ler(a.ultimos):
            print(ev["quando"], "|", ev["usuario"], "|", ev["acao"], "|", json.dumps(ev.get("detalhes", {}), ensure_ascii=False))
    ok, n, problema = verificar()
    print(f"{'✅ íntegro' if ok else '❌ ADULTERADO'}: {n} evento(s)" + (f" — {problema}" if problema else ""))
