"""
Baixa o GMIC (NYU) para modelo/modulo_gmic/ — opcional, mas recomendado: melhora a PRIORIDADE DA MAMA.

O GMIC é de terceiros (Shen et al., Medical Image Analysis 2021; github.com/nyukat/GMIC) e é licenciado sob a
GNU AGPLv3 — por isso NÃO vem junto com o MARIA. Ao baixar, você aceita a licença dele (modelo/modulo_gmic/LICENSE).
Versão fixa (a testada) e pesos conferidos por SHA-256.

    python config/baixar_gmic.py            (Windows: py config\\baixar_gmic.py)
    python config/baixar_gmic.py --conferir (só confere o que já foi baixado)
"""
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "modelo"))
import gmic_maria as G  # noqa: E402

PARTES = ("src/", "models/", "LICENSE", "NOTICE", "README.md")


def main():
    for fluxo in (sys.stdout, sys.stderr):       # Windows: acentos no console/pipe
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if "--conferir" in sys.argv:
        ruins = G.conferir()
        print("GMIC ok" if not ruins else "GMIC com problema:\n  " + "\n  ".join(ruins))
        sys.exit(1 if ruins else 0)
    if not G.conferir():
        print(f"O GMIC já está em {G.PASTA} e confere. Nada a fazer.")
        return
    print("O GMIC (NYU) é licenciado sob a GNU AGPLv3 e não faz parte do MARIA.")
    print(f"Baixando a versão {G.COMMIT[:7]} de github.com/nyukat/GMIC (~410 MB)...")
    with urllib.request.urlopen(G.URL_ZIP, timeout=120) as r:
        total, dados, lido = int(r.headers.get("Content-Length") or 0), io.BytesIO(), 0
        while bloco := r.read(1 << 20):
            dados.write(bloco)
            lido += len(bloco)
            print(f"\r   {lido / 1e6:.0f} MB" + (f" de {total / 1e6:.0f} MB" if total else ""), end="", flush=True)
    print()
    tmp = G.PASTA.with_name("modulo_gmic_baixando")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(dados) as z:
        for info in z.infolist():
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
            if info.is_dir() or not rel.startswith(PARTES) or ".." in Path(rel).parts:
                continue
            destino = tmp / rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(z.read(info))
    shutil.rmtree(G.PASTA, ignore_errors=True)
    tmp.rename(G.PASTA)
    ruins = G.conferir()
    if ruins:
        sys.exit("O download não confere:\n  " + "\n  ".join(ruins))
    print(f"GMIC instalado em {G.PASTA} e conferido (SHA-256). Licença: {G.PASTA / 'LICENSE'}")


if __name__ == "__main__":
    main()
