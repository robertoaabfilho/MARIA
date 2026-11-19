#!/usr/bin/env python3
"""
MARIA - Modelo de Análise Radiomamográfica por Inteligência Artificial — dashboard (Streamlit).

Fica na pasta principal do MARIA, ao lado da pasta `modelo/` (maria.py + pesos):
    streamlit run maria_dashboard.py        (ou os atalhos da pasta config/)

O dashboard NÃO reimplementa o modelo: ele chama `python maria.py` dentro da pasta modelo (subprocess),
lê o CSV que ele gera e mostra os resultados. Assim o resultado é idêntico ao da linha de comando.
A planilha do médico (densidade, arquivos, lesões marcadas) pode ser carregada e editada aqui.
Duas páginas na barra lateral: "Sobre o modelo" (descrição e desempenho, lidos de modelo/desempenho.json)
e "Executar" (imagens e planilha, resultados, anotações e retreino).

AVISO: protótipo de pesquisa, não é dispositivo médico. O limiar 0,5 foi calibrado no CBIS-DDSM.
"""
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="MARIA", page_icon="🩻", layout="wide")

if tuple(int(x) for x in st.__version__.split(".")[:2]) < (1, 67):
    st.error(f"Este dashboard precisa do Streamlit 1.67 ou mais novo (instalado: {st.__version__}). Atualize com:\n\n"
             f"    {sys.executable} -m pip install -U streamlit\n\nou rode config/preparar_ambiente.bat (Windows).")
    st.stop()

# -----------------------------------------------------------------------------
# Dependências: instala sozinho o que faltar, no MESMO Python que está rodando o Streamlit
# (é esse Python que roda o maria.py). Formato: módulo importável -> pacote do pip.
# -----------------------------------------------------------------------------
DEPENDENCIAS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "pydicom": "pydicom",
    "tqdm": "tqdm",
    "torch": "torch",
    "torchvision": "torchvision",
    "efficientnet_pytorch": "efficientnet_pytorch",
    "altair": "altair",
}


def instalar_dependencias():
    faltando = {m: p for m, p in DEPENDENCIAS.items() if importlib.util.find_spec(m) is None}
    if not faltando:
        return
    pacotes = list(faltando.values())
    with st.spinner(f"Instalando dependências que faltam: {', '.join(pacotes)} (pode levar alguns minutos)…"):
        r = subprocess.run([sys.executable, "-m", "pip", "install", *pacotes],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    importlib.invalidate_caches()
    ainda = [m for m in faltando if importlib.util.find_spec(m) is None]
    if r.returncode != 0 or ainda:
        st.error(f"Não consegui instalar: {', '.join(faltando[m] for m in (ainda or faltando))}. "
                 f"Instale manualmente com:  {sys.executable} -m pip install {' '.join(pacotes)}")
        with st.expander("Saída do pip"):
            st.code((r.stdout + "\n" + r.stderr)[-4000:], language="text")
        st.stop()
    st.success(f"Dependências instaladas: {', '.join(pacotes)}")


if not st.session_state.get("_deps_ok"):
    instalar_dependencias()
    st.session_state["_deps_ok"] = True

import time  # noqa: E402

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

AQUI = Path(__file__).resolve().parent
EXT_IMG = {".png", ".jpg", ".jpeg", ".dcm"}
SAIDAS = AQUI / "resultados"

# planilha do médico: colunas na ordem em que aparecem, e nomes alternativos aceitos ao carregar um CSV
COLUNAS_PLANILHA = ["mama", "densidade", "vista", "arquivo_completa", "arquivo_recorte", "lesao", "x0", "y0", "x1", "y1"]
SINONIMOS = {
    "mama": ("mama", "amostra", "paciente", "caso", "id"),
    "densidade": ("densidade", "densidade_mamaria", "densidade_birads", "acr"),
    "vista": ("vista", "incidencia", "view"),
    "arquivo_completa": ("arquivo_completa", "completa", "imagem_completa", "mamografia"),
    "arquivo_recorte": ("arquivo_recorte", "recorte", "imagem_recorte"),
    "lesao": ("lesao", "id_lesao"),
    "x0": ("x0", "xmin"), "y0": ("y0", "ymin"), "x1": ("x1", "xmax"), "y1": ("y1", "ymax"),
}

ORIGEM_RECORTE, ORIGEM_USUARIO, ORIGEM_MODELO, ORIGEM_NENHUMA = (
    "recorte fornecido", "marcada pelo usuário", "detectada pelo modelo", "nenhuma lesão encontrada")
NOMES_DENS = {1: "1 (A)", 2: "2 (B)", 3: "3 (C)", 4: "4 (D)"}
ORDEM_DENS = [*NOMES_DENS.values(), "sem densidade"]
TITULO_DENS = "Densidade BI-RADS: 1 (A) adiposa … 4 (D) extremamente densa"


def eixo_contagem(maximo):
    """Eixo de contagem só com números inteiros (o Vega arredonda marcas fracionárias e repete rótulos)."""
    maximo = max(1, int(maximo)) if pd.notna(maximo) else 1
    passo = max(1, -(-maximo // 8))
    return alt.Axis(values=list(range(0, maximo + passo, passo)), format="d")
COR_PATOLOGIA = alt.Scale(domain=["Benigno", "Maligno"], range=["#2e7d32", "#c62828"])
COR_ACHADO = alt.Scale(domain=["Massa/Nódulo", "Microcalcificação"], range=["#1565c0", "#ef6c00"])

# o que cada coluna do resultado significa (tabela e legenda)
LEGENDA = {
    "amostra": "Identificador da lesão classificada.",
    "mama": "Mama (paciente + lado) a que a lesão pertence.",
    "vista": "Incidência da imagem: CC (crânio-caudal) e/ou MLO (médio-lateral oblíqua).",
    "origem_lesao": "De onde veio a lesão: recorte fornecido pelo usuário; caixa marcada pelo usuário na mamografia "
                    "completa; caixa detectada pelo modelo; ou nenhuma lesão encontrada na imagem.",
    "imagens_usadas": "Imagens que entraram na classificação (recorte CC/MLO, análise multivista ou a mamografia "
                      "completa de onde a lesão foi recortada).",
    "patologia": "Benigno ou Maligno, conforme o limiar escolhido.",
    "conf_patologia": "Confiança na patologia indicada (50% = dúvida total, 100% = certeza).",
    "p_maligno": "Probabilidade de malignidade (0 a 1).",
    "achado": "Tipo de achado: massa/nódulo ou microcalcificação.",
    "conf_achado": "Confiança no tipo de achado.",
    "densidade": "Densidade mamária BI-RADS (1 = A, adiposa … 4 = D, extremamente densa).",
    "densidade_origem": "Se a densidade foi informada pelo usuário (planilha) ou inferida pelo modelo.",
    "concordancia": "Comparação entre a marcação do usuário e a detecção do modelo na mesma imagem.",
    "score_deteccao": "Confiança do detector na caixa (só para lesões detectadas pelo modelo).",
}


# -----------------------------------------------------------------------------
# utilidades
# -----------------------------------------------------------------------------

def mostrar(fn, *args, alt=None, **kw):
    """Chama um elemento do Streamlit com largura total e nome acessível, compatível com versões antigas."""
    try:
        return fn(*args, width="stretch", alt=alt, **kw) if alt else fn(*args, width="stretch", **kw)
    except TypeError:
        return fn(*args, use_container_width=True, **kw)


def contar_imagens(pasta: Path, limite=5000):
    n = 0
    try:
        for f in pasta.rglob("*"):
            if f.suffix.lower() in EXT_IMG and f.is_file():
                n += 1
                if n >= limite:
                    break
    except Exception:
        pass
    return n


@st.cache_data(ttl=60, show_spinner=False)
def listar_imagens(pasta: str, limite=3000):
    """Arquivos de imagem da pasta (caminhos relativos), para escolher na planilha."""
    base = Path(pasta)
    out = []
    try:
        for f in sorted(base.rglob("*")):
            if f.suffix.lower() in EXT_IMG and f.is_file():
                out.append(f.relative_to(base).as_posix())
                if len(out) >= limite:
                    break
    except Exception:
        pass
    return out


def listar_subpastas(pasta: Path):
    try:
        return sorted(p for p in pasta.iterdir() if p.is_dir() and not p.name.startswith("."))
    except Exception:
        return []


def checar_modelo(pasta_modelo: Path):
    """Devolve (ok, mensagens) conferindo os arquivos que o maria.py precisa."""
    msgs, ok = [], True
    if not (pasta_modelo / "maria.py").is_file():
        msgs.append("❌ falta maria.py")
        ok = False
    achado = pasta_modelo / "modelo_achado"
    pat = pasta_modelo / "modelo_patologia"
    if not achado.is_dir() or not list(achado.glob("*.pth")):
        msgs.append("❌ modelo_achado/ sem arquivos .pth")
        ok = False
    faltam = [k for k in range(1, 6) if not (pat / f"mammoclip512_dobra_{k}.pth").is_file()]
    if faltam:
        msgs.append(f"❌ modelo_patologia/ sem as dobras {faltam}")
        ok = False
    det = pasta_modelo / "modelo_deteccao"
    if not ((det / "detector_lesao.pth").is_file() and (det / "detector_lesao.json").is_file()):
        msgs.append("⚠️ modelo_deteccao/ sem o detector: o modelo não vai procurar lesões nas mamografias completas")
    dens = pasta_modelo / "modelo_densidade"
    if not ((dens / "densidade.pth").is_file() and (dens / "densidade.json").is_file()):
        msgs.append("⚠️ modelo_densidade/ sem o classificador: a densidade só virá da planilha")
    return ok, msgs


def ler_cinza(caminho: Path):
    """PNG/JPG/DICOM -> PIL cinza (percentis 1-99) na resolução original. Retorna None se falhar."""
    try:
        if caminho.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return Image.open(caminho).convert("L")
        import pydicom
        ds = pydicom.dcmread(str(caminho))
        a = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a
        lo, hi = np.percentile(a, [1, 99])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8))
    except Exception:
        return None


CORES_CAIXA = {"selecionada": (230, 40, 40), ORIGEM_USUARIO: (40, 180, 90), ORIGEM_MODELO: (255, 200, 0)}


def carregar_imagem(caminho: Path, caixas=(), lado=700):
    """Imagem reduzida para visualização, com as caixas desenhadas.
    caixas: [(x0, y0, x1, y1, rótulo, origem, destacar)] em pixels da imagem original."""
    img = ler_cinza(caminho)
    if img is None:
        return None
    if not caixas:
        img.thumbnail((lado, lado))
        return img
    fator = min(1.0, lado / max(img.size))
    img = img.resize((max(1, round(img.width * fator)), max(1, round(img.height * fator)))).convert("RGB")
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1, rotulo, origem, destacar in sorted(caixas, key=lambda c: c[6]):
        cor = CORES_CAIXA["selecionada"] if destacar else CORES_CAIXA.get(origem, (255, 200, 0))
        x0, y0, x1, y1 = (v * fator for v in (x0, y0, x1, y1))
        m = 6   # um pouco de folga para a caixa não cobrir a borda da lesão
        d.rectangle([x0 - m, y0 - m, x1 + m, y1 + m], outline=cor, width=4 if destacar else 2)
        d.text((x0 - m, max(0, y0 - m - 12)), rotulo, fill=cor)
    return img


def rodar(script: str, pasta_modelo: Path, pasta_imagens: Path, extra, saida_csv: Path, area_log):
    """Roda `python script --pasta ...` dentro de pasta_modelo, mostrando o log ao vivo."""
    cmd = [sys.executable, "-u", script, "--pasta", str(pasta_imagens), "--saida_csv", str(saida_csv), *extra]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")   # Windows: evita cp1252
    proc = subprocess.Popen(cmd, cwd=str(pasta_modelo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", text=True, bufsize=1, env=env)
    linhas = []
    for linha in proc.stdout:
        for parte in linha.replace("\r", "\n").split("\n"):
            parte = parte.rstrip()
            if parte and "it/s" not in parte and "s/it" not in parte:
                linhas.append(parte)
        area_log.code("\n".join(linhas[-25:]), language="text")
    proc.wait()
    return proc.returncode, "\n".join(linhas)


def csv_excel(df):
    """CSV no formato do Excel em português (";" e vírgula decimal, UTF-8 com BOM): abre direto no duplo clique.
    O maria.py e o dashboard leem esse formato de volta sem problema."""
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):      # texto (object no pandas 2, "str" no pandas 3)
            df[c] = df[c].map(lambda v: v.replace(".", ",") if isinstance(v, str) and re.fullmatch(r"\d+\.\d+%?", v)
                              else v)
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def estilo_patologia(v):
    if v == "Maligno":
        return "background-color:#fde2e2;color:#8a1c1c;font-weight:600"
    if v == "Benigno":
        return "background-color:#e3f4e8;color:#1d5f33;font-weight:600"
    return ""


# ---- planilha do médico ----

def _chave_coluna(nome):
    nome = str(nome).strip().lower().replace(" ", "_")
    for a, b in (("ã", "a"), ("á", "a"), ("â", "a"), ("é", "e"), ("ê", "e"), ("í", "i"), ("ó", "o"), ("ô", "o"),
                 ("õ", "o"), ("ú", "u"), ("ç", "c")):
        nome = nome.replace(a, b)
    return nome


def planilha_vazia():
    return pd.DataFrame({c: pd.Series(dtype="float" if c in ("x0", "y0", "x1", "y1") else "object")
                         for c in COLUNAS_PLANILHA})


def padronizar_planilha(df):
    """Qualquer CSV do usuário -> DataFrame com as colunas padrão (nomes alternativos aceitos)."""
    df = df.rename(columns={c: _chave_coluna(c) for c in df.columns})
    out = planilha_vazia().iloc[0:0].copy()
    for campo, nomes in SINONIMOS.items():
        col = next((n for n in nomes if n in df.columns), None)
        out[campo] = df[col] if col is not None else None
    for c in ("x0", "y0", "x1", "y1"):
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    for c in ("mama", "densidade", "vista", "arquivo_completa", "arquivo_recorte", "lesao"):
        out[c] = out[c].astype("string").str.strip().replace({"nan": pd.NA, "": pd.NA}).astype("object")
        out[c] = out[c].where(out[c].notna(), None)
    out["densidade"] = out["densidade"].map(lambda v: str(v).upper().replace(".0", "") if v else None)
    out["vista"] = out["vista"].map(lambda v: str(v).upper() if v and str(v).upper() in ("CC", "MLO") else None)
    return out.reset_index(drop=True)


def ler_planilha_bytes(dados: bytes):
    for cod in ("utf-8-sig", "cp1252"):
        try:
            texto = dados.decode(cod)
            break
        except UnicodeDecodeError:
            continue
    primeira = texto.splitlines()[0] if texto.strip() else ""
    sep = ";" if primeira.count(";") > primeira.count(",") else ","
    return padronizar_planilha(pd.read_csv(io.StringIO(texto), sep=sep, dtype=str, keep_default_na=False))


def gerar_planilha_da_pasta(pasta: Path):
    """Rascunho da planilha a partir das imagens da pasta (uma linha por mama + vista)."""
    padrao_cbis = re.compile(r"P_(\d+)_(LEFT|RIGHT)_(CC|MLO)")
    linhas = {}
    for rel in listar_imagens(str(pasta)):
        f = pasta / rel
        nome = f.stem.lower()
        tipo, mama, vista = None, None, None
        if f.suffix.lower() == ".dcm":
            try:
                import pydicom
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                desc = str(getattr(ds, "SeriesDescription", "")).lower()
                if "mask" in desc:
                    continue
                tipo = "recorte" if "crop" in desc else "completa"
                cb = padrao_cbis.search(str(getattr(ds, "PatientID", "")))
                if cb:
                    mama, vista = f"P{cb[1]}_{cb[2]}", cb[3]
                else:
                    mama = f"{getattr(ds, 'PatientID', f.stem)}_{getattr(ds, 'ImageLaterality', '')}".strip("_")
                    vista = str(getattr(ds, "ViewPosition", "") or "").upper() or None
            except Exception:
                continue
        else:
            m = re.match(r"(.+?)_(recorte|completa)(?:_(cc|mlo))?$", nome)
            if not m:
                continue
            mama, tipo, vista = f.stem[: len(m.group(1))], m.group(2), (m.group(3) or "").upper() or None
        chave = (mama, vista)
        lin = linhas.setdefault(chave, {"mama": mama, "vista": vista})
        lin[f"arquivo_{tipo}"] = rel
    df = pd.DataFrame(list(linhas.values()))
    return padronizar_planilha(df) if len(df) else planilha_vazia()


def ajustar_caminhos(df, arquivos_pasta):
    """Arquivo citado só pelo nome -> caminho relativo na pasta (quando o nome é único), para a lista de escolha."""
    por_nome = {}
    for rel in arquivos_pasta:
        por_nome.setdefault(Path(rel).name.lower(), []).append(rel)
    df = df.copy()
    for c in ("arquivo_completa", "arquivo_recorte"):
        df[c] = df[c].map(lambda v: (por_nome[Path(str(v).replace("\\", "/")).name.lower()][0]
                                     if v and v not in arquivos_pasta
                                     and len(por_nome.get(Path(str(v).replace("\\", "/")).name.lower(), [])) == 1
                                     else v))
    return df


def nova_planilha(df):
    """Troca a planilha-base do editor (o editor é recriado com uma chave nova)."""
    st.session_state.planilha_base = df
    st.session_state.planilha_versao = st.session_state.get("planilha_versao", 0) + 1


def densidade_txt(v):
    try:
        return NOMES_DENS[int(v)]
    except (TypeError, ValueError, KeyError):
        return "sem densidade"


def pct(v):
    try:
        return float(str(v).replace("%", ""))
    except ValueError:
        return np.nan


# ---- banco de anotações e retreino ----

ANOTACOES = AQUI / "anotacoes"
COLUNAS_ANOTACAO = ["id", "rodada", "data", "amostra", "mama", "vista", "origem_lesao", "imagem", "imagem_original",
                    "x0", "y0", "x1", "y1", "patologia_modelo", "achado_modelo", "densidade_modelo",
                    "densidade_origem", "lesao_real", "patologia_confirmada", "achado_confirmado",
                    "exame_confirmacao", "densidade_confirmada", "observacao"]
CAMPOS_REVISAO = ["lesao_real", "patologia_confirmada", "achado_confirmado", "exame_confirmacao",
                  "densidade_confirmada", "observacao"]
EXAMES_CONFIRMACAO = ["Biópsia (anatomopatológico)", "Ressonância magnética", "Ultrassonografia",
                      "Outro exame padrão-ouro"]
AVISO_CONFIRMACAO = ("Os resultados de **patologia** e **achado** do MARIA são apenas apoio e devem ser confirmados "
                     "por exames adicionais — **biópsia (anatomopatológico)**, **ressonância magnética** ou outro "
                     "exame considerado **padrão-ouro** na literatura — antes de qualquer conduta e antes de serem "
                     "registrados como confirmados. Só confirmações com o exame informado são usadas no retreino.")


def ler_banco():
    """Banco de anotações sempre com todas as colunas (um banco antigo ou importado pode não ter alguma)."""
    arq = ANOTACOES / "anotacoes.csv"
    if arq.is_file():
        b = pd.read_csv(arq, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        return b.reindex(columns=COLUNAS_ANOTACAO, fill_value="").astype(str)
    return pd.DataFrame(columns=COLUNAS_ANOTACAO)


def processo_vivo(pid):
    """O processo do retreino ainda existe? (No Windows NÃO usar os.kill(pid, 0): lá isso envia Ctrl+C.)"""
    try:
        if os.name == "nt":
            r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True,
                               errors="replace", timeout=10)
            return str(int(pid)) in r.stdout
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def salvar_anotacoes(revisao, carimbo):
    """Grava as linhas revisadas no banco (anotacoes/anotacoes.csv) e copia as imagens para anotacoes/imagens."""
    import hashlib
    import shutil
    (ANOTACOES / "imagens").mkdir(parents=True, exist_ok=True)
    revisadas = revisao[revisao[CAMPOS_REVISAO].fillna("").astype(str).apply(lambda c: c.str.strip()).ne("").any(axis=1)]
    banco = ler_banco()
    novas, sem_imagem = [], 0
    for r in revisadas.to_dict("records"):
        origem = Path(str(r.get("imagem") or ""))
        rel = ""
        if origem.is_file():
            nome = hashlib.sha1(str(origem.resolve()).encode()).hexdigest()[:16] + origem.suffix.lower()
            destino = ANOTACOES / "imagens" / nome
            if not destino.exists():
                shutil.copy2(origem, destino)
            rel = f"imagens/{nome}"
        else:
            sem_imagem += 1
        linha = {c: "" for c in COLUNAS_ANOTACAO}
        linha.update({
            "id": f"{carimbo}|{r['amostra']}", "rodada": carimbo, "data": time.strftime("%Y-%m-%d %H:%M"),
            "amostra": r["amostra"], "mama": r.get("mama") or "", "vista": r.get("vista") or "",
            "origem_lesao": r.get("origem_lesao") or "", "imagem": rel, "imagem_original": str(origem),
            "patologia_modelo": r.get("patologia") or "", "achado_modelo": r.get("achado") or "",
            "densidade_modelo": "" if pd.isna(r.get("densidade")) else str(r.get("densidade")),
            "densidade_origem": r.get("densidade_origem") or "",
        })
        for c in ("x0", "y0", "x1", "y1"):
            linha[c] = "" if pd.isna(r.get(c)) else str(int(r[c]))
        for c in CAMPOS_REVISAO:
            v = r.get(c)
            linha[c] = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
        novas.append(linha)
    if novas:
        novas = pd.DataFrame(novas)
        banco = pd.concat([banco[~banco["id"].isin(novas["id"])], novas], ignore_index=True)
        ANOTACOES.mkdir(exist_ok=True)
        banco[COLUNAS_ANOTACAO].to_csv(ANOTACOES / "anotacoes.csv", index=False, encoding="utf-8-sig")
    return len(novas), sem_imagem


@st.cache_data(ttl=20, show_spinner=False)
def situacao_retreino(pasta_modelo: str, versao_banco: float):
    """Pergunta ao retreinar.py se esta máquina tem os dados completos e quantos casos há (JSON)."""
    try:
        r = subprocess.run([sys.executable, "retreinar.py", "--verificar"], cwd=pasta_modelo, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=120,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"disponivel": False, "faltando": [f"não consegui verificar: {e}"], "gpu": False,
                "casos": {"densidade": 0, "detector": 0, "achado": 0}, "minimo": {}}


def estado_retreino():
    arq = ANOTACOES / "retreino_estado.json"
    if not arq.is_file():
        return None
    est = json.loads(arq.read_text(encoding="utf-8"))
    log = Path(est["log"]).read_text(encoding="utf-8", errors="replace") if Path(est["log"]).is_file() else ""
    est["texto"] = log
    est["terminou"] = "relatório:" in log or "❌" in log or "Traceback" in log
    # retreino que morreu no meio (computador desligado, falta de memória…): não pode travar o botão para sempre
    est["interrompido"] = not est["terminou"] and not processo_vivo(est.get("pid", -1))
    if est["interrompido"]:
        est["terminou"] = True
    return est


# -----------------------------------------------------------------------------
# barra lateral
# -----------------------------------------------------------------------------

st.sidebar.title("🩻 MARIA")
st.sidebar.caption("Modelo de Análise Radiomamográfica por Inteligência Artificial · protótipo de pesquisa")

pasta_modelo = Path(st.sidebar.text_input("Pasta do modelo", value=str(AQUI / "modelo"))).expanduser()
ok_modelo, msgs = checar_modelo(pasta_modelo)
if ok_modelo:
    st.sidebar.success("Modelo encontrado", icon=":material/check_circle:")
for m in msgs:
    (st.sidebar.warning if m.startswith("⚠️") else st.sidebar.error)(m)

st.sidebar.divider()
st.sidebar.warning("Não é dispositivo médico. Não usar para decisão clínica.")

# -----------------------------------------------------------------------------
# página principal
# -----------------------------------------------------------------------------

st.title("MARIA - Modelo de Análise Radiomamográfica por Inteligência Artificial")

if "pasta_img" not in st.session_state:
    st.session_state.pasta_img = str(AQUI)
if "planilha_base" not in st.session_state:
    nova_planilha(planilha_vazia())

# -----------------------------------------------------------------------------
# página "Sobre o modelo"
# -----------------------------------------------------------------------------

def pagina_sobre():
    st.subheader("Como o MARIA funciona")
    st.markdown("O MARIA combina **quatro modelos** de inteligência artificial. Cada imagem segue um caminho "
                "conforme o tipo (recorte da lesão ou mamografia completa):")
    st.mermaid_chart("""
flowchart LR
    E["Pasta de imagens<br/>(DICOM, PNG, JPG)<br/>+ planilha do médico"] --> R["Recorte da lesão"]
    E --> C["Mamografia completa"]
    C --> D["Detector de lesões<br/>(Faster R-CNN)"]
    C --> M["Lesões marcadas<br/>pelo médico"]
    C --> DE["Densidade BI-RADS<br/>(ResNet-50)"]
    D --> RC["Recorte da caixa<br/>(+20 px)"]
    M --> RC
    R --> P["Patologia<br/>benigno × maligno<br/>(5 × Mammo-CLIP)"]
    RC --> P
    R --> A["Achado<br/>massa × microcalcificação<br/>(ResNet-50 + multivista)"]
    RC --> A
    DE --> AV["Aviso de confiabilidade<br/>por densidade"]
    P --> S["Resultado por lesão"]
    A --> S
    AV --> S
""", alt="Fluxo do MARIA: recortes vão direto para patologia e achado; mamografias completas passam pelo detector "
         "e pela densidade, e cada lesão encontrada ou marcada é recortada e classificada")

    st.subheader("Arquitetura")
    st.table(pd.DataFrame([
        {"Componente": "Patologia (benigno × maligno)",
         "Arquitetura": "Conjunto de 5 redes EfficientNet-B2 com o codificador de imagem do Mammo-CLIP "
                        "(pré-treinado em mamografias), ajustadas no CBIS-DDSM",
         "Entrada": "Recorte da lesão em tela de 512 × 512 px, escala preservada, contraste por percentis 1–99",
         "Detalhes": "Média das 5 redes e de 5 variações da imagem (original, espelhamentos, ±10°); "
                     "limiar padrão 0,50"},
        {"Componente": "Achado (massa × microcalcificação)",
         "Arquitetura": "ResNet-50 (pré-treino ImageNet) ajustada em recortes do CBIS-DDSM; ResNet-50 "
                        "multivista (siamesa, CC + MLO) quando há as duas vistas da mesma lesão",
         "Entrada": "Recorte da lesão redimensionado para 224 × 224 px",
         "Detalhes": "Média do modelo do recorte (5 variações) e do multivista"},
        {"Componente": "Detector de lesões",
         "Arquitetura": "Faster R-CNN com ResNet-50 FPN v2 (pré-treino COCO), uma classe: lesão",
         "Entrada": "Mamografia completa reduzida ao lado maior de 1.536 px",
         "Detalhes": "Score mínimo padrão 0,50; cada caixa é recortada com 20 px de margem e classificada"},
        {"Componente": "Densidade mamária (BI-RADS 1–4)",
         "Arquitetura": "ResNet-50 (pré-treino ImageNet), 4 classes",
         "Entrada": "Mamografia completa em 768 × 768 px",
         "Detalhes": "Média de 2 variações (original e espelhada) e das vistas da mesma mama; a densidade "
                     "informada pelo médico tem prioridade"},
    ]).set_index("Componente"))

    st.subheader("Bases de dados")
    st.table(pd.DataFrame([
        {"Base": "CBIS-DDSM (TCIA)",
         "Tipo": "Mamografia em filme digitalizado (EUA)",
         "Conteúdo": "3.032 mamografias completas, 3.568 lesões com contorno (1.696 massas, 1.872 "
                     "calcificações), patologia confirmada, densidade BI-RADS",
         "Uso no MARIA": "Treino e teste de todos os modelos (teste oficial separado por paciente)",
         "Licença": "CC BY 3.0"},
        {"Base": "CDD-CESM (TCIA)",
         "Tipo": "Mamografia digital (imagens de baixa energia de mamografia com contraste; Egito)",
         "Conteúdo": "1.003 imagens de 326 pacientes: 341 normais, 331 benignas, 331 malignas; densidade ACR",
         "Uso no MARIA": "Treino e teste do detector (inclui exames normais) e da densidade; "
                         "validação externa da patologia",
         "Licença": "CC BY 4.0"},
        {"Base": "Mammo-CLIP (pesos pré-treinados)",
         "Tipo": "Modelo de linguagem-imagem para mamografia",
         "Conteúdo": "Codificador de imagem pré-treinado pelos autores com mamografias de rastreamento e laudos",
         "Uso no MARIA": "Ponto de partida das 5 redes de patologia",
         "Licença": "Ver o repositório dos autores"},
    ]).set_index("Base"))
    st.caption("Treino e teste sempre separados por paciente. Bases públicas: os resultados podem ser diferentes em "
               "outros equipamentos e populações.")

    st.subheader("Resultados medidos")
    st.caption("Todos em pacientes que o modelo avaliado não viu no treino.")
    with st.container(border=True):
        st.markdown("**Patologia** — validação cruzada sem vazamento, 2.781 lesões de 1.217 pacientes do CBIS-DDSM")
        with st.container(horizontal=True):
            st.metric("Acurácia", "76,3%", border=True)
            st.metric("AUC", "0,840", border=True)
            st.metric("Sensibilidade", "73,2%", border=True)
            st.metric("Especificidade", "78,5%", border=True)
        dens_pat = pd.DataFrame({"densidade": ["1 (A)", "2 (B)", "3 (C)", "4 (D)"], "lesões": [446, 1047, 858, 430],
                                 "AUC": [0.892, 0.854, 0.837, 0.747], "acurácia (%)": [81.2, 78.1, 76.5, 66.5],
                                 "sensibilidade (%)": [72.8, 74.9, 71.0, 73.6],
                                 "especificidade (%)": [86.4, 80.6, 80.4, 62.2]})
        c1, c2 = st.columns([3, 2])
        with c1:
            mostrar(st.dataframe, dens_pat, hide_index=True, alt="Desempenho da patologia por densidade mamária")
        with c2:
            graf = alt.Chart(dens_pat).mark_bar(color="#c62828", clip=True).encode(
                x=alt.X("densidade:N", title="Densidade BI-RADS", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.5, 1], clamp=True), title="AUC"),
                y2=alt.datum(0.5),
                tooltip=["densidade", "AUC", "lesões"]).properties(height=200, title="AUC da patologia por densidade")
            mostrar(st.altair_chart, graf, alt="AUC da patologia cai de 0,89 na densidade 1 para 0,75 na densidade 4")
        st.caption("A acurácia cai em mama densa (diferença de AUC densa × não densa: −0,059, IC95% −0,10 a −0,02). "
                   "Validação externa no CDD-CESM (outro hospital, mamografia digital): AUC entre 0,83 e 0,91, conforme "
                   "o recorte da análise. Os modelos finais também foram treinados com o teste oficial do CBIS-DDSM; "
                   "por isso a estimativa acima é a da validação cruzada.")
    with st.container(border=True):
        st.markdown("**Achado** — teste oficial do CBIS-DDSM, 704 lesões (378 massas, 326 calcificações)")
        with st.container(horizontal=True):
            st.metric("Acurácia", "93,2%", border=True)
            st.metric("AUC", "0,982", border=True)
            st.metric("Microcalcificações reconhecidas", "91,7%", border=True)
            st.metric("Massas reconhecidas", "94,4%", border=True)
        st.caption("Medido só com o modelo do recorte (sem o multivista). Ainda não validado em mamografia digital.")
    with st.container(border=True):
        st.markdown("**Detector de lesões** — teste: CBIS-DDSM (676 imagens, 787 lesões) e CDD-CESM (269 imagens "
                    "com 740 lesões + 131 exames normais)")
        froc = pd.DataFrame({"caixas falsas por imagem": [0.25, 0.5, 1, 2, 4] * 2,
                             "lesões encontradas (%)": [47.1, 57.7, 64.8, 72.2, 78.9, 44.6, 51.5, 59.5, 68.1, 75.8],
                             "base": ["CBIS-DDSM (filme)"] * 5 + ["CDD-CESM (digital)"] * 5})
        c1, c2 = st.columns([3, 2])
        with c1:
            graf = alt.Chart(froc).mark_line(point=True).encode(
                x=alt.X("caixas falsas por imagem:Q", scale=alt.Scale(type="log"),
                        axis=alt.Axis(values=[0.25, 0.5, 1, 2, 4])),
                y=alt.Y("lesões encontradas (%):Q", scale=alt.Scale(domain=[30, 90])),
                color=alt.Color("base:N", title="Base"), tooltip=list(froc.columns)
            ).properties(height=240, title="Curva FROC: lesões encontradas × caixas falsas")
            mostrar(st.altair_chart, graf, alt="Curva FROC do detector: cerca de 65% das lesões com 1 caixa falsa por "
                                               "imagem em filme e 60% em digital")
        with c2:
            st.table(pd.DataFrame({
                "score mínimo": ["0,30", "0,50 (padrão)", "0,70"],
                "lesões achadas (filme / digital)": ["64% / 62%", "59% / 55%", "50% / 49%"],
                "caixas falsas por exame normal": ["1,7", "0,9", "0,4"],
            }).set_index("score mínimo"))
        st.caption("No padrão (0,50), 59% dos exames normais recebem pelo menos uma caixa falsa (0,93 caixa por "
                   "exame), e mais em mama densa: 1,11 caixa por exame normal denso × 0,51 não denso.")
    with st.container(border=True):
        st.markdown("**Densidade mamária** — teste: 1.050 mamografias (674 do CBIS-DDSM, 376 do CDD-CESM)")
        with st.container(horizontal=True):
            st.metric("Densa × não densa", "86,5%", border=True)
            st.metric("Categoria exata", "73,2%", border=True)
            st.metric("Concordância (kappa)", "0,68", border=True)
            st.metric("Densidade 4 reconhecida", "76%", border=True)
        st.caption("Por base: filme — kappa 0,69, densidade 4 reconhecida em 89%; digital — kappa 0,62, densidade 4 "
                   "reconhecida em só 37% (poucos exemplos de densidade 4 em mamografia digital no treino).")

    # ---- gráficos detalhados (lidos de modelo/desempenho.json) ----
    st.subheader("Gráficos detalhados")
    arq_desemp = pasta_modelo / "desempenho.json"
    if not arq_desemp.is_file():
        st.info("Arquivo modelo/desempenho.json não encontrado: os gráficos detalhados não estão disponíveis.",
                icon=":material/info:")
    else:
        D = json.loads(arq_desemp.read_text(encoding="utf-8"))
        st.caption(f"Gerado em {D['gerado_em']} a partir dos registros de treino e das avaliações em pacientes fora "
                   "do treino. **Curva de aprendizado:** como a perda e o desempenho de validação evoluem a cada "
                   "época (curvas estáveis no fim = treino convergiu; validação piorando enquanto o treino melhora = "
                   "sobreajuste). **Matriz de confusão:** linhas = verdade, colunas = previsão do modelo.")

        def grafico_matriz(mat, classes, titulo):
            linhas = []
            for i, verdade in enumerate(classes):
                tot = max(1, sum(mat[i]))
                for j, prev in enumerate(classes):
                    linhas.append({"verdade": verdade, "previsão": prev, "n": mat[i][j],
                                   "pct": mat[i][j] / tot, "rótulo": f"{mat[i][j]}\n({mat[i][j] / tot:.0%})"})
            dm = pd.DataFrame(linhas)
            base = alt.Chart(dm).encode(x=alt.X("previsão:N", sort=classes, title="Previsão do modelo",
                                                axis=alt.Axis(labelAngle=0)),
                                        y=alt.Y("verdade:N", sort=classes, title="Verdade"))
            cel = base.mark_rect().encode(color=alt.Color("pct:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                                                          legend=None),
                                          tooltip=["verdade", "previsão", "n", alt.Tooltip("pct:Q", format=".0%")])
            txt = base.mark_text(fontSize=13, lineBreak="\n").encode(
                text="rótulo:N", color=alt.condition("datum.pct > 0.5", alt.value("white"), alt.value("black")))
            mostrar(st.altair_chart, (cel + txt).properties(title=titulo, height=60 * len(classes) + 40),
                    alt=f"Matriz de confusão: {titulo}")

        def grafico_roc(roc, auc, titulo, cor):
            dr = pd.DataFrame({"fpr": roc["fpr"], "tpr": roc["tpr"]})
            eixos = {"x": alt.X("fpr:Q", title="1 − especificidade (falsos positivos)", scale=alt.Scale(domain=[0, 1])),
                     "y": alt.Y("tpr:Q", title="Sensibilidade", scale=alt.Scale(domain=[0, 1]))}
            linha = alt.Chart(dr).mark_line(color=cor, strokeWidth=3).encode(**eixos)
            acaso = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(
                strokeDash=[4, 4], color="gray").encode(**eixos)
            mostrar(st.altair_chart, (acaso + linha).properties(title=f"{titulo} (AUC {auc:.3f})", height=300),
                    alt=f"Curva ROC: {titulo}, AUC {auc:.3f}; linha tracejada = acaso")

        def grafico_curvas(dc, metricas, titulo, cor_por=None, eixo_y="valor"):
            longa = dc.melt(id_vars=["epoca"] + ([cor_por] if cor_por else []), value_vars=list(metricas),
                            var_name="medida", value_name=eixo_y)
            longa["medida"] = longa["medida"].map(metricas)
            enc = {"x": alt.X("epoca:Q", title="Época", axis=alt.Axis(tickMinStep=1)),
                   "y": alt.Y(f"{eixo_y}:Q", title=eixo_y, scale=alt.Scale(zero=False)),
                   "color": alt.Color("medida:N", title=None),
                   "tooltip": ["epoca", "medida", alt.Tooltip(f"{eixo_y}:Q", format=".3f")]}
            if cor_por:
                enc["detail"] = f"{cor_por}:N"
                enc["tooltip"] = [cor_por] + enc["tooltip"]
            mostrar(st.altair_chart, alt.Chart(longa).mark_line(point=True).encode(**enc)
                    .properties(title=titulo, height=260), alt=f"Curva de aprendizado: {titulo}")

        t_pat, t_ach, t_det, t_den = st.tabs(["Patologia", "Achado", "Detector de lesões", "Densidade"])
        with t_pat:
            P = D["patologia"]
            cp = pd.DataFrame(P["curvas"])
            cp["dobra"] = "dobra " + cp["dobra"].astype(str)
            c1, c2 = st.columns(2)
            with c1:
                mostrar(st.altair_chart, alt.Chart(cp).mark_line(point=True).encode(
                    x=alt.X("epoca:Q", title="Época"), y=alt.Y("perda_treino:Q", title="Perda (treino)",
                                                               scale=alt.Scale(zero=False)),
                    color=alt.Color("dobra:N", title="Dobra"), tooltip=["dobra", "epoca", "perda_treino"])
                    .properties(title="Curva de aprendizado: perda de treino", height=260),
                    alt="Perda de treino da patologia caindo ao longo das 15 épocas nas 5 dobras")
            with c2:
                mostrar(st.altair_chart, alt.Chart(cp).mark_line(point=True).encode(
                    x=alt.X("epoca:Q", title="Época"), y=alt.Y("auc_val:Q", title="AUC (validação)",
                                                               scale=alt.Scale(zero=False)),
                    color=alt.Color("dobra:N", title="Dobra"), tooltip=["dobra", "epoca", "auc_val", "acc_val"])
                    .properties(title="Curva de aprendizado: AUC de validação", height=260),
                    alt="AUC de validação da patologia subindo até cerca de 0,82 a 0,84 nas 5 dobras")
            st.caption("5 redes treinadas em dobras diferentes (validação cruzada por paciente). O codificador começa "
                       "congelado (épocas 1–2) e é liberado na época 3, quando o desempenho dá o salto.")
            c1, c2 = st.columns(2)
            with c1:
                grafico_matriz(P["matriz"], P["classes"], f"Patologia — {P['n']} lesões (limiar 0,50)")
            with c2:
                grafico_roc(P["roc"], P["auc"], "Curva ROC da patologia", "#c62828")
            c1, c2 = st.columns(2)
            with c1:
                dcal = pd.DataFrame(P["calibracao"])
                pts = alt.Chart(dcal).mark_line(point=True, color="#c62828").encode(
                    x=alt.X("p_media:Q", title="P(maligno) prevista (média da faixa)", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("maligno_observado:Q", title="Proporção realmente maligna", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["faixa", alt.Tooltip("p_media:Q", format=".2f"),
                             alt.Tooltip("maligno_observado:Q", format=".2f"), "n"])
                ideal = alt.Chart(pd.DataFrame({"p_media": [0, 1], "maligno_observado": [0, 1]})).mark_line(
                    strokeDash=[4, 4], color="gray").encode(x="p_media:Q", y="maligno_observado:Q")
                mostrar(st.altair_chart, (ideal + pts).properties(title="Calibração da P(maligno)", height=280),
                        alt="Calibração: quanto a probabilidade prevista bate com a proporção real de malignos")
                st.caption("Pontos sobre a diagonal = a probabilidade é confiável como probabilidade.")
            with c2:
                dh = pd.DataFrame(P["histograma"])
                mostrar(st.altair_chart, alt.Chart(dh).mark_bar().encode(
                    x=alt.X("faixa:N", title="P(maligno) prevista", axis=alt.Axis(labelAngle=-40)),
                    xOffset=alt.XOffset("classe:N"), y=alt.Y("n:Q", title="Lesões"),
                    color=alt.Color("classe:N", title=None, scale=alt.Scale(
                        domain=["Benigno (verdade)", "Maligno (verdade)"], range=["#2e7d32", "#c62828"])),
                    tooltip=["faixa", "classe", "n"]).properties(title="Distribuição da P(maligno)", height=280),
                    alt="Distribuição da probabilidade de malignidade para lesões benignas e malignas")
                st.caption("Quanto mais separadas as barras verdes (à esquerda) e vermelhas (à direita), melhor.")
            tipos = pd.DataFrame([{"tipo de lesão": k, "lesões": v["n"], "AUC": round(v["auc"], 3),
                                   "acurácia (%)": round(100 * v["acc"], 1)} for k, v in P["por_tipo"].items()])
            st.markdown("**Por tipo de lesão**")
            mostrar(st.dataframe, tipos, hide_index=True, alt="Desempenho da patologia em massas e calcificações")
        with t_ach:
            A = D["achado"]
            ca = pd.DataFrame(A["curvas"])
            c1, c2 = st.columns(2)
            with c1:
                grafico_curvas(ca, {"perda_treino": "treino", "perda_val": "validação"},
                               "Curva de aprendizado: perda (treino × validação)", eixo_y="perda")
            with c2:
                grafico_curvas(ca, {"acc_treino": "treino", "acc_val": "validação"},
                               "Curva de aprendizado: acurácia do achado", eixo_y="acurácia")
            st.caption("O treino parou por parada antecipada (early stopping). Treino e validação próximos = sem "
                       "sobreajuste importante.")
            c1, c2 = st.columns(2)
            with c1:
                grafico_matriz(A["matriz"], A["classes"], f"Achado — teste oficial, {A['n']} lesões")
            with c2:
                grafico_roc(A["roc"], A["auc"], "Curva ROC do achado", "#1565c0")
        with t_det:
            T = D["detector"]
            ct = pd.DataFrame(T["curvas"])
            c1, c2 = st.columns(2)
            with c1:
                grafico_curvas(ct, {"perda_treino": "treino"}, "Curva de aprendizado: perda de treino",
                               eixo_y="perda")
            with c2:
                grafico_curvas(ct, {"sens_val_0.5fp": "0,5 caixa falsa/imagem", "sens_val_1fp": "1 caixa falsa/imagem",
                                    "sens_val_2fp": "2 caixas falsas/imagem"},
                               "Curva de aprendizado: lesões encontradas (validação)", eixo_y="sensibilidade")
            c1, c2 = st.columns(2)
            with c1:
                nm = pd.DataFrame([{"grupo": k, "caixas falsas por exame normal": v["fp_por_exame"], "exames": v["n"]}
                                   for k, v in T["normais"].items() if k != "limiar"])
                mostrar(st.altair_chart, alt.Chart(nm).mark_bar(color="#f9a825").encode(
                    x=alt.X("grupo:N", title=None, sort=["todos", "não densa (1-2)", "densa (3-4)"],
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("caixas falsas por exame normal:Q"), tooltip=["grupo", "caixas falsas por exame normal",
                                                                          "exames"])
                    .properties(title="Alarmes falsos em exames normais (limiar 0,50)", height=280),
                    alt="Caixas falsas por exame normal: 0,51 em mama não densa e 1,11 em mama densa")
            with c2:
                st.caption("O detector não tem matriz de confusão: ele procura regiões, não classifica imagens. O "
                           "equivalente é a curva FROC (em *Resultados medidos*, acima), que mostra quantas lesões ele "
                           "acha para cada quantidade de caixas falsas aceita. Ao lado, os alarmes falsos em exames "
                           "normais: o dobro em mama densa.")
        with t_den:
            E = D["densidade"]
            ce = pd.DataFrame(E["curvas"])
            c1, c2 = st.columns(2)
            with c1:
                grafico_curvas(ce, {"perda_treino": "treino"}, "Curva de aprendizado: perda de treino",
                               eixo_y="perda")
            with c2:
                grafico_curvas(ce, {"kappa_val": "concordância (kappa)", "densa_val": "densa × não densa",
                                    "dens4_val": "densidade 4 reconhecida"},
                               "Curva de aprendizado: validação", eixo_y="valor")
            c1, c2 = st.columns([3, 2])
            with c1:
                grafico_matriz(E["matriz"], E["classes"], "Densidade — teste, 1.050 mamografias")
            with c2:
                st.caption("Os erros ficam quase sempre na categoria vizinha (1↔2, 2↔3, 3↔4), como acontece também "
                           "entre radiologistas. Erros grosseiros (1↔3-4 ou 4↔1-2) são raros. A seleção da época foi "
                           "pela concordância na validação; a densidade 4 oscila muito entre épocas porque a "
                           "validação tem poucos casos dela.")

    st.subheader("Limitações e uso responsável")
    st.warning("**Protótipo de pesquisa — não é dispositivo médico.** Não substitui a avaliação do radiologista.\n\n"
               "- Patologia e achado devem ser confirmados por biópsia, ressonância ou outro exame padrão-ouro.\n"
               "- O detector aponta regiões para revisão; não serve para triagem automática (marca caixas falsas em "
               "cerca de metade dos exames normais e não acha todas as lesões).\n"
               "- O limiar 0,50 foi calibrado em filme digitalizado (CBIS-DDSM); em outros equipamentos o modelo "
               "tende a chamar mais lesões de malignas.\n"
               "- A confiabilidade cai em mama densa e em densidade 4 na mamografia digital.\n"
               "- Os exames normais usados para medir alarmes falsos são poucos (131) e de um único hospital.",
               icon=":material/info:")

    st.subheader("Versão dos modelos em uso")
    arquivos_modelo = [("Patologia (5 redes)", "modelo_patologia/mammoclip512_dobra_1.pth"),
                       ("Achado (recorte)", "modelo_achado/resnet50_multitask_mama_do_zero.pth"),
                       ("Achado (multivista)", "modelo_achado/resnet50_multivista.pth"),
                       ("Detector de lesões", "modelo_deteccao/detector_lesao.pth"),
                       ("Densidade", "modelo_densidade/densidade.pth")]
    linhas_v = []
    for nome, rel in arquivos_modelo:
        f = pasta_modelo / rel
        linhas_v.append({"modelo": nome, "arquivo": rel,
                         "última atualização": time.strftime("%d/%m/%Y %H:%M", time.localtime(f.stat().st_mtime))
                         if f.is_file() else "não encontrado"})
    mostrar(st.dataframe, pd.DataFrame(linhas_v), hide_index=True, alt="Arquivos e datas dos modelos em uso")
    st.caption("Se um modelo foi retreinado com as anotações do usuário (aba 3), a data muda e os números acima "
               "deixam de valer para ele; o relatório do retreino mostra o desempenho do modelo novo.")

    st.subheader("Referências")
    st.markdown(
        "- Lee RS et al. *A curated mammography data set for use in computer-aided detection and diagnosis research* "
        "(CBIS-DDSM). Scientific Data, 2017.\n"
        "- Khaled R et al. *Categorized contrast enhanced mammography dataset for diagnostic and artificial "
        "intelligence research* (CDD-CESM). Scientific Data, 2022.\n"
        "- Ghosh S et al. *Mammo-CLIP: a vision language foundation model to enhance data efficiency and robustness "
        "in mammography*. MICCAI, 2024.\n"
        "- Tan M, Le Q. *EfficientNet: rethinking model scaling for convolutional neural networks*. ICML, 2019.\n"
        "- He K et al. *Deep residual learning for image recognition* (ResNet). CVPR, 2016.\n"
        "- Ren S et al. *Faster R-CNN: towards real-time object detection with region proposal networks*. "
        "NeurIPS, 2015.\n"
        "- American College of Radiology. *ACR BI-RADS® Atlas*, 5ª edição, 2013.")


# -----------------------------------------------------------------------------
# resultados
# -----------------------------------------------------------------------------

def grafico_agrupado(les, x_campo, x_ordem, x_titulo, nome_x, campo, conf, escala, titulo, artigo):
    """Barras agrupadas (contagem de lesões por x e por campo) + confiança de cada lesão (pontos e média)."""
    dados = les[les[campo].isin(escala.domain) & les[x_campo].notna()].copy()
    if dados.empty:
        st.caption("Sem lesões classificadas para este gráfico.")
        return
    dados["confianca"] = dados[conf].map(pct)
    cor = alt.Color(f"{campo}:N", scale=escala, title=titulo)
    eixo_x = alt.X(f"{x_campo}:N", sort=x_ordem, title=x_titulo, axis=alt.Axis(labelAngle=0))
    barras = alt.Chart(dados).mark_bar().encode(
        x=eixo_x, xOffset=alt.XOffset(f"{campo}:N", sort=escala.domain), color=cor,
        y=alt.Y("count():Q", title="Lesões", axis=eixo_contagem(dados.groupby([x_campo, campo]).size().max())),
        tooltip=[alt.Tooltip(f"{x_campo}:N", title=nome_x.capitalize()), alt.Tooltip(f"{campo}:N", title=titulo),
                 alt.Tooltip("count():Q", title="Lesões")],
    ).properties(title=f"{titulo} por {nome_x}", height=280)
    pontos = alt.Chart(dados).mark_circle(size=70, opacity=0.65).encode(
        x=eixo_x, xOffset=alt.XOffset(f"{campo}:N", sort=escala.domain),
        y=alt.Y("confianca:Q", title=f"Confiança {artigo} {titulo.lower()} (%)", scale=alt.Scale(domain=[50, 100])),
        color=cor, tooltip=["amostra:N", alt.Tooltip(f"{campo}:N", title=titulo),
                            alt.Tooltip("confianca:Q", title="Confiança (%)", format=".1f")],
    )
    media = alt.Chart(dados).mark_tick(thickness=3, size=22, color="black").encode(
        x=eixo_x, xOffset=alt.XOffset(f"{campo}:N", sort=escala.domain), y="mean(confianca):Q",
        tooltip=[alt.Tooltip("mean(confianca):Q", title="Confiança média (%)", format=".1f")],
    )
    c1, c2 = st.columns(2)
    with c1:
        mostrar(st.altair_chart, barras, alt=f"Contagem de lesões por {nome_x} e {titulo.lower()}")
    with c2:
        mostrar(st.altair_chart, (pontos + media).properties(
            title=f"Confiança {artigo} {titulo.lower()} por {nome_x}", height=280),
            alt=f"Confiança {artigo} {titulo.lower()} de cada lesão por {nome_x}; traço preto = média")


# -----------------------------------------------------------------------------
# página "Executar": imagens e planilha, resultados, anotações e retreino
# -----------------------------------------------------------------------------

def pagina_executar():
    # controles que só fazem sentido ao executar (aparecem na barra lateral só nesta página)
    st.sidebar.divider()
    limiar = st.sidebar.slider("Limiar para 'Maligno'", 0.05, 0.95, 0.50, 0.05,
                               help="0,5 foi calibrado no CBIS-DDSM. Em outras bases, o modelo tende a marcar mais casos "
                                    "como malignos (sensibilidade alta, especificidade menor). Use limiar maior para ser menos sensível.")
    limiar_det = st.sidebar.slider(
        "Sensibilidade do detector (score mínimo da caixa)", 0.20, 0.90, 0.50, 0.05,
        help="Só vale para mamografias completas. Menor = encontra mais lesões, mas marca mais caixas falsas. "
             "No teste (lesões achadas em filme / digital; caixas falsas por exame normal): "
             "0,30 → 64% / 62%, 1,7 caixa; 0,50 → 59% / 55%, 0,9 caixa; 0,70 → 50% / 49%, 0,4 caixa.")
    aba_sel, aba_res, aba_ret = st.tabs(["1) Imagens e planilha", "2) Resultados", "3) Anotações e retreino"])

    with aba_sel:
        # ---- pasta de imagens ----
        st.subheader("Pasta de imagens")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            texto = st.text_input("Pasta com as imagens (PNG, JPG ou DICOM; busca também nas subpastas)",
                                  value=st.session_state.pasta_img, key="campo_pasta")
            if st.button("Pasta acima", icon=":material/arrow_upward:", key="pasta_acima"):
                st.session_state.pasta_img = str(Path(texto).expanduser().parent)
                st.rerun()
        if texto != st.session_state.pasta_img:
            st.session_state.pasta_img = texto
        atual = Path(st.session_state.pasta_img).expanduser()

        n_img = 0
        if not atual.is_dir():
            st.error(f"Pasta não encontrada: {atual}")
        else:
            n_img = contar_imagens(atual)
            st.caption(f"📂 {atual}  —  {n_img}{'+' if n_img >= 5000 else ''} imagem(ns) (incluindo subpastas)")
            subs = listar_subpastas(atual)
            if subs:
                nomes = ["(entrar em uma subpasta…)"] + [p.name for p in subs]
                ir = st.selectbox("Subpastas", nomes, index=0, key=f"sub_{atual}")
                if ir != nomes[0]:
                    st.session_state.pasta_img = str(atual / ir)
                    st.rerun()

        # ---- planilha do médico (editável) ----
        st.subheader("Planilha do médico (opcional)")
        st.caption("Uma linha por imagem ou lesão. Só **mama** é obrigatória. **Densidade** (1-4 ou A-D) e as **lesões "
                   "marcadas** (caixa x0, y0, x1, y1 em pixels da mamografia completa) têm prioridade sobre o modelo. "
                   "Arquivos citados aqui não precisam seguir nenhuma regra de nome. Mesma **mama** = mesma mama "
                   "(a densidade vale para todas as lesões dela).")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            arq = st.file_uploader("Carregar planilha (CSV)", type=["csv"], key="upload_planilha",
                                   help="Separador vírgula ou ponto e vírgula; UTF-8 ou o padrão do Excel no Windows.")
            if st.button("Gerar a partir da pasta", icon=":material/auto_awesome:", disabled=not atual.is_dir(),
                         key="gerar_planilha",
                         help="Cria um rascunho com uma linha por mama e vista, a partir dos arquivos da pasta."):
                nova_planilha(gerar_planilha_da_pasta(atual))
            if st.button("Limpar planilha", icon=":material/delete:", key="limpar_planilha"):
                nova_planilha(planilha_vazia())
        if arq is not None and st.session_state.get("planilha_arquivo_id") != arq.file_id:
            try:
                nova_planilha(ajustar_caminhos(ler_planilha_bytes(arq.getvalue()),
                                               listar_imagens(str(atual)) if atual.is_dir() else []))
                st.session_state.planilha_arquivo_id = arq.file_id
            except Exception as e:
                st.error(f"Não consegui ler a planilha: {e}")

        arquivos_pasta = listar_imagens(str(atual)) if atual.is_dir() else []
        base = st.session_state.planilha_base
        extras = sorted({v for c in ("arquivo_completa", "arquivo_recorte") for v in base[c].dropna()} - set(arquivos_pasta))
        opcoes_arquivo = arquivos_pasta + extras
        planilha = st.data_editor(
            base, key=f"editor_planilha_{st.session_state.planilha_versao}", num_rows="dynamic", hide_index=True,
            column_order=COLUNAS_PLANILHA,
            column_config={
                "mama": st.column_config.TextColumn("mama", help="Id da mama (paciente + lado). Obrigatória.",
                                                    required=True),
                "densidade": st.column_config.SelectboxColumn("densidade", options=["1", "2", "3", "4", "A", "B", "C", "D"],
                                                              help="Densidade BI-RADS do laudo (1-4 ou A-D)."),
                "vista": st.column_config.SelectboxColumn("vista", options=["CC", "MLO"]),
                "arquivo_completa": st.column_config.SelectboxColumn(
                    "arquivo da mamografia completa", options=opcoes_arquivo, width="large"),
                "arquivo_recorte": st.column_config.SelectboxColumn(
                    "arquivo do recorte", options=opcoes_arquivo, width="large"),
                "lesao": st.column_config.TextColumn("lesão", help="Id da lesão, quando a mama tem mais de uma."),
                **{c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d",
                                                    help="Caixa da lesão marcada pelo médico (pixels da imagem original).")
                   for c in ("x0", "y0", "x1", "y1")},
            },
            alt="Planilha do médico, editável",
        )
        planilha_valida = planilha[planilha["mama"].notna() & (planilha["mama"].astype(str).str.strip() != "")]
        nomes_pasta = {Path(a).name.lower() for a in arquivos_pasta}
        faltando = [v for c in ("arquivo_completa", "arquivo_recorte") for v in planilha_valida[c].dropna()
                    if v not in arquivos_pasta and Path(str(v).replace("\\", "/")).name.lower() not in nomes_pasta]
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"{len(planilha_valida)} linha(s) válida(s) · "
                       f"{int(planilha_valida['densidade'].notna().sum())} com densidade · "
                       f"{int(planilha_valida[['x0', 'y0', 'x1', 'y1']].notna().all(axis=1).sum())} lesão(ões) marcada(s)")
            st.download_button("Baixar planilha editada", csv_excel(planilha_valida),
                               file_name="planilha_medico.csv", mime="text/csv", icon=":material/download:",
                               disabled=planilha_valida.empty)
        if faltando:
            st.warning(f"{len(faltando)} arquivo(s) da planilha não estão nesta pasta: " + ", ".join(faltando[:5])
                       + (" …" if len(faltando) > 5 else ""), icon=":material/warning:")

        # ---- rodar ----
        st.divider()
        rodar_btn = st.button("Classificar", type="primary", icon=":material/play_arrow:", key="classificar",
                              disabled=not (ok_modelo and n_img > 0))
        if rodar_btn:
            SAIDAS.mkdir(exist_ok=True)
            carimbo = time.strftime("%Y%m%d_%H%M%S")
            resultado = {"pasta": str(atual), "carimbo": carimbo, "df": None, "log": {}}
            st.write("**MARIA** — carregando modelos e classificando…")
            area = st.empty()
            csv_saida = SAIDAS / f"results_{carimbo}.csv"
            extra = ["--limiar", f"{limiar}", "--limiar_deteccao", f"{limiar_det}"]
            if not planilha_valida.empty:
                csv_planilha = SAIDAS / f"planilha_{carimbo}.csv"
                planilha_valida.to_csv(csv_planilha, index=False, encoding="utf-8")
                extra += ["--planilha", str(csv_planilha)]
            cod, log = rodar("maria.py", pasta_modelo, atual, extra, csv_saida, area)
            resultado["log"]["maria"] = log
            if cod != 0 or not csv_saida.is_file():
                st.error("O MARIA terminou com erro. Veja o log acima.")
            else:
                df = pd.read_csv(csv_saida)
                df["p_maligno"] = pd.to_numeric(df["p_maligno"], errors="coerce")
                resultado["df"] = df
                st.success("Pronto! Veja a aba **2) Resultados**.")
            st.session_state.resultado = resultado

    # -----------------------------------------------------------------------------
    # anotações e retreino
    # -----------------------------------------------------------------------------

    with aba_ret:
        st.subheader("Banco de anotações")
        banco = ler_banco()
        arq_banco = ANOTACOES / "anotacoes.csv"
        versao_banco = arq_banco.stat().st_mtime if arq_banco.is_file() else 0.0
        with st.container(horizontal=True):
            st.metric("Lesões anotadas", len(banco), border=True)
            st.metric("Mamas", int(banco["mama"].nunique()) if len(banco) else 0, border=True)
            st.metric("Com exame de confirmação", int((banco["exame_confirmacao"] != "").sum()) if len(banco) else 0,
                      border=True)
            st.metric("Caixas revisadas (lesão real?)", int(banco["lesao_real"].isin(["sim", "não"]).sum()) if len(banco)
                      else 0, border=True)
        st.caption("As anotações são feitas na aba **2) Resultados** (seção *Revisar e confirmar resultados*) e ficam na "
                   "pasta **anotacoes/** do MARIA, junto com cópias das imagens. São dados de pacientes: trate como "
                   "informação sensível (LGPD) e não compartilhe a pasta sem autorização.")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            if len(banco):
                import zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(arq_banco, "anotacoes.csv")
                    for f in (ANOTACOES / "imagens").glob("*"):
                        z.write(f, f"imagens/{f.name}")
                st.download_button("Exportar anotações (.zip)", buf.getvalue(), file_name=f"maria_anotacoes_"
                                   f"{time.strftime('%Y%m%d')}.zip", mime="application/zip", icon=":material/upload_file:",
                                   help="Para levar as anotações até a máquina que tem os dados completos de treino.")
            importado = st.file_uploader("Importar anotações (.zip)", type=["zip"], key="importar_anotacoes",
                                         help="Junta ao banco as anotações exportadas em outro computador.")
        if importado is not None and st.session_state.get("zip_importado") != importado.file_id:
            import zipfile
            st.session_state.zip_importado = importado.file_id
            try:
                with zipfile.ZipFile(importado) as z:
                    if "anotacoes.csv" not in z.namelist():
                        raise ValueError("o arquivo não tem anotacoes.csv (não foi exportado pelo MARIA)")
                    novo = pd.read_csv(io.BytesIO(z.read("anotacoes.csv")), dtype=str, keep_default_na=False,
                                       encoding="utf-8-sig").reindex(columns=COLUNAS_ANOTACAO, fill_value="")
                    if (novo["id"] == "").any():
                        raise ValueError("anotacoes.csv sem a coluna id")
                    (ANOTACOES / "imagens").mkdir(parents=True, exist_ok=True)
                    ignorados = 0
                    for nome in z.namelist():
                        # só "imagens/<nome simples>": impede gravar fora da pasta (ex.: "imagens/../../arquivo")
                        if re.fullmatch(r"imagens/[A-Za-z0-9_.-]+", nome) and ".." not in nome:
                            if not (ANOTACOES / nome).exists():
                                (ANOTACOES / nome).write_bytes(z.read(nome))
                        elif nome != "anotacoes.csv" and not nome.endswith("/"):
                            ignorados += 1
                banco = pd.concat([banco[~banco["id"].isin(novo["id"])], novo], ignore_index=True)
                banco.reindex(columns=COLUNAS_ANOTACAO, fill_value="").to_csv(arq_banco, index=False, encoding="utf-8-sig")
                st.success(f"{len(novo)} anotação(ões) importadas." +
                           (f" {ignorados} arquivo(s) estranho(s) no zip foram ignorados." if ignorados else ""),
                           icon=":material/check:")
                versao_banco = arq_banco.stat().st_mtime
            except (zipfile.BadZipFile, ValueError, KeyError, pd.errors.ParserError) as e:
                st.error(f"Não consegui importar: {e}", icon=":material/error:")
        if len(banco):
            with st.expander("Ver o banco de anotações", icon=":material/table:"):
                mostrar(st.dataframe, banco, hide_index=True, alt="Banco de anotações do usuário")

        st.subheader("Retreinar o modelo")
        st.warning("O retreino mistura as anotações com os **dados originais de treino** (CBIS-DDSM e CDD-CESM, ~160 GB), "
                   "que só existem na máquina onde o MARIA foi treinado. Em qualquer outro computador o botão fica "
                   "desativado: exporte as anotações (acima) e importe-as na máquina de treino.",
                   title="Disponível apenas na máquina com os dados completos do modelo", icon=":material/dns:")
        sit = situacao_retreino(str(pasta_modelo), versao_banco)
        if sit["disponivel"]:
            st.success("Esta máquina tem os dados completos de treino." +
                       (" GPU disponível." if sit.get("gpu") else " Sem GPU: o retreino será muito lento."),
                       icon=":material/check_circle:")
        else:
            st.error("Esta máquina NÃO tem os dados completos de treino: o retreino está desativado aqui.",
                     icon=":material/block:")
            with st.expander("O que falta nesta máquina"):
                st.code("\n".join(sit.get("faltando", [])), language="text")

        nomes_modelo = {"densidade": "Densidade", "detector": "Detector de lesões", "achado": "Achado"}
        casos, minimo = sit.get("casos", {}), sit.get("minimo", {})
        with st.container(horizontal=True):
            for m, nome in nomes_modelo.items():
                st.metric(f"{nome}: casos utilizáveis", f"{casos.get(m, 0)} / mín. {minimo.get(m, '?')}", border=True,
                          help={"densidade": "Mamas com densidade confirmada e mamografia completa.",
                                "detector": "Mamografias completas com todas as caixas do modelo revisadas (lesão real "
                                            "sim/não) ou com lesões marcadas pelo usuário.",
                                "achado": "Lesões com achado confirmado E exame de confirmação informado."}[m])
        prontos = [m for m in nomes_modelo if casos.get(m, 0) >= minimo.get(m, 10 ** 9)]
        escolha = st.pills("Modelos a retreinar", list(nomes_modelo), selection_mode="multi", default=prontos,
                           format_func=lambda m: nomes_modelo[m], key="modelos_retreino")
        poucos = [nomes_modelo[m] for m in escolha if m not in prontos]
        if poucos:
            st.info(f"Poucos casos para: {', '.join(poucos)} (abaixo do mínimo). Esses modelos serão pulados.",
                    icon=":material/info:")
        st.caption("A **patologia** (benigno/maligno) não é retreinada. Tempo aproximado com GPU: densidade ~15 min, "
                   "detector ~40 min, achado ~10 min (+ ~10 min na primeira vez, para preparar os dados). O retreino "
                   "continua mesmo se você fechar o navegador.")
        est = estado_retreino()
        rodando = bool(est and not est["terminou"])
        confirma = st.checkbox("Entendo que o modelo novo só substitui o atual se não piorar no teste fixo, e que a "
                               "versão anterior fica guardada para restaurar.", key="confirma_retreino")
        if st.button("Retreinar o modelo", type="primary", icon=":material/model_training:", key="retreinar",
                     disabled=not (sit["disponivel"] and escolha and confirma and not rodando)):
            pasta_log = ANOTACOES / "retreinos"
            pasta_log.mkdir(parents=True, exist_ok=True)
            carimbo_r = time.strftime("%Y%m%d_%H%M%S")
            log_r = pasta_log / f"log_{carimbo_r}.txt"
            separado = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
                        if os.name == "nt" else {"start_new_session": True})
            with open(log_r, "w", encoding="utf-8") as f:
                teste = ["--rapido"] if os.environ.get("MARIA_RETREINO_TESTE") == "1" else []   # só para testar o fluxo
                proc = subprocess.Popen([sys.executable, "-u", "retreinar.py", "--modelos", *escolha, *teste],
                                        cwd=str(pasta_modelo),
                                        stdout=f, stderr=subprocess.STDOUT, **separado,
                                        env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
            (ANOTACOES / "retreino_estado.json").write_text(json.dumps(
                {"pid": proc.pid, "inicio": time.strftime("%d/%m %H:%M"), "modelos": escolha, "log": str(log_r)}),
                encoding="utf-8")
            st.rerun()

        if est:
            @st.fragment(run_every=5 if rodando else None)
            def acompanhar():
                e = estado_retreino()
                if e is None:
                    return
                if e.get("interrompido"):
                    st.error(f"O retreino iniciado {e['inicio']} ({', '.join(e['modelos'])}) foi interrompido antes de "
                             "terminar (computador desligado, falta de memória ou processo encerrado). Os modelos em uso "
                             "não foram alterados; você pode retreinar de novo.", icon=":material/error:")
                elif e["terminou"]:
                    st.success(f"Último retreino ({', '.join(e['modelos'])}, iniciado {e['inicio']}) terminado.",
                               icon=":material/task_alt:")
                else:
                    st.info(f"Retreinando {', '.join(e['modelos'])} (iniciado {e['inicio']})… esta área atualiza sozinha.",
                            icon=":material/hourglass_top:")
                with st.expander("Andamento (log)", expanded=not e["terminou"], icon=":material/terminal:"):
                    st.code("\n".join(e["texto"].splitlines()[-30:]), language="text")
            acompanhar()

        relatorios = sorted((ANOTACOES / "retreinos").glob("relatorio_*.json")) if (ANOTACOES / "retreinos").is_dir() else []
        if relatorios:
            rel = json.loads(relatorios[-1].read_text(encoding="utf-8"))
            st.markdown(f"**Último relatório** ({rel['carimbo']}, {rel['dispositivo']}"
                        + (", modo de teste rápido" if rel.get("rapido") else "") + ")")
            for m, info in rel["modelos"].items():
                cor = {"substituído": "green", "mantido": "blue", "pulado": "gray", "erro": "red"}.get(info["situacao"], "gray")
                with st.container(border=True):
                    with st.container(horizontal=True, vertical_alignment="center"):
                        st.markdown(f"**{nomes_modelo.get(m, m)}**")
                        st.badge(info["situacao"], color=cor)
                    if "antes" in info:
                        linhas = []
                        for conjunto in ("teste_fixo", "usuario_guardado"):
                            for k in info["antes"].get(conjunto, {}):
                                a, d = info["antes"][conjunto].get(k), info.get("depois", {}).get(conjunto, {}).get(k)
                                if isinstance(a, float) or isinstance(d, float):
                                    linhas.append({"conjunto": conjunto.replace("_", " "), "medida": k.replace("_", " "),
                                                   "modelo anterior": a, "modelo retreinado": d})
                        if linhas:
                            mostrar(st.dataframe, pd.DataFrame(linhas), hide_index=True,
                                    column_config={"modelo anterior": st.column_config.NumberColumn(format="%.3f"),
                                                   "modelo retreinado": st.column_config.NumberColumn(format="%.3f")},
                                    alt=f"Comparação do modelo de {m} antes e depois do retreino")
                    else:
                        st.caption(info.get("motivo", ""))

        try:
            lista_versoes = json.loads(subprocess.run(
                [sys.executable, "retreinar.py", "--versoes"], cwd=str(pasta_modelo), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60).stdout.strip().splitlines()[-1])
        except Exception:
            lista_versoes = []
        if lista_versoes:
            with st.expander("Versões anteriores dos modelos (restaurar)", icon=":material/history:"):
                opcoes_v = {v["versao"]: f"{v['versao']}  ·  {', '.join(v['modelos'])}" for v in lista_versoes}
                versao = st.selectbox("Versão", list(opcoes_v), format_func=opcoes_v.get, key="versao_restaurar")
                ok_rest = st.checkbox("Confirmo que quero voltar os modelos para esta versão.", key="confirma_restaurar")
                if st.button("Restaurar esta versão", icon=":material/restore:", disabled=not ok_rest or rodando,
                             key="restaurar"):
                    r = subprocess.run([sys.executable, "retreinar.py", "--restaurar", versao], cwd=str(pasta_modelo),
                                       capture_output=True, text=True, encoding="utf-8", errors="replace")
                    (st.success if r.returncode == 0 else st.error)(r.stdout[-500:] or r.stderr[-500:])

    with aba_res:
        res = st.session_state.get("resultado")
        if not res or res.get("df") is None:
            st.info("Nenhum resultado ainda. Escolha a pasta, (opcionalmente) a planilha, e clique em **Classificar**.")
        else:
            pasta_res = Path(res["pasta"])
            st.caption(f"Pasta: {pasta_res}  ·  rodada {res['carimbo']}")
            tab = res["df"].copy()
            for c in ("mama", "vista", "origem_lesao", "imagens_usadas", "imagem", "x0", "y0", "x1", "y1", "score_deteccao",
                      "densidade", "densidade_origem", "aviso_densidade", "concordancia"):
                if c not in tab:
                    tab[c] = np.nan
            # colunas de texto sempre como texto (uma coluna toda vazia seria lida como número)
            for c in ("amostra", "mama", "vista", "origem_lesao", "imagens_usadas", "patologia", "achado",
                      "densidade_origem", "aviso_densidade", "concordancia", "imagem"):
                tab[c] = tab[c].astype("object").where(tab[c].notna(), None)
            tab["densidade"] = pd.to_numeric(tab["densidade"], errors="coerce").astype("Int64")
            tab["densidade_txt"] = tab["densidade"].map(densidade_txt)
            lesoes = tab[tab["origem_lesao"] != ORIGEM_NENHUMA]
            if tab.empty:
                st.warning("A rodada não produziu nenhum resultado: nenhuma imagem pôde ser classificada. "
                           "Veja o log da execução abaixo.", icon=":material/warning:")
                with st.expander("Logs da execução", icon=":material/terminal:", expanded=True):
                    st.code(res["log"].get("maria", "")[-6000:], language="text")
                st.stop()   # a aba de resultados é o último trecho da página

            # ---- resumo ----
            with st.container(horizontal=True):
                st.metric("Lesões classificadas", len(lesoes), border=True)
                st.metric("Maligno", int((lesoes["patologia"] == "Maligno").sum()), border=True)
                st.metric("Benigno", int((lesoes["patologia"] == "Benigno").sum()), border=True)
                st.metric("Mamas", int(tab["mama"].nunique()), border=True)
                st.metric("Limiar", f"{limiar:.2f}", border=True)

            # ---- tabela ----
            st.subheader("Resultados por lesão")
            f1, f2, f3 = st.columns([2, 2, 1.5])
            opcoes = sorted(tab["patologia"].dropna().unique(), key=lambda v: (v not in ("Maligno", "Benigno"), v))
            filtro = f1.pills("Patologia", opcoes, selection_mode="multi", default=opcoes)
            origens = [o for o in (ORIGEM_RECORTE, ORIGEM_USUARIO, ORIGEM_MODELO, ORIGEM_NENHUMA)
                       if o in set(tab["origem_lesao"])]
            filtro_or = f2.pills("Origem da lesão", origens, selection_mode="multi", default=origens)
            busca = f3.text_input("Buscar amostra ou mama")
            vis = tab[tab["patologia"].isin(filtro) & tab["origem_lesao"].isin(filtro_or)]
            if busca:
                vis = vis[vis["amostra"].str.contains(busca, case=False, na=False)
                          | vis["mama"].astype(str).str.contains(busca, case=False, na=False)]
            vis = vis.sort_values("p_maligno", ascending=False)

            colunas_tela = ["amostra", "mama", "vista", "origem_lesao", "imagens_usadas", "patologia", "conf_patologia",
                            "p_maligno", "achado", "conf_achado", "densidade", "densidade_origem", "concordancia",
                            "score_deteccao"]
            estilo = vis[colunas_tela].style
            aplicar = getattr(estilo, "map", None) or estilo.applymap
            titulos = {"origem_lesao": "origem da lesão", "imagens_usadas": "imagens usadas",
                       "conf_patologia": "confiança patologia", "p_maligno": "P(maligno)", "conf_achado": "confiança achado",
                       "densidade_origem": "origem da densidade", "concordancia": "usuário × modelo",
                       "score_deteccao": "score do detector"}
            mostrar(st.dataframe, aplicar(estilo_patologia, subset=["patologia"]), hide_index=True,
                    column_config={c: st.column_config.Column(titulos.get(c, c), help=LEGENDA[c]) for c in colunas_tela},
                    alt="Resultados por lesão: patologia, achado, densidade e origem de cada informação")
            with st.container(horizontal=True, vertical_alignment="center"):
                st.download_button("Baixar tabela (CSV para Excel)", csv_excel(vis.drop(columns=["densidade_txt"])),
                                   file_name=f"maria_resultados_{res['carimbo']}.csv",
                                   mime="text/csv", icon=":material/download:")
            with st.expander("O que significa cada coluna", icon=":material/help:"):
                st.table(pd.DataFrame({"coluna": [titulos.get(c, c) for c in colunas_tela],
                                       "significado": [LEGENDA[c] for c in colunas_tela]}).set_index("coluna"))

            # ---- revisão / confirmação (vira o banco de anotações usado no retreino) ----
            st.subheader("Revisar e confirmar resultados")
            st.warning(AVISO_CONFIRMACAO, icon=":material/medical_information:")
            st.caption("Preencha só o que foi confirmado. **Lesão real?**: para caixas marcadas ou detectadas, informe se é "
                       "uma lesão de verdade ou um falso positivo (para o detector, revise TODAS as caixas do modelo na "
                       "imagem). **Densidade confirmada** vale para a mama toda. Clique em **Salvar anotações** ao terminar; "
                       "as anotações ficam na pasta anotacoes/ e podem ser usadas no retreino (aba 3).")
            banco = ler_banco()
            anteriores = banco[banco["rodada"] == res["carimbo"]].set_index("amostra")
            rev = tab[["amostra", "mama", "vista", "origem_lesao", "patologia", "achado", "densidade", "densidade_origem",
                       "imagem", "x0", "y0", "x1", "y1"]].copy()
            for c in CAMPOS_REVISAO:
                rev[c] = [anteriores.at[a, c] if a in anteriores.index and anteriores.at[a, c] else None for a in rev["amostra"]]
            revisado = st.data_editor(
                rev, key=f"revisao_{res['carimbo']}", hide_index=True, num_rows="fixed",
                column_order=["amostra", "mama", "vista", "origem_lesao", "patologia", "achado", "densidade", *CAMPOS_REVISAO],
                disabled=["amostra", "mama", "vista", "origem_lesao", "patologia", "achado", "densidade"],
                column_config={
                    "origem_lesao": st.column_config.TextColumn("origem da lesão"),
                    "patologia": st.column_config.TextColumn("patologia (MARIA)"),
                    "achado": st.column_config.TextColumn("achado (MARIA)"),
                    "densidade": st.column_config.NumberColumn("densidade (MARIA)", format="%d"),
                    "lesao_real": st.column_config.SelectboxColumn(
                        "lesão real?", options=["sim", "não"],
                        help="A caixa é uma lesão de verdade (sim) ou um falso positivo (não)?"),
                    "patologia_confirmada": st.column_config.SelectboxColumn(
                        "patologia confirmada", options=["Benigno", "Maligno"],
                        help="Resultado confirmado por exame padrão-ouro (fica registrado; não é usado no retreino)."),
                    "achado_confirmado": st.column_config.SelectboxColumn(
                        "achado confirmado", options=["Massa/Nódulo", "Microcalcificação"],
                        help="Tipo de achado confirmado por exame adicional."),
                    "exame_confirmacao": st.column_config.SelectboxColumn(
                        "exame de confirmação", options=EXAMES_CONFIRMACAO, width="medium",
                        help="Exame que confirmou a patologia/achado. Sem ele, a confirmação não entra no retreino."),
                    "densidade_confirmada": st.column_config.SelectboxColumn(
                        "densidade confirmada", options=["1", "2", "3", "4", "A", "B", "C", "D"],
                        help="Densidade do laudo (vale para a mama inteira)."),
                    "observacao": st.column_config.TextColumn("observação", width="medium"),
                },
                alt="Revisão dos resultados: confirmações do usuário para cada lesão",
            )
            txt = revisado[CAMPOS_REVISAO].fillna("").astype(str)
            sem_exame = int((((txt["patologia_confirmada"] != "") | (txt["achado_confirmado"] != ""))
                             & (txt["exame_confirmacao"] == "")).sum())
            if sem_exame:
                st.info(f"{sem_exame} lesão(ões) com patologia/achado confirmado mas sem o exame de confirmação: serão "
                        "salvas, mas não entram no retreino.", icon=":material/info:")
            with st.container(horizontal=True, vertical_alignment="center"):
                if st.button("Salvar anotações", type="primary", icon=":material/save:", key="salvar_anotacoes"):
                    n, sem_img = salvar_anotacoes(revisado, res["carimbo"])
                    if n:
                        st.success(f"{n} lesão(ões) salvas no banco de anotações." +
                                   (f" {sem_img} sem imagem localizada." if sem_img else ""), icon=":material/check:")
                    else:
                        st.info("Nada para salvar: nenhuma linha foi preenchida.", icon=":material/info:")
                st.download_button("Baixar resultados revisados (CSV para Excel)",
                                   csv_excel(revisado.drop(columns=["imagem", "x0", "y0", "x1", "y1"])),
                                   file_name=f"maria_revisado_{res['carimbo']}.csv", mime="text/csv",
                                   icon=":material/download:")

            # ---- gráficos ----
            st.subheader("Gráficos")
            g1, g2 = st.columns(2)
            with g1:
                mamas = tab.dropna(subset=["mama"]).drop_duplicates("mama")[["mama", "densidade_txt", "densidade_origem"]]
                mamas["densidade_origem"] = mamas["densidade_origem"].fillna("sem densidade").replace("", "sem densidade")
                graf = alt.Chart(mamas).mark_bar().encode(
                    x=alt.X("densidade_txt:N", sort=ORDEM_DENS, title=TITULO_DENS, axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("densidade_origem:N", sort=["informada pelo usuário", "inferida pelo modelo",
                                                                    "sem densidade"]),
                    y=alt.Y("count():Q", title="Mamas",
                            axis=eixo_contagem(mamas.groupby(["densidade_txt", "densidade_origem"]).size().max())),
                    color=alt.Color("densidade_origem:N", title="Origem da densidade",
                                    scale=alt.Scale(domain=["informada pelo usuário", "inferida pelo modelo",
                                                            "sem densidade"], range=["#6a1b9a", "#00897b", "#9e9e9e"])),
                    tooltip=[alt.Tooltip("densidade_txt:N", title="Densidade"),
                             alt.Tooltip("densidade_origem:N", title="Origem"), alt.Tooltip("count():Q", title="Mamas")],
                ).properties(title="Densidade: informada pelo usuário × inferida pelo modelo", height=280)
                mostrar(st.altair_chart, graf, alt="Contagem de mamas por densidade, separando informada e inferida")
            with g2:
                usr = tab[tab["origem_lesao"] == ORIGEM_USUARIO]
                mod = tab[tab["origem_lesao"] == ORIGEM_MODELO]
                if usr.empty and mod.empty:
                    st.caption("Nenhuma lesão marcada pelo usuário ou detectada pelo modelo nesta rodada "
                               "(só recortes fornecidos).")
                else:
                    conc_u, conc_m = usr["concordancia"].fillna(""), mod["concordancia"].fillna("")
                    situacoes = [
                        ("Marcada pelo usuário e\ndetectada pelo modelo", (conc_u == "o modelo também detectou").sum(), "#2e7d32"),
                        ("Marcada só pelo usuário\n(o modelo não detectou)", (conc_u == "o modelo NÃO detectou").sum(), "#c62828"),
                        ("Detectada só pelo modelo\n(o usuário não marcou)", (conc_m == "não marcada pelo usuário").sum(), "#f9a825"),
                        ("Detectada pelo modelo em imagem\nsem nenhuma marcação do usuário", (conc_m == "").sum(), "#90a4ae"),
                        ("Marcada pelo usuário\n(detecção do modelo desligada)",
                         (conc_u == "detecção do modelo desligada").sum(), "#cfd8dc"),
                    ]
                    situacoes = [(n, int(q), c) for n, q, c in situacoes if q or not n.startswith("Marcada pelo usuário\n(")]
                    sit = pd.DataFrame(situacoes, columns=["situação", "lesões", "cor"])
                    base_sit = alt.Chart(sit).encode(
                        y=alt.Y("situação:N", sort=list(sit["situação"]), title=None,
                                axis=alt.Axis(labelLimit=320, labelExpr="split(datum.label, '\\n')")),
                        x=alt.X("lesões:Q", title="Lesões (caixas)", axis=eixo_contagem(sit["lesões"].max())),
                        tooltip=[alt.Tooltip("situação:N", title="Situação"), alt.Tooltip("lesões:Q", title="Lesões")],
                    )
                    graf = (base_sit.mark_bar().encode(color=alt.Color("cor:N", scale=None, legend=None))
                            + base_sit.mark_text(align="left", dx=4).encode(text="lesões:Q")
                            ).properties(title="Lesões: marcadas pelo usuário × detectadas pelo modelo", height=60 * len(sit))
                    mostrar(st.altair_chart, graf, alt="Quantas lesões o usuário e o modelo marcaram, juntas e separadas")
                    n_nenhuma = int((tab["origem_lesao"] == ORIGEM_NENHUMA).sum())
                    if n_nenhuma:
                        st.caption(f"{n_nenhuma} mamografia(s) completa(s) sem nenhuma lesão marcada ou detectada.")

            st.markdown("**Densidade × patologia** (cada ponto é uma lesão; o traço preto é a média)")
            grafico_agrupado(lesoes, "densidade_txt", ORDEM_DENS, TITULO_DENS, "densidade",
                             "patologia", "conf_patologia", COR_PATOLOGIA, "Patologia", "na")
            st.markdown("**Densidade × achado** (cada ponto é uma lesão; o traço preto é a média)")
            grafico_agrupado(lesoes, "densidade_txt", ORDEM_DENS, TITULO_DENS, "densidade",
                             "achado", "conf_achado", COR_ACHADO, "Achado", "no")
            st.markdown("**Achado × patologia** (cada ponto é uma lesão; o traço preto é a média)")
            grafico_agrupado(lesoes, "achado", COR_ACHADO.domain, "Achado", "achado",
                             "patologia", "conf_patologia", COR_PATOLOGIA, "Patologia", "na")

            with st.expander("Distribuição de P(maligno)", icon=":material/bar_chart:"):
                bins = np.linspace(0, 1, 11)
                cont, _ = np.histogram(lesoes["p_maligno"].dropna(), bins=bins)
                mostrar(st.bar_chart, pd.DataFrame({"lesões": cont},
                                                   index=[f"{bins[i]:.1f}–{bins[i + 1]:.1f}" for i in range(10)]),
                        alt="Histograma de P(maligno) das lesões")
                st.caption("P(maligno) perto de 0,5 = caso incerto; confie menos nele. Em bases diferentes do CBIS-DDSM, "
                           "o ranking costuma ser confiável, mas o limiar 0,5 pode precisar de ajuste.")

            # ---- detalhe por lesão ----
            st.subheader("Detalhe da lesão")
            rotulos = dict(zip(tab["amostra"], tab["origem_lesao"]))
            sel = st.selectbox("Lesão", list(vis["amostra"]) if len(vis) else [], key="lesao_sel",
                               format_func=lambda a: f"{a}  ·  {rotulos.get(a, '')}")
            if sel:
                linha = tab[tab["amostra"] == sel].iloc[0]
                origem = linha["origem_lesao"]
                caminho = Path(str(linha["imagem"])) if isinstance(linha["imagem"], str) else None
                p = float(linha["p_maligno"]) if pd.notna(linha["p_maligno"]) else np.nan
                tem_caixa = pd.notna(linha["x0"])
                vista = linha["vista"] if isinstance(linha.get("vista"), str) and linha["vista"] else "não informada"
                cor_origem = {ORIGEM_RECORTE: "blue", ORIGEM_USUARIO: "green", ORIGEM_MODELO: "orange"}.get(origem, "gray")

                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"#### {sel}")
                    st.badge(origem, color=cor_origem)
                    if isinstance(linha["concordancia"], str) and linha["concordancia"]:
                        st.badge(f"usuário × modelo: {linha['concordancia']}", color="gray")

                ci, cd = st.columns([2, 3], gap="large")
                # ---- imagens ----
                with ci:
                    if caminho is not None and origem != ORIGEM_RECORTE:
                        mesmas = tab[(tab["imagem"] == linha["imagem"]) & tab["x0"].notna()]
                        caixas = [(r.x0, r.y0, r.x1, r.y1, r.amostra.rsplit("_", 1)[-1], r.origem_lesao, r.amostra == sel)
                                  for r in mesmas.itertuples()]
                        img = carregar_imagem(caminho, caixas)
                        if img is not None:
                            mostrar(st.image, img, caption=f"Mamografia completa ({vista}): {caminho.name}")
                            st.caption("Caixas: **vermelho** = esta lesão · **verde** = marcadas pelo usuário · "
                                       "**amarelo** = detectadas pelo modelo")
                        if tem_caixa:
                            cheia = ler_cinza(caminho)
                            if cheia is not None:
                                m = 20
                                zoom = cheia.crop((max(0, linha.x0 - m), max(0, linha.y0 - m),
                                                   min(cheia.width, linha.x1 + m), min(cheia.height, linha.y1 + m)))
                                zoom.thumbnail((350, 350))
                                st.image(zoom, caption="Recorte que foi classificado (caixa + 20 px)")
                    elif caminho is not None and caminho.is_file():
                        img = carregar_imagem(caminho)
                        if img is not None:
                            mostrar(st.image, img, caption=f"Recorte fornecido: {caminho.name}")
                    else:
                        st.caption("Imagem não localizada.")

                # ---- informações ----
                with cd:
                    with st.container(border=True):
                        st.markdown("**Identificação**")
                        st.markdown(f"- **Mama:** {linha['mama']}\n"
                                    f"- **Vista:** {vista}\n"
                                    f"- **Imagens analisadas:** {linha['imagens_usadas']}")

                    with st.container(border=True):
                        st.markdown("**Localização e detecção**")
                        itens = [f"- **Origem da lesão:** {origem}"]
                        if tem_caixa:
                            larg, alt_px = int(linha.x1 - linha.x0), int(linha.y1 - linha.y0)
                            itens.append(f"- **Posição na imagem** (pixels): x {int(linha.x0)}–{int(linha.x1)}, "
                                         f"y {int(linha.y0)}–{int(linha.y1)} · tamanho {larg} × {alt_px} px")
                        if origem == ORIGEM_MODELO and str(linha["score_deteccao"]).strip() not in ("", "nan"):
                            itens.append(f"- **Confiança do detector:** {float(linha['score_deteccao']):.2f} "
                                         f"(limiar usado: {limiar_det:.2f})")
                        if isinstance(linha["concordancia"], str) and linha["concordancia"]:
                            itens.append(f"- **Comparação usuário × modelo:** {linha['concordancia']}")
                        elif origem == ORIGEM_RECORTE:
                            itens.append("- O recorte foi fornecido pronto; não houve detecção nesta lesão.")
                        st.markdown("\n".join(itens))

                    if np.isnan(p):
                        st.info(f"Resultado: {linha['patologia']}", icon=":material/info:")
                    else:
                        with st.container(horizontal=True):
                            st.metric("Patologia", linha["patologia"], delta=f"confiança {linha['conf_patologia']}",
                                      delta_color="off", delta_arrow="off", border=True,
                                      help="Benigno ou Maligno pelo limiar escolhido; confiança = quão longe de 50%.")
                            st.metric("Achado", linha["achado"], delta=f"confiança {linha['conf_achado']}",
                                      delta_color="off", delta_arrow="off", border=True,
                                      help="Massa/nódulo ou microcalcificação.")
                            if pd.notna(linha["densidade"]):
                                st.metric("Densidade (BI-RADS)", densidade_txt(linha["densidade"]),
                                          delta=str(linha["densidade_origem"]), delta_color="off", delta_arrow="off",
                                          border=True, help="1 (A) adiposa … 4 (D) extremamente densa.")
                            else:
                                st.metric("Densidade (BI-RADS)", "—", delta="não disponível", delta_color="off",
                                          delta_arrow="off", border=True)
                        st.markdown(f"**P(maligno) = {p:.3f}** · limiar {limiar:.2f} "
                                    f"({'acima' if p >= limiar else 'abaixo'} do limiar → {linha['patologia']})")
                        st.progress(min(max(p, 0.0), 1.0))

                    if pd.notna(linha["densidade"]):
                        (st.warning if int(linha["densidade"]) >= 3 else st.info)(
                            linha["aviso_densidade"], title=f"Densidade {densidade_txt(linha['densidade'])} "
                                                            f"({linha['densidade_origem']})")
                    else:
                        st.caption("Densidade não disponível: inclua a mamografia completa da mesma mama ou informe a "
                                   "densidade na planilha.")

            with st.expander("Logs da execução", icon=":material/terminal:"):
                for k, v in res["log"].items():
                    st.markdown(f"**{k}**")
                    st.code(v[-6000:], language="text")



# -----------------------------------------------------------------------------
# navegação (barra lateral)
# -----------------------------------------------------------------------------

navegacao = st.navigation([
    st.Page(pagina_sobre, title="Sobre o modelo", icon=":material/info:", url_path="sobre"),
    st.Page(pagina_executar, title="Executar", icon=":material/play_circle:", url_path="executar", default=True),
])
navegacao.run()
