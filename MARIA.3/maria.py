"""
================================================================================
 MARIA.PY — Diagnóstico automático a partir de uma pasta de DICOMs e/ou PNGs
================================================================================
Recebe uma pasta com arquivos DICOM (.dcm) e/ou PNG/JPG de mamografia — de
qualquer origem, não precisa ser do CBIS-DDSM — e entrega um results.csv com
o diagnóstico de patologia (Benigno/Maligno) e achado (Massa/Nódulo vs
Microcalcificação) pra cada amostra encontrada. Os dois tipos de arquivo
podem estar juntos na mesma pasta — o script trata cada um do seu jeito.

COMO IDENTIFICA "RECORTE" x "COMPLETA" — ARQUIVOS DICOM (.dcm):
  1) Primeiro tenta pela tag SeriesDescription (se o arquivo vier com
     descrição tipo "cropped images" ou "full mammogram images" — como no
     CBIS-DDSM). Máscaras de ROI ("mask") são sempre ignoradas.
  2) Se a descrição não ajudar, usa um fallback por RESOLUÇÃO: entre
     imagens do mesmo paciente+lado+vista, a de maior resolução é tratada
     como mamografia completa, e as menores como recorte. Se só houver UMA
     imagem pra aquele paciente+lado+vista (nada pra comparar), ela é
     tratada como mamografia completa (é a suposição mais segura pra uma
     aquisição clínica padrão).

COMO IDENTIFICA "RECORTE" x "COMPLETA" — ARQUIVOS PNG/JPG:
  PNG e JPG não carregam metadado DICOM, então a identificação é pelo NOME
  do arquivo — o mesmo padrão usado no analisar_casos_png.py:

      <nome_do_caso>_recorte_cc.png    -> recorte da lesão, vista CC
      <nome_do_caso>_recorte_mlo.png   -> recorte da lesão, vista MLO
      <nome_do_caso>_recorte.png       -> recorte da lesão, vista única
      <nome_do_caso>_completa.png      -> mamografia completa

  Um arquivo PNG/JPG sem um desses sufixos é ignorado (com aviso no console).

COMO AGRUPA AS AMOSTRAS:
  Pra DICOM: usa as tags padrão PatientID + Laterality (ImageLaterality).
  Dentro de cada grupo, separa por ViewPosition (CC/MLO) quando essa tag
  existir — isso permite acionar o modelo multivista quando as duas vistas
  existem. Se ViewPosition não existir, todas as imagens daquele lado
  entram numa "vista" só (sem multivista, mas patologia/achado continuam
  funcionando pelo recorte/completa disponíveis).
  Pra PNG/JPG: o <nome_do_caso> do próprio arquivo já define a amostra.

  IMPORTANTE (só pra DICOM): isso assume uma lesão relevante por lado da
  mama. DICOMs reais não trazem "número da lesão" como o CBIS-DDSM — se
  houver mais de uma lesão suspeita no mesmo lado, elas ficam agrupadas
  juntas nesta versão.

COMBINAÇÃO DE MODELOS USADA (mesma do projeto, validada no CBIS-DDSM):
  - PATOLOGIA: média de recorte + completa (TTA nos dois, quando existem).
  - ACHADO: média de recorte + multivista (TTA no recorte; multivista só
    quando há recorte CC e recorte MLO da mesma amostra).

Como rodar (a partir de ~/MARIA):
    source ~/tcia_env/bin/activate
    cd ~/MARIA
    python3 maria.py --pasta /caminho/para/pasta/com/dicoms_ou_pngs

ESTRUTURA DE PASTAS ESPERADA (pra copiar pra outra máquina, leve isso tudo):

    MARIA/                                    (ou qualquer nome de pasta)
        maria.py                              <- este script
        modelo/
            treinar_maria_cbis_ddsm.py
            treinar_maria_multivista.py
            resnet50_multitask_mama_do_zero.pth
            resnet50_multitask_mama_completa.pth
            resnet50_multivista.pth

O maria.py encontra a pasta modelo/ sozinho (relativa a onde ELE está salvo,
não à pasta onde o comando é rodado), então funciona em qualquer máquina
desde que essa estrutura seja mantida.

Gera results.csv (por padrão, na mesma pasta onde o maria.py está salvo —
não na pasta de entrada) com as colunas: amostra, dados, patologia, achado,
conf_patologia, conf_achado.
================================================================================
"""

# =============================================================================
# 0. CHECAGEM DE DEPENDÊNCIAS (roda ANTES de qualquer outro import, pra dar
#    uma mensagem clara — com o comando de instalação certo — em vez de um
#    traceback confuso caso falte alguma biblioteca)
# =============================================================================
import importlib.util
import sys

REQUISITOS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "pydicom": "pydicom",
    "PIL": "pillow",
    "tqdm": "tqdm",
}

_faltando = [pacote_pip for modulo, pacote_pip in REQUISITOS.items() if importlib.util.find_spec(modulo) is None]

if _faltando:
    print("❌ Faltam bibliotecas Python para rodar o maria.py:")
    for pacote in _faltando:
        print(f"    - {pacote}")
    print("\nInstale com:")
    print(f"    pip install {' '.join(_faltando)} --break-system-packages")
    print("\n(Se estiver usando um ambiente virtual, ative-o antes: source ~/tcia_env/bin/activate)")
    sys.exit(1)

# =============================================================================
# 1. IMPORTS
# =============================================================================

import argparse
import csv
from pathlib import Path

import pydicom
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---- Torna a pasta modelo/ (ao lado deste arquivo) importável e a fonte
# padrão dos modelos .pth, independente de onde o maria.py estiver rodando ----
import sys

PASTA_SCRIPT = Path(__file__).resolve().parent
PASTA_MODELO = PASTA_SCRIPT / "modelo"
if str(PASTA_MODELO) not in sys.path:
    sys.path.insert(0, str(PASTA_MODELO))

from treinar_maria_cbis_ddsm import CFG, MultiTaskResNet50, dicom_para_imagem_pil
from treinar_maria_multivista import MultiViewResNet50, parse_patient_id

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
PALAVRAS_MASCARA = ("mask", "roi mask")

NOMES_PATOLOGIA = {0: "Benigno", 1: "Maligno"}
NOMES_ACHADO = {0: "Massa/Nódulo", 1: "Microcalcificação"}

EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg")

# Sufixos reconhecidos em arquivos PNG/JPG (ordem importa: mais específicos primeiro)
SUFIXOS_PNG = [
    ("_recorte_cc", "CC", "recorte"),
    ("_recorte_mlo", "MLO", "recorte"),
    ("_recorte", "SEM_VISTA", "recorte"),
    ("_completa", "SEM_VISTA", "completa"),
]


def identificar_png(caminho: Path):
    """A partir do nome do arquivo, devolve (amostra_id, vista, tipo) ou
    (None, None, None) se o nome não bater com nenhum sufixo esperado."""
    nome = caminho.stem
    nome_lower = nome.lower()
    for sufixo, vista, tipo in SUFIXOS_PNG:
        if nome_lower.endswith(sufixo):
            return nome[: len(nome) - len(sufixo)], vista, tipo
    return None, None, None


def carregar_imagem_generica(caminho: str) -> Image.Image:
    """Abre DICOM (com VOI LUT + correção MONOCHROME1) ou PNG/JPG (direto),
    dependendo da extensão do arquivo."""
    if Path(caminho).suffix.lower() in EXTENSOES_IMAGEM:
        return Image.open(caminho).convert("RGB")
    return dicom_para_imagem_pil(caminho)


# =============================================================================
# 2. LEITURA E CLASSIFICAÇÃO DOS DICOMs DA PASTA
# =============================================================================

def ler_metadados(caminho: Path):
    """Lê só os metadados (rápido, sem pixel data) de um DICOM."""
    try:
        ds = pydicom.dcmread(str(caminho), stop_before_pixels=True)
    except Exception:
        return None

    return {
        "caminho": str(caminho),
        "patient_id": str(getattr(ds, "PatientID", "")) or "SEM_ID",
        "lado": str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))).upper() or "SEM_LADO",
        "vista": str(getattr(ds, "ViewPosition", "")).upper() if getattr(ds, "ViewPosition", "") else "SEM_VISTA",
        "descricao": str(getattr(ds, "SeriesDescription", "")).lower(),
        "area": int(getattr(ds, "Rows", 0)) * int(getattr(ds, "Columns", 0)),
    }


def montar_amostras(pasta: Path):
    """Varre a pasta (recursivo) em busca de .dcm E de .png/.jpg/.jpeg,
    classifica cada um como recorte/completa/máscara e agrupa por amostra.
    Retorna um dict: amostra_id -> {vista: {tipo: {"caminho":..., "area":...}}}.

    Pra DICOM, tenta primeiro reconhecer o padrão de PatientID do CBIS-DDSM
    (via parse_patient_id) — nesse caso usa agrupamento em DUAS PASSADAS,
    porque a mamografia completa não tem número de lesão no PatientID (só o
    recorte tem), então a completa precisa ser anexada a TODAS as lesões do
    mesmo lado, não só à que bate o número por coincidência. Quando o
    PatientID não bate com esse padrão (DICOM de outra origem), cai no
    agrupamento genérico por PatientID + Laterality (tags DICOM padrão)."""
    todos_arquivos = [p for p in pasta.rglob("*") if p.is_file()]
    arquivos_imagem = [p for p in todos_arquivos if p.suffix.lower() in EXTENSOES_IMAGEM]
    arquivos_dcm = [p for p in todos_arquivos if p.suffix.lower() == ".dcm"]
    if not arquivos_dcm:
        extensoes_ignoradas = EXTENSOES_IMAGEM + (".csv", ".txt", ".py", ".md", ".json")
        arquivos_dcm = [p for p in todos_arquivos if p.suffix.lower() not in extensoes_ignoradas]

    print(f"📂 {len(arquivos_dcm)} arquivo(s) DICOM candidato(s) e {len(arquivos_imagem)} PNG/JPG encontrado(s) em: {pasta}")

    amostras = {}

    # ---- 1) Arquivos DICOM: classificação por SeriesDescription + fallback por resolução ----
    metadados = []
    for caminho in arquivos_dcm:
        m = ler_metadados(caminho)
        if m is not None:
            metadados.append(m)
    print(f"   -> {len(metadados)} DICOM(s) lido(s) com sucesso.")

    indefinidos_por_grupo = {}
    for m in metadados:
        if any(p in m["descricao"] for p in PALAVRAS_MASCARA):
            m["tipo"] = "mascara"
        elif "crop" in m["descricao"]:
            m["tipo"] = "recorte"
        elif "full" in m["descricao"]:
            m["tipo"] = "completa"
        else:
            m["tipo"] = None
            chave_grupo = (m["patient_id"], m["lado"], m["vista"])
            indefinidos_por_grupo.setdefault(chave_grupo, []).append(m)

    for chave_grupo, lista in indefinidos_por_grupo.items():
        lista.sort(key=lambda m: m["area"], reverse=True)
        for i, m in enumerate(lista):
            m["tipo"] = "completa" if i == 0 else "recorte"

    # Separa quem bate com o padrão CBIS-DDSM (PatientID reconhecível) de quem não bate
    com_padrao, sem_padrao = [], []
    for m in metadados:
        if m["tipo"] == "mascara":
            continue
        m["parsed"] = parse_patient_id(m["patient_id"])
        (com_padrao if m["parsed"] else sem_padrao).append(m)

    # ---- 1a) Padrão CBIS-DDSM reconhecido: agrupamento em duas passadas ----
    recorte_cbis = [m for m in com_padrao if m["tipo"] == "recorte"]
    completa_cbis = [m for m in com_padrao if m["tipo"] == "completa"]

    for m in recorte_cbis:
        num, lado, vista, lesao = m["parsed"]
        chave_amostra = f"P{num}_{lado}_{lesao}"
        amostras.setdefault(chave_amostra, {})
        amostras[chave_amostra].setdefault(vista, {})
        existente = amostras[chave_amostra][vista].get("recorte")
        if existente is None or m["area"] > existente["area"]:
            amostras[chave_amostra][vista]["recorte"] = {"caminho": m["caminho"], "area": m["area"]}

    chaves_por_lado = {}
    for chave_amostra in amostras:
        # chave_amostra tem o formato "P{num}_{lado}_{lesao}" — separa de volta
        partes = chave_amostra.split("_")
        num_lado = "_".join(partes[:-1])  # "P{num}_{lado}"
        chaves_por_lado.setdefault(num_lado, []).append(chave_amostra)

    for m in completa_cbis:
        num, lado, vista, _lesao_ignorada = m["parsed"]
        num_lado = f"P{num}_{lado}"
        chaves_existentes = chaves_por_lado.get(num_lado)
        if chaves_existentes:
            for chave_amostra in chaves_existentes:
                amostras[chave_amostra].setdefault(vista, {})
                existente = amostras[chave_amostra][vista].get("completa")
                if existente is None or m["area"] > existente["area"]:
                    amostras[chave_amostra][vista]["completa"] = {"caminho": m["caminho"], "area": m["area"]}
        else:
            # Nenhum recorte pra esse lado — cria uma amostra só com a completa
            chave_nova = f"{num_lado}_0"
            amostras.setdefault(chave_nova, {})
            amostras[chave_nova].setdefault(vista, {})
            chaves_por_lado.setdefault(num_lado, []).append(chave_nova)
            existente = amostras[chave_nova][vista].get("completa")
            if existente is None or m["area"] > existente["area"]:
                amostras[chave_nova][vista]["completa"] = {"caminho": m["caminho"], "area": m["area"]}

    # ---- 1b) PatientID não reconhecido (DICOM de outra origem): agrupamento genérico ----
    for m in sem_padrao:
        chave_amostra = f"{m['patient_id']}_{m['lado']}"
        amostras.setdefault(chave_amostra, {})
        amostras[chave_amostra].setdefault(m["vista"], {})
        existente = amostras[chave_amostra][m["vista"]].get(m["tipo"])
        if existente is None or m["area"] > existente["area"]:
            amostras[chave_amostra][m["vista"]][m["tipo"]] = {"caminho": m["caminho"], "area": m["area"]}

    # ---- 2) Arquivos PNG/JPG: classificação pelo sufixo do nome ----
    ignorados_png = []
    for caminho in arquivos_imagem:
        amostra_id, vista, tipo = identificar_png(caminho)
        if amostra_id is None:
            ignorados_png.append(caminho.name)
            continue
        amostras.setdefault(amostra_id, {})
        amostras[amostra_id].setdefault(vista, {})
        amostras[amostra_id][vista][tipo] = {"caminho": str(caminho), "area": 0}

    if ignorados_png:
        print("⚠️  PNG/JPG ignorado(s) (nome não termina em _recorte_cc, _recorte_mlo, _recorte ou _completa):")
        for nome in ignorados_png:
            print(f"    - {nome}")

    return amostras


# =============================================================================
# 3. INFERÊNCIA (TTA + combinação de modelos)
# =============================================================================

def transformacoes_tta(tamanho: int):
    base = transforms.Compose([transforms.Resize((tamanho, tamanho)), transforms.ToTensor(), NORMALIZE])
    return [
        lambda img: base(img),
        lambda img: base(transforms.functional.hflip(img)),
        lambda img: base(transforms.functional.vflip(img)),
        lambda img: base(transforms.functional.rotate(img, 10)),
        lambda img: base(transforms.functional.rotate(img, -10)),
    ]


def prever_com_tta(modelo, caminho: str, tamanho: int, device):
    imagem_pil = carregar_imagem_generica(caminho)
    vistas = transformacoes_tta(tamanho)
    tensores = torch.stack([v(imagem_pil) for v in vistas]).to(device)
    saida_pat, saida_ach = modelo(tensores)
    prob_pat = F.softmax(saida_pat, dim=1).mean(dim=0)
    prob_ach = F.softmax(saida_ach, dim=1).mean(dim=0)
    return prob_pat, prob_ach


def diagnosticar_amostra(amostra_id, vistas, modelo_recorte, modelo_completa, modelo_multivista, tamanho, device):
    transformacao_simples = transforms.Compose(
        [transforms.Resize((tamanho, tamanho)), transforms.ToTensor(), NORMALIZE]
    )

    tipos_encontrados = set()
    probs_pat_fontes, probs_ach_fontes = [], []

    # ---- Recorte (uma ou mais vistas) ----
    recorte_probs_pat, recorte_probs_ach = [], []
    for vista, tipos in vistas.items():
        if "recorte" in tipos:
            tipos_encontrados.add(f"recorte_{vista}".lower())
            try:
                prob_pat, prob_ach = prever_com_tta(modelo_recorte, tipos["recorte"]["caminho"], tamanho, device)
                recorte_probs_pat.append(prob_pat)
                recorte_probs_ach.append(prob_ach)
            except Exception:
                pass
    if recorte_probs_pat:
        probs_pat_fontes.append(torch.stack(recorte_probs_pat).mean(dim=0))
        probs_ach_fontes.append(torch.stack(recorte_probs_ach).mean(dim=0))

    # ---- Completa (uma ou mais vistas) — só patologia ----
    for vista, tipos in vistas.items():
        if "completa" in tipos:
            tipos_encontrados.add(f"completa_{vista}".lower())
    completa_probs_pat = []
    for vista, tipos in vistas.items():
        if "completa" in tipos:
            try:
                prob_pat, _prob_ach = prever_com_tta(modelo_completa, tipos["completa"]["caminho"], tamanho, device)
                completa_probs_pat.append(prob_pat)
            except Exception:
                pass
    if completa_probs_pat:
        probs_pat_fontes.append(torch.stack(completa_probs_pat).mean(dim=0))

    # ---- Multivista (só achado, precisa de duas VISTAS NOMEADAS diferentes com recorte) ----
    vistas_com_recorte = [v for v, tipos in vistas.items() if "recorte" in tipos and v != "SEM_VISTA"]
    if len(vistas_com_recorte) >= 2:
        try:
            caminho_a = vistas[vistas_com_recorte[0]]["recorte"]["caminho"]
            caminho_b = vistas[vistas_com_recorte[1]]["recorte"]["caminho"]
            imagem_a = carregar_imagem_generica(caminho_a)
            imagem_b = carregar_imagem_generica(caminho_b)
            tensor_a = transformacao_simples(imagem_a).unsqueeze(0).to(device)
            tensor_b = transformacao_simples(imagem_b).unsqueeze(0).to(device)
            _saida_pat, saida_ach = modelo_multivista(tensor_a, tensor_b)
            probs_ach_fontes.append(F.softmax(saida_ach, dim=1)[0])
            tipos_encontrados.add("multivista")
        except Exception:
            pass

    resultado = {"amostra": amostra_id, "dados": "+".join(sorted(tipos_encontrados)) or "nenhum"}

    if probs_pat_fontes:
        prob_pat_final = torch.stack(probs_pat_fontes).mean(dim=0)
        pred_pat = int(prob_pat_final.argmax().item())
        resultado["patologia"] = NOMES_PATOLOGIA[pred_pat]
        conf_pat = prob_pat_final[pred_pat].item() * 100
    else:
        resultado["patologia"] = "indisponível"
        conf_pat = None

    if probs_ach_fontes:
        prob_ach_final = torch.stack(probs_ach_fontes).mean(dim=0)
        pred_ach = int(prob_ach_final.argmax().item())
        resultado["achado"] = NOMES_ACHADO[pred_ach]
        conf_ach = prob_ach_final[pred_ach].item() * 100
    else:
        resultado["achado"] = "indisponível"
        conf_ach = None

    txt_conf_pat = f"{conf_pat:.2f}%" if conf_pat is not None else "N/D"
    txt_conf_ach = f"{conf_ach:.2f}%" if conf_ach is not None else "N/D"
    resultado["conf_patologia"] = txt_conf_pat
    resultado["conf_achado"] = txt_conf_ach

    return resultado


# =============================================================================
# 4. PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnóstico MARIA a partir de uma pasta de DICOMs.")
    parser.add_argument("--pasta", required=True, help="Pasta com os arquivos DICOM (busca recursiva).")
    parser.add_argument("--modelo_recorte", default=str(PASTA_MODELO / "resnet50_multitask_mama_do_zero.pth"))
    parser.add_argument("--modelo_completa", default=str(PASTA_MODELO / "resnet50_multitask_mama_completa.pth"))
    parser.add_argument("--modelo_multivista", default=str(PASTA_MODELO / "resnet50_multivista.pth"))
    parser.add_argument("--saida_csv", default=None, help="Padrão: results.csv na mesma pasta do maria.py.")
    args = parser.parse_args()

    # ---- Checagem amigável: pasta modelo/ e arquivos necessários existem? ----
    if not PASTA_MODELO.is_dir():
        raise SystemExit(
            f"❌ Pasta 'modelo' não encontrada em: {PASTA_MODELO}\n"
            "   Crie essa pasta ao lado do maria.py e coloque nela:\n"
            "     treinar_maria_cbis_ddsm.py, treinar_maria_multivista.py,\n"
            "     resnet50_multitask_mama_do_zero.pth, resnet50_multitask_mama_completa.pth,\n"
            "     resnet50_multivista.pth"
        )
    for caminho_modelo in (args.modelo_recorte, args.modelo_completa, args.modelo_multivista):
        if not Path(caminho_modelo).is_file():
            raise SystemExit(f"❌ Arquivo de modelo não encontrado: {caminho_modelo}")

    pasta = Path(args.pasta).expanduser()
    if not pasta.is_dir():
        raise SystemExit(f"❌ Pasta não encontrada: {pasta}")
    caminho_saida_csv = Path(args.saida_csv).expanduser() if args.saida_csv else PASTA_SCRIPT / "results.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    amostras = montar_amostras(pasta)
    if not amostras:
        raise SystemExit("❌ Nenhuma amostra válida encontrada (nenhum DICOM legível, ou tudo era máscara de ROI).")
    print(f"\n🧬 {len(amostras)} amostra(s) (paciente+lado) identificada(s).")

    print("\n🧠 Carregando modelos...")
    modelo_recorte = MultiTaskResNet50(pesos_imagenet=False, dropout=0.0).to(device)
    modelo_recorte.load_state_dict(torch.load(args.modelo_recorte, map_location=device))
    modelo_recorte.eval()

    modelo_completa = MultiTaskResNet50(pesos_imagenet=False, dropout=0.0).to(device)
    modelo_completa.load_state_dict(torch.load(args.modelo_completa, map_location=device))
    modelo_completa.eval()

    modelo_multivista = MultiViewResNet50(pesos_imagenet=False, dropout=0.0).to(device)
    modelo_multivista.load_state_dict(torch.load(args.modelo_multivista, map_location=device))
    modelo_multivista.eval()

    resultados = []
    print("\n🔎 Diagnosticando...")
    with torch.no_grad():
        for amostra_id, vistas in tqdm(sorted(amostras.items()), desc="MARIA"):
            resultado = diagnosticar_amostra(
                amostra_id, vistas, modelo_recorte, modelo_completa, modelo_multivista,
                CFG.tamanho_imagem, device,
            )
            resultados.append(resultado)

    with open(caminho_saida_csv, "w", newline="", encoding="utf-8") as f:
        campos = ["amostra", "dados", "patologia", "achado", "conf_patologia", "conf_achado"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n💾 Resultado salvo em: {caminho_saida_csv}")
    print("\n" + "=" * 100)
    print(f"{'AMOSTRA':<22}{'DADOS':<28}{'PATOLOGIA':<12}{'ACHADO':<20}{'CONF.PAT':<12}{'CONF.ACH'}")
    print("-" * 100)
    for r in resultados:
        print(f"{r['amostra']:<22}{r['dados']:<28}{r['patologia']:<12}{r['achado']:<20}{r['conf_patologia']:<12}{r['conf_achado']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
