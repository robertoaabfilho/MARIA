"""
================================================================================
 MARIA — Diagnóstico a partir de uma pasta de DICOMs e/ou PNG/JPG de mamografia
================================================================================
Gera um results.csv com, para cada amostra (lesão):
  - LOCALIZAÇÃO (só para mamografias completas): detector Faster R-CNN treinado no CBIS-DDSM
    (modelo_deteccao/). Cada lesão encontrada vira um recorte (caixa + 20 px de margem, como os
    recortes do CBIS-DDSM) e é classificada como abaixo. As coordenadas saem no CSV (pixels da imagem original).
  - PATOLOGIA (Benigno/Maligno): ensemble de 5 Mammo-CLIP (EfficientNet-B2) ajustados no recorte da lesão,
    cada um com TTA de 5 vistas; tela 512 px, escala 1,0, percentis 1-99. Nas lesões de MAMOGRAFIA COMPLETA
    (detectadas ou marcadas), média com mais 5 redes de CONTEXTO (janela da mamografia em volta da lesão).
    Várias vistas de recorte da mesma amostra -> média. Decisão por --limiar (padrão 0,5).
  - DENSIDADE MAMÁRIA (BI-RADS 1-4 = ACR A-D): estimada na mamografia completa (modelo_densidade/) ou
    informada pelo usuário (--planilha). Vale para a mama inteira (paciente + lado) e acompanha cada
    lesão com um aviso do desempenho medido para aquela densidade (a acurácia cai em mama densa).
  - ACHADO (Massa/Nódulo vs Microcalcificação): média de ResNet-50 do recorte (TTA) +
    ResNet-50 multivista (só quando há recorte CC e MLO da mesma amostra).

Identificação do tipo de imagem (máscaras de ROI são ignoradas):
  - DICOM: pela SeriesDescription ("crop"/"full"/"mask", como no CBIS-DDSM). Sem descrição,
    entre imagens do mesmo paciente+lado+vista a maior é a completa e as menores são recortes
    (uma imagem sozinha é tratada como completa). Agrupamento dos recortes: PatientID do CBIS-DDSM
    (P_xxxxx_LADO_VISTA_n) quando reconhecido; senão PatientID + ImageLaterality, por ViewPosition.
  - PNG/JPG: pelo nome: <caso>_recorte_cc, <caso>_recorte_mlo, <caso>_recorte (recortes da lesão) e
    <caso>_completa_cc, <caso>_completa_mlo ou <caso>_completa (mamografia inteira). Tudo com o mesmo <caso>
    é tratado como a MESMA mama (a densidade estimada nas completas vale para os recortes).

Estrutura esperada (ao lado deste arquivo):
    modelo_achado/     resnet50_multitask_mama_do_zero.pth, resnet50_multivista.pth
    modelo_patologia/  mammoclip512_dobra_1..5.pth (recorte) e contexto_dobra_1..5.pth (contexto; opcional: sem eles,
                       as mamografias completas usam só o recorte)
    modelo_deteccao/   detector_lesao.pth, detector_lesao.json   (sem eles, a detecção é pulada)
    modelo_densidade/  densidade.pth, densidade.json             (sem eles, a densidade só vem do --densidade_csv)

Uso (Linux/macOS: python3 | Windows: python ou py):
    python3 maria.py --pasta /caminho/com/dicoms_ou_pngs [--planilha planilha.csv] [--limiar 0.5]
                     [--saida_csv results.csv] [--limiar_deteccao 0.x] [--sem_deteccao]

PLANILHA DO USUÁRIO (opcional; CSV com "," ou ";"; uma linha por imagem/lesão; só "mama" é obrigatória):
    mama              id da mama (paciente + lado). Tudo com a mesma "mama" é a mesma mama.
    densidade         1-4 ou A-D (do laudo). Tem prioridade sobre a estimada pelo modelo.
    vista             CC ou MLO
    arquivo_completa  nome (ou caminho) do arquivo da mamografia inteira
    arquivo_recorte   nome (ou caminho) do arquivo do recorte da lesão
    lesao             id da lesão (quando a mesma mama tem mais de uma)
    x0, y0, x1, y1    caixa da lesão marcada pelo médico na mamografia completa (pixels da imagem original)
    Arquivos citados na planilha não precisam seguir regra de nome. O formato antigo "amostra,densidade" também vale.

SAÍDA (results.csv): amostra, mama, vista (CC/MLO), origem_lesao (recorte fornecido | marcada pelo usuário | detectada pelo
    modelo | nenhuma lesão encontrada), imagens_usadas, patologia, conf_patologia, p_maligno, achado, conf_achado,
    densidade, densidade_origem (informada pelo usuário | inferida pelo modelo), aviso_densidade, concordancia
    (marcação do usuário x detecção do modelo), score_deteccao, imagem, x0, y0, x1, y1.

ATENÇÃO: os Mammo-CLIP finais foram treinados com o teste oficial do CBIS-DDSM (OOF limpo:
acc 76,1%, AUC 0,841). O limiar 0,5 foi calibrado no CBIS-DDSM; em outra base a
sensibilidade/especificidade se desloca. Protótipo de pesquisa, não é dispositivo médico.
================================================================================
"""
import importlib.util
import sys

# Windows: quando a saída vai para um pipe (ex.: o dashboard), o Python usa cp1252 e quebra em acentos/emojis.
for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REQUISITOS = {"torch": "torch", "torchvision": "torchvision", "pydicom": "pydicom", "PIL": "pillow",
              "numpy": "numpy", "tqdm": "tqdm", "efficientnet_pytorch": "efficientnet_pytorch"}
_faltando = [pip for mod, pip in REQUISITOS.items() if importlib.util.find_spec(mod) is None]
if _faltando:
    sys.exit(f"❌ Faltam bibliotecas: {', '.join(_faltando)}\n   Instale com: pip install {' '.join(_faltando)}")

import argparse
import csv
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def progresso(itens, desc, total_geral=None):
    """Itera mostrando o andamento. Pelo dashboard (MARIA_DASHBOARD), imprime uma linha por imagem
    "@@PROGRESSO|etapa|i|n|nome" (o dashboard transforma em barra de progresso); no terminal, usa o tqdm."""
    itens = list(itens)
    if not os.environ.get("MARIA_DASHBOARD"):
        yield from tqdm(itens, desc=desc, disable=not itens)
        return
    for i, item in enumerate(itens, 1):
        nome = item[0] if isinstance(item, tuple) else item
        print(f"@@PROGRESSO|{desc}|{i}|{len(itens)}|{nome}", flush=True)
        yield item

try:
    from pydicom.pixels import apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_voi_lut

PASTA_SCRIPT = Path(__file__).resolve().parent
PASTA_ACHADO = PASTA_SCRIPT / "modelo_achado"
PASTA_PATOLOGIA = PASTA_SCRIPT / "modelo_patologia"
PASTA_DETECCAO = PASTA_SCRIPT / "modelo_deteccao"
PASTA_DENSIDADE = PASTA_SCRIPT / "modelo_densidade"
MARGEM_RECORTE = 20                # px em volta da caixa, como nos recortes do CBIS-DDSM

NOMES_DENSIDADE = {1: "adiposa", 2: "fibroglandular dispersa", 3: "heterogeneamente densa", 4: "extremamente densa"}
# Desempenho do modelo de PATOLOGIA por densidade (OOF limpo do CBIS-DDSM, 2.781 lesões, limiar 0,5):
#   densidade: (n, AUC, sensibilidade %, especificidade %)
DESEMPENHO_DENSIDADE = {1: (446, 0.903, 75.1, 85.7), 2: (1047, 0.853, 75.3, 79.2),
                        3: (858, 0.839, 72.4, 80.4), 4: (430, 0.739, 70.6, 61.8)}

TAMANHO_ACHADO = 224               # entrada das ResNet-50 de achado
MC_TELA, MC_ESCALA, N_DOBRAS = 512, 1.0, 5
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]

NOMES_PATOLOGIA = {0: "Benigno", 1: "Maligno"}
NOMES_ACHADO = {0: "Massa/Nódulo", 1: "Microcalcificação"}
EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg")
SUFIXOS_PNG = [("_recorte_cc", "CC"), ("_recorte_mlo", "MLO"), ("_recorte", "SEM_VISTA"),
               ("_completa_cc", "COMPLETA_CC"), ("_completa_mlo", "COMPLETA_MLO"), ("_completa", "COMPLETA")]
PADRAO_CBIS = re.compile(r"P_(\d+)_(LEFT|RIGHT)_(CC|MLO)(?:_(\d+))?")


# =============================================================================
# 1. MODELOS (mesma arquitetura do treino, só para inferência)
# =============================================================================

class MultiTaskResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        n = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc_patologia = nn.Linear(n, 2)
        self.fc_achado = nn.Linear(n, 2)

    def forward(self, x):
        return self.fc_achado(self.resnet(x))


class MultiViewResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        n = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_patologia = nn.Linear(n * 2, 2)
        self.fc_achado = nn.Linear(n * 2, 2)

    def forward(self, x_cc, x_mlo):
        return self.fc_achado(torch.cat([self.backbone(x_cc), self.backbone(x_mlo)], dim=1))


class EncoderMammoCLIP(nn.Module):
    def __init__(self, variante="b2"):
        super().__init__()
        self.net = EfficientNet.from_name(f"efficientnet-{variante}")
        self.num_features = self.net._conv_head.out_channels

    def forward(self, x):
        return self.net._avg_pooling(self.net.extract_features(x)).flatten(1)


class ClassificadorMammoCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderMammoCLIP()
        n = self.encoder.num_features
        self.fc_patologia = nn.Linear(n, 2)
        self.fc_achado = nn.Linear(n, 2)

    def forward(self, x):
        return self.fc_patologia(self.encoder(x))


class DetectorLesao:
    """Faster R-CNN (1 classe: lesão) sobre a mamografia reduzida para lado maior `lado` px."""

    def __init__(self, pasta, device):
        cfg = json.loads((Path(pasta) / "detector_lesao.json").read_text(encoding="utf-8"))
        self.lado, self.limiar, self.device = cfg["lado"], cfg["limiar"], device
        m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=self.lado, max_size=self.lado,
                                       box_detections_per_img=20, box_score_thresh=0.01)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        self.modelo = carregar(m, Path(pasta) / "detector_lesao.pth", device)

    def detectar(self, arr, limiar):
        """arr float (H, W) -> [(x0, y0, x1, y1, score)] em pixels da imagem original, score >= limiar."""
        H, W = arr.shape
        fator = self.lado / max(H, W)
        img = Image.fromarray(normalizar_percentis(arr)).resize((round(W * fator), round(H * fator)), Image.LANCZOS)
        x = torch.from_numpy(np.array(img)).float().div_(255.0).unsqueeze(0).expand(3, -1, -1).to(self.device)
        r = self.modelo([x])[0]
        return [(*(b / fator).tolist(), float(s)) for b, s in zip(r["boxes"].float().cpu(), r["scores"].float().cpu())
                if s >= limiar]


class DensidadeMammoCLIP(nn.Module):
    """Codificador do Mammo-CLIP (EfficientNet-B2 pré-treinada em mamografias) + 4 classes de densidade."""

    def __init__(self):
        super().__init__()
        self.encoder = EncoderMammoCLIP()
        self.fc = nn.Linear(self.encoder.num_features, 4)

    def forward(self, x):
        return self.fc(self.encoder(x))


def modelo_densidade(arquitetura="resnet50"):
    """Rede de densidade (sem pesos) conforme o campo "arquitetura" do densidade.json."""
    if arquitetura == "mammoclip_b2":
        return DensidadeMammoCLIP()
    if arquitetura != "resnet50":
        raise ValueError(f"arquitetura de densidade desconhecida: {arquitetura}")
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    return m


class ClassificadorDensidade:
    """4 classes (BI-RADS 1-4) sobre a mamografia completa reduzida para tam x tam. A rede (ResNet-50 ou
    Mammo-CLIP) vem do campo "arquitetura" do densidade.json (sem o campo = ResNet-50)."""

    def __init__(self, pasta, device):
        cfg = json.loads((Path(pasta) / "densidade.json").read_text(encoding="utf-8"))
        self.tam, self.device = cfg["tam"], device
        self.modelo = carregar(modelo_densidade(cfg.get("arquitetura", "resnet50")), Path(pasta) / "densidade.pth", device)

    @torch.no_grad()
    def estimar(self, arr):
        """arr float (H, W) -> probabilidades das densidades 1-4 (mesmo caminho do treino: 1536 px e depois tam)."""
        H, W = arr.shape
        f = 1536 / max(H, W)
        img = Image.fromarray(normalizar_percentis(arr)).resize((round(W * f), round(H * f)), Image.LANCZOS)
        img = img.resize((self.tam, self.tam), Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).float().div_(255.0).view(1, 1, self.tam, self.tam).expand(1, 3, -1, -1)
        x = TF.normalize(x, MEDIA, DESVIO).to(self.device)
        p = (F.softmax(self.modelo(x).float(), 1) + F.softmax(self.modelo(x.flip(3)).float(), 1)) / 2
        return p[0].cpu().numpy()


def texto_densidade(d):
    n, auc, sens, espec = DESEMPENHO_DENSIDADE[d]
    txt = (f"BI-RADS {d} ({NOMES_DENSIDADE[d]}): nos testes, AUC {auc:.2f}, sensibilidade {sens:.0f}%, "
           f"especificidade {espec:.0f}%").replace(".", ",")
    if d == 4:
        txt += ". Mama extremamente densa: confiabilidade menor e mais benignos classificados como malignos"
    elif d == 3:
        txt += ". Mama densa: confiabilidade um pouco menor"
    return txt


ACR = {"A": 1, "B": 2, "C": 3, "D": 4, "1": 1, "2": 2, "3": 3, "4": 4}
# nomes de coluna aceitos na planilha do usuário (sem acento, minúsculas; espaços viram "_")
# Formato padrão (colunas 1-11): paciente; idade; arquivo_completa; arquivo_recorte; lado; vista;
# x_inicio; y_inicio; x_fim; y_fim; densidade.
# Também aceita os nomes das bases CBIS-DDSM/CDD-CESM e o formato antigo (mama, lesao, x0..y1).
SINONIMOS = {
    "paciente": ("paciente", "id_paciente", "patient_id", "mama", "amostra", "caso", "id"),
    "idade": ("idade", "idade_anos", "age"),
    "arquivo_completa": ("arquivo_completa", "completa", "imagem_completa", "mamografia"),
    "arquivo_recorte": ("arquivo_recorte", "recorte", "imagem_recorte"),
    "lado": ("lado", "lateralidade", "side", "left_or_right_breast"),
    "vista": ("vista", "incidencia", "view", "image_view"),
    "x0": ("x_inicio", "x0", "xmin", "caixa_esq"), "y0": ("y_inicio", "y0", "ymin", "caixa_topo"),
    "x1": ("x_fim", "x1", "xmax", "caixa_dir"), "y1": ("y_fim", "y1", "ymax", "caixa_base"),
    "densidade": ("densidade", "densidade_mamaria", "densidade_birads", "acr", "breast_density",
                  "breast_density_(acr)"),
    "fabricante": ("fabricante", "equipamento", "aparelho", "machine"),
    "lesao": ("lesao", "id_lesao"),                      # opcional: força o pareamento CC <-> MLO
}
# valores aceitos (sem acento, minúsculas) -> valor padrão
LADOS = {**dict.fromkeys(("direito", "direita", "d", "dir", "r", "right"), "direito"),
         **dict.fromkeys(("esquerdo", "esquerda", "e", "esq", "l", "left"), "esquerdo")}



def _chave_coluna(nome):
    nome = str(nome).strip().lower().replace(" ", "_")
    for a, b in (("ã", "a"), ("á", "a"), ("â", "a"), ("é", "e"), ("ê", "e"), ("í", "i"), ("ó", "o"), ("ô", "o"),
                 ("õ", "o"), ("ú", "u"), ("ç", "c")):
        nome = nome.replace(a, b)
    return nome


def _valor_padrao(texto, tabela, campo, n):
    if not texto:
        return ""
    chave = _chave_coluna(texto)
    if chave not in tabela:
        print(f"⚠️  planilha, linha {n}: {campo} '{texto}' não reconhecido(a), ignorado(a)")
        return ""
    return tabela[chave]


def ler_planilha(caminho):
    """Planilha do usuário (CSV, separador "," ou ";", UTF-8 ou Windows/cp1252) -> lista de linhas.
    Uma linha por imagem (ou por lesão): paciente, idade, arquivos da mamografia completa e do recorte, lado,
    vista, caixa da lesão (x_inicio, y_inicio, x_fim, y_fim, em pixels da imagem original) e densidade.
    Todas opcionais, menos o paciente.
    "mama" (id interno) = paciente + lado."""
    bruto = Path(caminho).read_bytes()
    for cod in ("utf-8-sig", "cp1252"):
        try:
            texto = bruto.decode(cod)
            break
        except UnicodeDecodeError:
            continue
    primeira = texto.splitlines()[0] if texto.strip() else ""
    sep = ";" if primeira.count(";") > primeira.count(",") else ","
    linhas = []
    for n, bruta in enumerate(csv.DictReader(io.StringIO(texto), delimiter=sep), 2):
        b = {_chave_coluna(k): (v or "").strip() for k, v in bruta.items() if k is not None}
        l = {campo: next((b[c] for c in nomes if b.get(c)), "") for campo, nomes in SINONIMOS.items()}
        if not l["paciente"]:
            continue
        l["linha"] = n
        l["lado"] = _valor_padrao(l["lado"], LADOS, "lado", n)
        l["mama"] = f"{l['paciente']}_{l['lado']}" if l["lado"] else l["paciente"]
        if l["densidade"] and l["densidade"].upper() not in ACR:
            print(f"⚠️  planilha, linha {n}: densidade '{l['densidade']}' não reconhecida (use 1-4 ou A-D), ignorada")
        l["densidade"] = ACR.get(l["densidade"].upper())
        if l["vista"] and l["vista"].upper() not in ("CC", "MLO"):
            print(f"⚠️  planilha, linha {n}: vista '{l['vista']}' não reconhecida (use CC ou MLO), ignorada")
        l["vista"] = l["vista"].upper() if l["vista"].upper() in ("CC", "MLO") else ""
        try:
            cx = tuple(float(l[c].replace(",", ".")) for c in ("x0", "y0", "x1", "y1"))
            l["caixa"] = cx if cx[2] > cx[0] and cx[3] > cx[1] else None
            if l["caixa"] is None:
                print(f"⚠️  planilha, linha {n}: caixa inválida (x_fim <= x_inicio ou y_fim <= y_inicio), ignorada")
        except ValueError:
            l["caixa"] = None
        linhas.append(l)
    return linhas


def vista_do_nome(caminho):
    m = re.search(r"(?:^|[^a-z])(cc|mlo)(?:[^a-z]|$)", Path(caminho).stem.lower())
    return m.group(1).upper() if m else ""


def carregar(modelo, caminho, device):
    modelo.load_state_dict(torch.load(caminho, map_location=device))
    return modelo.to(device).eval()


# =============================================================================
# 2. LEITURA DE IMAGEM
# =============================================================================

def ler_imagem(caminho):
    """DICOM (VOI LUT + correção de MONOCHROME1) ou PNG/JPG (cinza) -> (array float32, é_dicom)."""
    if Path(caminho).suffix.lower() in EXTENSOES_IMAGEM:
        return np.asarray(Image.open(caminho).convert("L")).astype(np.float32), False
    ds = pydicom.dcmread(str(caminho))
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr, True


def normalizar_percentis(arr):
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def entrada_achado(arr, eh_dicom):
    """PIL RGB para as ResNet de achado (DICOM normalizado por min-max, como no treino; PNG como está)."""
    if eh_dicom:
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = arr * 255
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def entrada_patologia(arr):
    """Recorte -> tela MC_TELA x MC_TELA uint8: percentis 1-99, redução por fator fixo, centralizada."""
    img = normalizar_percentis(arr)
    h, w = img.shape
    fator = min(MC_ESCALA, MC_TELA / float(max(h, w)))
    nh, nw = max(1, round(h * fator)), max(1, round(w * fator))
    if (nh, nw) != (h, w):
        img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.LANCZOS if fator < 1 else Image.BICUBIC))
    tela = np.zeros((MC_TELA, MC_TELA), np.uint8)
    y0, x0 = (MC_TELA - nh) // 2, (MC_TELA - nw) // 2
    tela[y0:y0 + nh, x0:x0 + nw] = img
    return tela


# =============================================================================
# 3. MONTAGEM DAS AMOSTRAS
# =============================================================================

def ler_metadados(caminho):
    try:
        ds = pydicom.dcmread(str(caminho), stop_before_pixels=True)
    except Exception:
        return None
    return {
        "caminho": str(caminho),
        "patient_id": str(getattr(ds, "PatientID", "")) or "SEM_ID",
        "lado": str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))).upper() or "SEM_LADO",
        "vista": str(getattr(ds, "ViewPosition", "") or "SEM_VISTA").upper(),
        "descricao": str(getattr(ds, "SeriesDescription", "")).lower(),
        "area": int(getattr(ds, "Rows", 0)) * int(getattr(ds, "Columns", 0)),
    }


def montar_amostras(pasta, planilha=None):
    """Lê a pasta (e a planilha do usuário, se houver) e devolve:
      recortes        = {amostra_id: {vista: {"caminho", "area"}}}
      completas       = {imagem_id: {"caminho", "area"}}   (mamografias inteiras: detecção e densidade)
      mama_de         = {amostra_id ou imagem_id: id da mama (paciente + lado)}
      caixas_usuario  = {imagem_id: [((x0, y0, x1, y1), rótulo), ...]}   (lesões marcadas pelo médico)
      dens_informada  = {id da mama ou amostra: densidade 1-4}
      fixos           = amostras com CC + MLO juntados pela coluna "lesao" da planilha
    Os demais recortes saem um por vista; quem decide se CC e MLO são a mesma lesão é parear_por_achado()
    (depois de o modelo classificar o achado de cada vista).
    Arquivos citados na planilha seguem a planilha; os demais seguem as regras automáticas
    (DICOM pelos metadados, PNG/JPG pelo sufixo do nome)."""
    arquivos = [p for p in pasta.rglob("*") if p.is_file()]
    imagens = [p for p in arquivos if p.suffix.lower() in EXTENSOES_IMAGEM]
    dicoms = [p for p in arquivos if p.suffix.lower() == ".dcm"]
    if not dicoms:
        ignorar = EXTENSOES_IMAGEM + (".csv", ".txt", ".py", ".md", ".json", ".xlsx", ".ini", ".db")
        dicoms = [p for p in arquivos if p.suffix.lower() not in ignorar]
    print(f"📂 {len(dicoms)} DICOM candidato(s) e {len(imagens)} PNG/JPG em: {pasta}")

    recortes, completas, mama_de, caixas_usuario, dens_informada = {}, {}, {}, {}, {}
    fixos = set()
    PRIORIDADE = 10 ** 15   # o que vem da planilha vence o que as regras automáticas acharem

    def guardar(amostra, vista, caminho, area, mama=None):
        mama_de[amostra] = mama or amostra
        if vista == "COMPLETA":
            atual = completas.get(amostra)
            if atual is None or area > atual["area"]:
                completas[amostra] = {"caminho": caminho, "area": area}
            return
        atual = recortes.setdefault(amostra, {}).get(vista)
        if atual is None or area > atual["area"]:
            recortes[amostra][vista] = {"caminho": caminho, "area": area}

    # ---- 0) planilha do usuário ----
    citados = set()
    if planilha:
        indice = {}
        for p in arquivos:
            indice.setdefault(p.name.lower(), p)

        def resolver(valor):
            if not valor:
                return None
            v = Path(valor.replace("\\", "/"))
            for cand in (v, pasta / v):
                if cand.is_file():
                    return cand
            return indice.get(v.name.lower())

        # uma imagem completa = um id, mesmo que apareça em várias linhas (a vista vem de qualquer linha que a informe)
        vista_arq, id_arq, arq_id = {}, {}, {}
        for l in planilha:
            c = resolver(l["arquivo_completa"])
            if c is not None and l["vista"]:
                vista_arq.setdefault(c.resolve(), l["vista"])

        def id_completa(mama, arquivo):
            chave = arquivo.resolve()
            if chave not in id_arq:
                vista = vista_arq.get(chave) or vista_do_nome(arquivo)
                ident = f"{mama}_{vista.lower()}" if vista else f"{mama}_{arquivo.stem}"
                if ident in arq_id:                       # outra imagem da mesma mama com a mesma vista
                    ident = f"{ident}_{arquivo.stem}"
                id_arq[chave], arq_id[ident] = ident, chave
            return id_arq[chave]

        faltando = []
        for l in planilha:
            mama = l["mama"]
            if l["densidade"]:
                dens_informada[mama] = l["densidade"]
            completa, recorte = resolver(l["arquivo_completa"]), resolver(l["arquivo_recorte"])
            for nome, valor, achado in (("completa", l["arquivo_completa"], completa),
                                        ("recorte", l["arquivo_recorte"], recorte)):
                if valor and achado is None:
                    faltando.append(f"linha {l['linha']}: {nome} '{valor}'")
            if completa is not None:
                citados.add(completa.resolve())
                img_id = id_completa(mama, completa)
                guardar(img_id, "COMPLETA", str(completa), PRIORIDADE, mama)
                if l["caixa"]:
                    ja = caixas_usuario.setdefault(img_id, [])
                    rotulo = l["lesao"] or str(len(ja) + 1)
                    ja.append((l["caixa"], rotulo))
            elif l["caixa"]:
                print(f"⚠️  planilha, linha {l['linha']}: caixa sem arquivo_completa, ignorada")
            if recorte is not None:
                citados.add(recorte.resolve())
                vista = l["vista"] or vista_do_nome(recorte) or "SEM_VISTA"
                if l["lesao"]:
                    amostra = f"{mama}_{l['lesao']}"
                    fixos.add(amostra)
                else:
                    amostra, k = f"{mama}_{vista.lower()}", 2
                    while amostra in recortes:
                        amostra, k = f"{mama}_{vista.lower()}{k}", k + 1
                guardar(amostra, vista, str(recorte), PRIORIDADE, mama)
        if faltando:
            print("⚠️  arquivo(s) da planilha não encontrado(s) na pasta:")
            for f in faltando:
                print(f"    - {f}")
        n_cx = sum(len(v) for v in caixas_usuario.values())
        print(f"   -> planilha: {len(planilha)} linha(s), {len(citados)} arquivo(s), {n_cx} lesão(ões) marcada(s), "
              f"{len(dens_informada)} densidade(s) informada(s)")

    # ---- 1) DICOM (pelos metadados) ----
    metadados = [m for m in map(ler_metadados, dicoms) if m is not None and Path(m["caminho"]).resolve() not in citados]
    print(f"   -> {len(metadados)} DICOM(s) lido(s) pelas regras automáticas.")
    indefinidos = {}
    for m in metadados:
        d = m["descricao"]
        m["tipo"] = "mascara" if "mask" in d else "recorte" if "crop" in d else "completa" if "full" in d else None
        if m["tipo"] is None:
            indefinidos.setdefault((m["patient_id"], m["lado"], m["vista"]), []).append(m)
    for grupo in indefinidos.values():   # sem descrição: a maior é a completa, as demais são recortes
        grupo.sort(key=lambda m: m["area"], reverse=True)
        for i, m in enumerate(grupo):
            m["tipo"] = "completa" if i == 0 else "recorte"

    for m in metadados:
        if m["tipo"] == "mascara":
            continue
        cbis = PADRAO_CBIS.search(m["patient_id"])
        if m["tipo"] == "completa":
            mama = f"P{cbis[1]}_{cbis[2]}" if cbis else f"{m['patient_id']}_{m['lado']}"
            guardar(f"{mama}_{cbis[3] if cbis else m['vista']}", "COMPLETA", m["caminho"], m["area"], mama)
        elif cbis:
            # o número da anotação no CBIS ("abnormality id") NÃO identifica a mesma lesão nas duas vistas
            num, lado, vista, lesao = cbis.groups()
            guardar(f"P{num}_{lado}_{vista.lower()}{lesao or ''}", vista, m["caminho"], m["area"], f"P{num}_{lado}")
        else:
            mama = f"{m['patient_id']}_{m['lado']}"
            guardar(f"{mama}_{m['vista'].lower()}", m["vista"], m["caminho"], m["area"], mama)

    # ---- 2) PNG/JPG (pelo sufixo do nome) ----
    ignorados_png = []
    for caminho in imagens:
        if caminho.resolve() in citados:
            continue
        nome = caminho.stem
        for sufixo, vista in SUFIXOS_PNG:
            if nome.lower().endswith(sufixo):
                caso = nome[: -len(sufixo)]
                if vista.startswith("COMPLETA_"):      # uma completa por vista, todas da mesma mama (caso)
                    guardar(f"{caso}_{vista[9:].lower()}", "COMPLETA", str(caminho), 0, caso)
                else:
                    guardar(f"{caso}_{vista.lower()}" if vista in ("CC", "MLO") else caso, vista, str(caminho), 0, caso)
                break
        else:
            ignorados_png.append(caminho.name)
    if ignorados_png:
        print("⚠️  PNG/JPG ignorado(s): não citado(s) na planilha e o nome não termina em _recorte_cc, "
              "_recorte_mlo, _recorte, _completa_cc, _completa_mlo ou _completa:")
        for nome in ignorados_png:
            print(f"    - {nome}")
    return recortes, completas, mama_de, caixas_usuario, dens_informada, fixos


def parear_por_achado(recortes, mama_de, fixos, m_recorte, device):
    """Junta CC + MLO (análise multivista) quando a mama tem exatamente UM recorte CC e UM MLO com o MESMO
    achado previsto pelo modelo (massa/nódulo ou microcalcificação). Achados diferentes nas duas vistas,
    ou mais de um do mesmo tipo numa vista, ficam separados (cada recorte é um caso).
    -> (nº de pares, {amostra: aviso}) — o aviso marca CC e MLO únicas da mama com achados diferentes."""
    grupos, por_mama = {}, {}
    for a, vistas in recortes.items():
        if a in fixos or len(vistas) != 1 or next(iter(vistas)) not in ("CC", "MLO"):
            continue
        vista, info = next(iter(vistas.items()))
        try:
            arr, eh_dicom = ler_imagem(info["caminho"])
            classe = int(prob_achado_recorte(m_recorte, entrada_achado(arr, eh_dicom), device).argmax())
        except Exception:
            continue
        grupos.setdefault((mama_de[a], classe), {}).setdefault(vista, []).append(a)
        por_mama.setdefault(mama_de[a], {}).setdefault(vista, []).append((a, classe))
    divergentes = {}
    for mama, vs in por_mama.items():
        if sorted(vs) == ["CC", "MLO"] and len(vs["CC"]) == len(vs["MLO"]) == 1 and vs["CC"][0][1] != vs["MLO"][0][1]:
            for a, _ in (vs["CC"][0], vs["MLO"][0]):
                divergentes[a] = ("achado previsto diferente na CC e na MLO: vistas analisadas separadamente "
                                  "(correlacione as vistas)")
    pares = 0
    for (mama, classe), por_vista in grupos.items():
        if sorted(por_vista) != ["CC", "MLO"] or any(len(v) != 1 for v in por_vista.values()):
            continue
        (cc,), (mlo,) = por_vista["CC"], por_vista["MLO"]
        novo = f"{mama}_{'massa' if classe == 0 else 'calcificacao'}"
        while novo in recortes:
            novo += "_par"
        recortes[novo] = {**recortes.pop(cc), **recortes.pop(mlo)}
        mama_de[novo] = mama
        pares += 1
    return pares, divergentes


def id_usuario(imagem_id, rotulo):
    """Id da lesão marcada pelo usuário numa mamografia completa."""
    rotulo = str(rotulo)
    return f"{imagem_id}_usuario{rotulo}" if rotulo.isdigit() else f"{imagem_id}_usuario_{rotulo}"


# =============================================================================
# 4. INFERÊNCIA
# =============================================================================

def tta(x):
    """x (1,C,H,W) -> lote (5,C,H,W): original, flips e rotações de ±10°."""
    return torch.cat([x, TF.hflip(x), TF.vflip(x),
                      TF.rotate(x, 10, interpolation=InterpolationMode.BILINEAR),
                      TF.rotate(x, -10, interpolation=InterpolationMode.BILINEAR)])


BASE_ACHADO = transforms.Compose([transforms.Resize((TAMANHO_ACHADO, TAMANHO_ACHADO)), transforms.ToTensor(),
                                  transforms.Normalize(MEDIA, DESVIO)])

# valores da coluna "origem_lesao"
ORIGEM_RECORTE, ORIGEM_USUARIO, ORIGEM_MODELO, ORIGEM_NENHUMA = (
    "recorte fornecido", "marcada pelo usuário", "detectada pelo modelo", "nenhuma lesão encontrada")


# contexto da execução (preenchido no main): calibração, pasta dos mapas de calor, versão do modelo
CONTEXTO = {"cal": None, "pasta_mapas": None, "mapas": True, "modelos_ctx": None}
ENTRADA_RECORTE, ENTRADA_CONTEXTO = "recorte", "recorte + contexto"
CTX_ESCALA = 0.30          # janela de contexto: pixel original -> pixel da tela (mesma do treino do modelo de contexto)


def _p_redes(modelos, tela):
    """tela uint8 (MC_TELA x MC_TELA) -> P(maligno) de cada rede (média das 5 variações do TTA)."""
    x = torch.from_numpy(tela).float().div_(255.0).view(1, 1, MC_TELA, MC_TELA)
    lote = TF.normalize(tta(x).repeat(1, 3, 1, 1), MEDIA, DESVIO).to(next(modelos[0].parameters()).device)
    return torch.stack([F.softmax(m(lote).float(), 1)[:, 1].mean() for m in modelos])


def prob_maligno(modelos_mc, arr, device):
    """-> (P média das 5 redes, desvio-padrão entre as redes = incerteza). Cada rede usa 5 variações (TTA)."""
    ps = _p_redes(modelos_mc, entrada_patologia(arr))
    return float(ps.mean()), float(ps.std(unbiased=False))


def normalizar_mamografia(arr, p_baixo=1.0, p_alto=99.0):
    """Mamografia inteira float -> uint8 com percentis do TECIDO (fundo preto); entrada da janela de contexto."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    tecido = arr > (mn + 0.05 * (mx - mn))
    amostra = arr[tecido] if tecido.sum() > 1000 else arr.ravel()
    lo, hi = np.percentile(amostra, [p_baixo, p_alto])
    if hi <= lo:
        lo, hi = mn, mx
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~tecido] = 0.0
    return (out * 255.0).astype(np.uint8)


def janela_contexto(img_u8, caixa, tamanho=MC_TELA, escala=CTX_ESCALA, folga=1.25):
    """Janela quadrada da mamografia inteira centrada na lesão (lado tamanho/escala px originais, cresce se a lesão
    não couber), reduzida para tamanho x tamanho. caixa = (x0, y0, x1, y1) em pixels originais."""
    x0, y0, x1, y1 = caixa
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    lado = int(round(max(tamanho / escala, max(y1 - y0, x1 - x0) * folga)))
    wy0, wx0 = int(round(cy - lado / 2.0)), int(round(cx - lado / 2.0))
    canvas = np.zeros((lado, lado), dtype=np.uint8)
    H, W = img_u8.shape
    sy0, sx0, sy1, sx1 = max(0, wy0), max(0, wx0), min(H, wy0 + lado), min(W, wx0 + lado)
    if sy1 > sy0 and sx1 > sx0:
        canvas[sy0 - wy0:sy1 - wy0, sx0 - wx0:sx1 - wx0] = img_u8[sy0:sy1, sx0:sx1]
    fator = tamanho / float(lado)
    return np.asarray(Image.fromarray(canvas).resize((tamanho, tamanho), Image.LANCZOS if fator < 1 else Image.BICUBIC))


def prob_maligno_contexto(modelos_mc, modelos_ctx, recorte, img_u8, caixa):
    """Caixa numa mamografia completa: média das redes de RECORTE (lesão) e de CONTEXTO (janela da mamografia em
    volta da lesão). -> (P média, desvio-padrão entre todas as redes). Validação externa (CMMD e CDD-CESM): a média
    é melhor ou igual ao recorte sozinho; o contexto sozinho não (quase nunca chama de maligno fora do CBIS)."""
    ps = torch.cat([_p_redes(modelos_mc, entrada_patologia(recorte)), _p_redes(modelos_ctx, janela_contexto(img_u8, caixa))])
    return float(ps.mean()), float(ps.std(unbiased=False))


def calibrar(p, chave="patologia"):
    """P(maligno) calibrada (Platt; ver modelo/calibracao.json). chave = "patologia" (só recorte) ou
    "patologia_contexto" (recorte + contexto); sem calibração própria do contexto, usa a do recorte."""
    todas = CONTEXTO["cal"] or {}
    cal = todas.get(chave) or todas.get("patologia")
    if not cal or p is None:
        return p
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(1 / (1 + np.exp(-(cal["a"] * np.log(p / (1 - p)) + cal["b"]))))


def mapa_calor(modelos_mc, arr, device, nome):
    """Grad-CAM da classe 'maligno' (média das 5 redes) sobre o recorte: onde o modelo 'olhou'.
    Salva um PNG lado a lado (recorte | mapa) e devolve o caminho."""
    if not CONTEXTO["mapas"] or CONTEXTO["pasta_mapas"] is None:
        return ""
    x = torch.from_numpy(entrada_patologia(arr)).float().div_(255.0).view(1, 1, MC_TELA, MC_TELA)
    x = TF.normalize(x.repeat(1, 3, 1, 1), MEDIA, DESVIO).to(device)
    cams = []
    with torch.enable_grad():
        for m in modelos_mc:
            f = m.encoder.net.extract_features(x)
            f.retain_grad()
            logit = m.fc_patologia(m.encoder.net._avg_pooling(f).flatten(1))[0, 1]
            m.zero_grad(set_to_none=True)
            logit.backward()
            cam = F.relu((f.grad.mean(dim=(2, 3), keepdim=True) * f).sum(1))[0]
            cams.append((cam / (cam.max() + 1e-8)).detach())
    cam = F.interpolate(torch.stack(cams).mean(0)[None, None], size=(MC_TELA, MC_TELA), mode="bilinear",
                        align_corners=False)[0, 0].cpu().numpy()
    # só a região do recorte dentro da tela (a tela é preta em volta)
    h, w = arr.shape
    fator = min(MC_ESCALA, MC_TELA / float(max(h, w)))
    nh, nw = max(1, round(h * fator)), max(1, round(w * fator))
    y0, x0 = (MC_TELA - nh) // 2, (MC_TELA - nw) // 2
    cinza = entrada_patologia(arr)[y0:y0 + nh, x0:x0 + nw].astype(np.float32) / 255
    c = cam[y0:y0 + nh, x0:x0 + nw]
    c = (c - c.min()) / (c.max() - c.min() + 1e-8)
    cor = np.stack([np.clip(1.5 - np.abs(4 * c - 3), 0, 1), np.clip(1.5 - np.abs(4 * c - 2), 0, 1),
                    np.clip(1.5 - np.abs(4 * c - 1), 0, 1)], -1)
    base = np.repeat(cinza[..., None], 3, -1)
    sobre = base * (1 - 0.45 * c[..., None]) + cor * 0.45 * c[..., None]
    painel = np.concatenate([base, np.ones((nh, 6, 3)), sobre], 1)
    destino = CONTEXTO["pasta_mapas"] / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', nome)}.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(painel, 0, 1) * 255).astype(np.uint8)).save(destino)
    return str(destino)


def metadados_imagem(caminho):
    """Informações para qualidade e subgrupos: modalidade, região, fabricante, idade; e se o PNG/JPG é colorido."""
    meta = {"modalidade": "", "regiao": "", "fabricante": "", "idade": "", "colorida": False}
    if Path(caminho).suffix.lower() in EXTENSOES_IMAGEM:
        try:
            img = Image.open(caminho)
            if img.mode in ("RGB", "RGBA"):
                a = np.asarray(img.convert("RGB").resize((128, 128))).astype(np.int16)
                meta["colorida"] = bool(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() > 8)
        except Exception:
            pass
        return meta
    try:
        ds = pydicom.dcmread(str(caminho), stop_before_pixels=True)
    except Exception:
        return meta
    meta["modalidade"] = str(getattr(ds, "Modality", "") or "").upper()
    meta["regiao"] = str(getattr(ds, "BodyPartExamined", "") or "").upper()
    meta["fabricante"] = " ".join(str(getattr(ds, t, "") or "").strip()
                                  for t in ("Manufacturer", "ManufacturerModelName")).strip()
    idade = str(getattr(ds, "PatientAge", "") or "")
    if re.fullmatch(r"\d{3}Y", idade):
        meta["idade"] = str(int(idade[:3]))
    else:
        nasc, exame = str(getattr(ds, "PatientBirthDate", "") or ""), str(getattr(ds, "StudyDate", "") or "")
        if re.fullmatch(r"\d{8}", nasc) and re.fullmatch(r"\d{8}", exame):
            meta["idade"] = str(int(exame[:4]) - int(nasc[:4]) - (exame[4:] < nasc[4:]))
    return meta


def checar_qualidade(arr, meta, tipo):
    """-> lista de (nível, motivo); nível 'não usar' ou 'atenção'. tipo = 'completa' ou 'recorte'."""
    avisos = []
    if meta.get("modalidade") and meta["modalidade"] != "MG":
        avisos.append(("não usar", f"modalidade DICOM '{meta['modalidade']}' (não é mamografia, MG)"))
    if meta.get("regiao") and not any(t in str(meta["regiao"]).upper() for t in ("BREAST", "MAMA")):   # "RIGHT BREAST" etc.
        avisos.append(("atenção", f"região examinada '{meta['regiao']}' (esperado: BREAST)"))
    if meta.get("colorida"):
        avisos.append(("não usar", "imagem colorida (não parece mamografia)"))
    h, w = arr.shape
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        avisos.append(("não usar", "imagem sem contraste (praticamente uniforme)"))
        return avisos
    if tipo == "recorte" and min(h, w) < 32:
        avisos.append(("não usar", f"recorte muito pequeno ({w}×{h} px)"))
    if tipo == "completa":
        if max(h, w) < 1000:
            avisos.append(("atenção", f"resolução baixa para mamografia completa ({w}×{h} px)"))
        frac = float((arr > lo + 0.1 * (hi - lo)).mean())
        if not 0.05 <= frac <= 0.97:
            avisos.append(("atenção", "mama não identificada com clareza (imagem fora do padrão)"))
    sat = float((arr >= hi).mean()) if tipo == "recorte" else float((arr >= arr.max()).mean())
    if sat > 0.30:
        avisos.append(("atenção", "imagem muito saturada (muitos pixels no brilho máximo)"))
    return avisos


def prob_achado_recorte(modelo, img, device):
    variantes = [img, TF.hflip(img), TF.vflip(img), TF.rotate(img, 10), TF.rotate(img, -10)]
    lote = torch.stack([BASE_ACHADO(v) for v in variantes]).to(device)
    return F.softmax(modelo(lote), 1).mean(0)


def prob_achado_multivista(modelo, img_a, img_b, device):
    a = BASE_ACHADO(img_a).unsqueeze(0).to(device)
    b = BASE_ACHADO(img_b).unsqueeze(0).to(device)
    return F.softmax(modelo(a, b), 1)[0]


def prioridade_de(p_cal, desvio, nivel_qualidade, chave="prioridade"):
    """Prioridade de leitura para o radiologista (o MARIA prioriza; quem diagnostica é o radiologista).
    chave "prioridade_contexto" (limites próprios do recorte + contexto) cai em "prioridade" se não existir."""
    if nivel_qualidade == "não usar":
        return "não avaliar (qualidade)"
    if p_cal is None:
        return "indisponível"
    todas = CONTEXTO["cal"] or {}
    cfg = todas.get(chave) or todas.get("prioridade", {"alta": 0.5, "media": 0.2})
    return "alta" if p_cal >= cfg["alta"] else "média" if p_cal >= cfg["media"] else "baixa"


def resultado(amostra_id, origem, imagens_usadas, ps_mal, fontes_ach, limiar, avisos=(), mapas=(), meta=None, **extra):
    """ps_mal = [(P média, desvio)] por vista; avisos = [(nível, motivo)] de qualidade."""
    meta = meta or {}
    r = {"amostra": amostra_id, "origem_lesao": origem, "imagens_usadas": imagens_usadas,
         "idade": meta.get("idade", ""), "fabricante": meta.get("fabricante", ""),
         "mapa_calor": ";".join(m for m in mapas if m), **extra}
    avisos = list(avisos)
    p_cal = desvio = None
    if ps_mal:
        p = float(np.mean([m for m, _ in ps_mal]))
        desvio = float(max(d for _, d in ps_mal))
        com_ctx = extra.get("patologia_entrada") == ENTRADA_CONTEXTO
        p_cal = calibrar(p, "patologia_contexto" if com_ctx else "patologia")
        if limiar is None:      # automático: o limite da prioridade ALTA da entrada ("Maligno" = prioridade alta)
            todas = CONTEXTO["cal"] or {}
            limiar = (todas.get("prioridade_contexto" if com_ctx else "prioridade")
                      or todas.get("prioridade") or {"alta": 0.5})["alta"]
        pred = int(p_cal >= limiar)                       # sobre a P CALIBRADA
        r.update(patologia=NOMES_PATOLOGIA[pred], conf_patologia=f"{(p_cal if pred else 1 - p_cal) * 100:.2f}%",
                 p_maligno=f"{p:.4f}", p_maligno_calibrada=f"{p_cal:.4f}", incerteza=f"{desvio:.3f}",
                 limiar_maligno=f"{limiar:.2f}")
        inc = (CONTEXTO["cal"] or {}).get("incerteza", {"desvio_maximo": 0.15, "faixa_incerta": [0.35, 0.65]})
        if desvio > inc["desvio_maximo"]:
            avisos.append(("atenção", f"resultado incerto: as redes discordam (desvio {desvio:.2f})".replace(".", ",")))
        elif inc["faixa_incerta"][0] <= p_cal <= inc["faixa_incerta"][1]:
            avisos.append(("atenção", f"resultado incerto: probabilidade intermediária ({p_cal:.2f})".replace(".", ",")))
    else:
        r.update(patologia="indisponível", conf_patologia="N/D", p_maligno="N/D", p_maligno_calibrada="N/D",
                 incerteza="N/D")
    if fontes_ach:
        prob = torch.stack(fontes_ach).mean(0)
        pred = int(prob.argmax())
        r.update(achado=NOMES_ACHADO[pred], conf_achado=f"{prob[pred].item() * 100:.2f}%")
    else:
        r.update(achado="indisponível", conf_achado="N/D")
    nivel = "não usar" if any(n == "não usar" for n, _ in avisos) else "atenção" if avisos else "ok"
    r.update(qualidade=nivel, avisos_qualidade="; ".join(dict.fromkeys(m for _, m in avisos)),
             prioridade=prioridade_de(p_cal, desvio, nivel, "prioridade_contexto"
                                      if extra.get("patologia_entrada") == ENTRADA_CONTEXTO else "prioridade"))
    return r


def diagnosticar(amostra_id, vistas, m_recorte, m_multivista, modelos_mc, device, limiar):
    usadas, ps_mal, ps_ach, imgs, avisos, mapas, meta = [], [], [], {}, [], [], {}
    for vista, info in vistas.items():
        try:
            arr, eh_dicom = ler_imagem(info["caminho"])
            meta = metadados_imagem(info["caminho"]) if not meta.get("fabricante") else meta
            avisos += checar_qualidade(arr, meta, "recorte")
            imgs[vista] = entrada_achado(arr, eh_dicom)
            ps_mal.append(prob_maligno(modelos_mc, arr, device))
            ps_ach.append(prob_achado_recorte(m_recorte, imgs[vista], device))
            mapas.append(mapa_calor(modelos_mc, arr, device, f"{amostra_id}_{vista}"))
            usadas.append("recorte" if vista == "SEM_VISTA" else f"recorte {vista}")
        except Exception as e:
            print(f"   ⚠️  {amostra_id}: falha no recorte {vista} ({e})")
            avisos.append(("não usar", f"falha ao ler o recorte {vista}"))

    fontes_ach = [torch.stack(ps_ach).mean(0)] if ps_ach else []
    nomeadas = [v for v in imgs if v != "SEM_VISTA"]
    if len(nomeadas) >= 2:
        try:
            fontes_ach.append(prob_achado_multivista(m_multivista, imgs[nomeadas[0]], imgs[nomeadas[1]], device))
            usadas.append("multivista (CC+MLO)")
        except Exception as e:
            print(f"   ⚠️  {amostra_id}: falha no multivista ({e})")
    return resultado(amostra_id, ORIGEM_RECORTE, " + ".join(sorted(usadas)) or "nenhuma", ps_mal, fontes_ach, limiar,
                     avisos=avisos, mapas=mapas, meta=meta, patologia_entrada=ENTRADA_RECORTE,
                     imagem=vistas[next(iter(vistas))]["caminho"], concordancia="",
                     vista=" + ".join(sorted(v for v in vistas if v != "SEM_VISTA")))


def vista_da_imagem(imagem_id, caminho):
    """CC/MLO de uma mamografia completa: pelo id (…_cc / …_mlo) ou pelo nome do arquivo."""
    fim = imagem_id.rsplit("_", 1)[-1].upper()
    return fim if fim in ("CC", "MLO") else vista_do_nome(caminho)


def coincide(a, b):
    """Mesma lesão? IoU >= 0,2 ou o centro de uma caixa dentro da outra."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    dentro = lambda c, d: d[0] <= (c[0] + c[2]) / 2 <= d[2] and d[1] <= (c[1] + c[3]) / 2 <= d[3]
    return (uniao > 0 and inter / uniao >= 0.2) or dentro(a, b) or dentro(b, a)


def diagnosticar_completa(imagem_id, caminho, detector, limiar_det, caixas_usr, m_recorte, modelos_mc, device, limiar):
    """Mamografia completa: classifica as lesões marcadas pelo usuário e as detectadas pelo modelo
    (recorte = caixa + MARGEM_RECORTE px) e compara as duas marcações."""
    try:
        arr, eh_dicom = ler_imagem(caminho)
    except Exception as e:
        print(f"   ⚠️  {imagem_id}: falha ao ler a imagem ({e})")
        return []
    meta = metadados_imagem(caminho)
    avisos_img = checar_qualidade(arr, meta, "completa")
    H, W = arr.shape
    do_modelo = []
    if detector is not None and not any(n == "não usar" for n, _ in avisos_img):
        try:
            do_modelo = sorted(detector.detectar(arr, limiar_det), key=lambda c: -c[4])
        except Exception as e:
            print(f"   ⚠️  {imagem_id}: falha na detecção ({e})")
    do_usuario = [((max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)), lesao)
                  for (x0, y0, x1, y1), lesao in (caixas_usr or [])]
    nome_img = Path(caminho).name
    vista = vista_da_imagem(imagem_id, caminho)

    modelos_ctx = CONTEXTO.get("modelos_ctx")
    img_u8 = normalizar_mamografia(arr) if modelos_ctx else None
    entrada = ENTRADA_CONTEXTO if modelos_ctx else ENTRADA_RECORTE

    def classificar(x0, y0, x1, y1, nome):
        cx0, cy0 = max(0, int(x0) - MARGEM_RECORTE), max(0, int(y0) - MARGEM_RECORTE)
        cx1, cy1 = min(W, int(np.ceil(x1)) + MARGEM_RECORTE), min(H, int(np.ceil(y1)) + MARGEM_RECORTE)
        recorte = arr[cy0:cy1, cx0:cx1]
        p_mal = (prob_maligno_contexto(modelos_mc, modelos_ctx, recorte, img_u8, (x0, y0, x1, y1)) if modelos_ctx
                 else prob_maligno(modelos_mc, recorte, device))
        return ([p_mal],
                [prob_achado_recorte(m_recorte, entrada_achado(recorte, eh_dicom), device)],
                avisos_img + checar_qualidade(recorte, {}, "recorte"), [mapa_calor(modelos_mc, recorte, device, nome)])

    linhas = []
    for n, (cx, lesao) in enumerate(do_usuario, 1):
        if detector is None:
            conc = "detecção do modelo desligada"
        else:
            conc = ("o modelo também detectou" if any(coincide(m[:4], cx) for m in do_modelo)
                    else "o modelo NÃO detectou")
        amostra = id_usuario(imagem_id, lesao or n)
        p, a, av, mp = classificar(*cx, amostra)
        linhas.append(resultado(amostra, ORIGEM_USUARIO, f"mamografia completa ({nome_img})",
                                p, a, limiar, avisos=av, mapas=mp, meta=meta, imagem=caminho, x0=round(cx[0]),
                                y0=round(cx[1]), x1=round(cx[2]), y1=round(cx[3]), score_deteccao="",
                                concordancia=conc, _imagem=imagem_id, vista=vista,
                                patologia_entrada=entrada))
    for n, m in enumerate(do_modelo, 1):
        if do_usuario:
            conc = ("confere com marcação do usuário" if any(coincide(m[:4], u) for u, _ in do_usuario)
                    else "não marcada pelo usuário")
        else:
            conc = ""
        amostra = f"{imagem_id}_modelo{n}"
        p, a, av, mp = classificar(*m[:4], amostra)
        linhas.append(resultado(amostra, ORIGEM_MODELO, f"mamografia completa ({nome_img})",
                                p, a, limiar, avisos=av, mapas=mp, meta=meta, imagem=caminho, x0=round(m[0]),
                                y0=round(m[1]), x1=round(m[2]), y1=round(m[3]), score_deteccao=f"{m[4]:.3f}",
                                concordancia=conc, _imagem=imagem_id, vista=vista,
                                patologia_entrada=entrada))
    if not linhas:
        nivel = "não usar" if any(n == "não usar" for n, _ in avisos_img) else "atenção" if avisos_img else "ok"
        linhas.append({"amostra": imagem_id, "origem_lesao": ORIGEM_NENHUMA,
                       "imagens_usadas": f"mamografia completa ({nome_img})", "imagem": caminho,
                       "patologia": "sem lesão detectada", "achado": "—", "conf_patologia": "N/D",
                       "conf_achado": "N/D", "p_maligno": "N/D", "p_maligno_calibrada": "N/D", "incerteza": "N/D",
                       "concordancia": "", "_imagem": imagem_id, "vista": vista,
                       "qualidade": nivel, "avisos_qualidade": "; ".join(m for _, m in avisos_img),
                       "prioridade": "não avaliar (qualidade)" if nivel == "não usar" else "baixa (nada detectado)",
                       "idade": meta.get("idade", ""), "fabricante": meta.get("fabricante", ""), "mapa_calor": ""})
    return linhas


# =============================================================================
# 5. PRINCIPAL
# =============================================================================

CAMPOS_SAIDA = ["amostra", "mama", "vista", "prioridade", "p_maligno_calibrada", "incerteza", "qualidade",
                "avisos_qualidade", "origem_lesao", "imagens_usadas", "patologia", "conf_patologia", "p_maligno",
                "achado", "conf_achado", "densidade", "densidade_origem", "aviso_densidade", "concordancia",
                "score_deteccao", "patologia_entrada", "limiar_maligno", "idade", "fabricante", "mapa_calor", "versao_modelo", "calibracao", "imagem", "x0", "y0", "x1", "y1"]


def main():
    ap = argparse.ArgumentParser(description="MARIA: detecção, patologia (Mammo-CLIP), achado (ResNet-50) e densidade.")
    ap.add_argument("--pasta", required=True, help="Pasta com DICOMs e/ou PNG/JPG (busca recursiva).")
    ap.add_argument("--planilha", default=None,
                    help="CSV do usuário: mama, densidade, vista, arquivo_completa, arquivo_recorte, x0, y0, x1, y1 "
                         "(todas opcionais exceto mama). Densidade e lesões informadas têm prioridade.")
    ap.add_argument("--densidade_csv", default=None, help=argparse.SUPPRESS)   # nome antigo de --planilha
    ap.add_argument("--limiar", type=float, default=None,
                    help="limiar da P(maligno) CALIBRADA para 'Maligno' (padrão: automático = o limite da prioridade "
                         "alta: 0,50 no recorte e 0,37 no recorte + contexto)")
    ap.add_argument("--limiar_deteccao", type=float, default=None,
                    help="score mínimo de uma caixa do detector (padrão: o do detector_lesao.json)")
    ap.add_argument("--sem_deteccao", action="store_true", help="não procura lesões nas mamografias completas")
    ap.add_argument("--saida_csv", default=None, help="Padrão: results.csv ao lado do maria.py.")
    ap.add_argument("--pasta_pesos", default=str(PASTA_PATOLOGIA), help="pasta com mammoclip512_dobra_{1..5}.pth")
    ap.add_argument("--pasta_deteccao", default=str(PASTA_DETECCAO), help="pasta com detector_lesao.pth/.json")
    ap.add_argument("--pasta_densidade", default=str(PASTA_DENSIDADE), help="pasta com densidade.pth/.json")
    ap.add_argument("--modelo_recorte", default=str(PASTA_ACHADO / "resnet50_multitask_mama_do_zero.pth"))
    ap.add_argument("--modelo_multivista", default=str(PASTA_ACHADO / "resnet50_multivista.pth"))
    ap.add_argument("--sem_mapas", action="store_true", help="não gera os mapas de calor (mais rápido, sobretudo na CPU)")
    args = ap.parse_args()

    pesos_mc = [Path(args.pasta_pesos) / f"mammoclip512_dobra_{k}.pth" for k in range(1, N_DOBRAS + 1)]
    faltando = [str(c) for c in [args.modelo_recorte, args.modelo_multivista, *pesos_mc] if not Path(c).is_file()]
    if faltando:
        raise SystemExit("❌ Arquivo(s) de modelo não encontrado(s):\n    " + "\n    ".join(faltando))
    pasta = Path(args.pasta).expanduser().resolve()   # caminhos absolutos no resultado (o dashboard reabre as imagens)
    if not pasta.is_dir():
        raise SystemExit(f"❌ Pasta não encontrada: {pasta}")
    arq_planilha = args.planilha or args.densidade_csv
    planilha = ler_planilha(Path(arq_planilha).expanduser()) if arq_planilha else None
    saida = Path(args.saida_csv).expanduser() if args.saida_csv else PASTA_SCRIPT / "results.csv"

    # rastreabilidade: versão do modelo (hash dos pesos) conferida contra o manifesto
    sys.path.insert(0, str(PASTA_SCRIPT))
    import manifesto  # noqa: E402
    versao_modelo, versao_registrada, divergencias = manifesto.verificar()
    print(f"🔖 Versão do modelo: {versao_modelo}" + ("" if not divergencias else
          f" — ⚠️ diferente da registrada ({versao_registrada}): " + "; ".join(divergencias[:3])))
    arq_cal = PASTA_SCRIPT / "calibracao.json"
    CONTEXTO["cal"] = json.loads(arq_cal.read_text(encoding="utf-8")) if arq_cal.is_file() else None
    calibracao_usada = "CBIS-DDSM (padrão)"
    if CONTEXTO["cal"] is None:
        print("⚠️  calibracao.json não encontrado: P(maligno) sem calibração e prioridade com limites padrão.")
        calibracao_usada = "sem calibração"
    arq_local = PASTA_SCRIPT / "calibracao_local.json"      # criada no dashboard com casos confirmados do serviço
    if arq_local.is_file() and CONTEXTO["cal"] is not None:
        local = json.loads(arq_local.read_text(encoding="utf-8"))
        # a calibração local é ajustada nas P que o serviço realmente produziu (recorte e recorte + contexto):
        # vale para as duas saídas da patologia
        CONTEXTO["cal"]["patologia"] = CONTEXTO["cal"]["patologia_contexto"] = local["patologia"]
        if "prioridade" in local:
            CONTEXTO["cal"]["prioridade"] = CONTEXTO["cal"]["prioridade_contexto"] = local["prioridade"]
        assinatura = manifesto.sha256(arq_local)[:8]
        versao_modelo = f"{versao_modelo}+cal-local-{assinatura}"
        calibracao_usada = f"local ({local['patologia'].get('n', '?')} casos, {local.get('criada_em', '')})"
        print(f"🎯 Calibração local em uso: {calibracao_usada}")
    CONTEXTO["mapas"] = not args.sem_mapas
    CONTEXTO["pasta_mapas"] = saida.parent / f"{saida.stem}_mapas"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")
    recortes, completas, mama_de, caixas_usuario, dens_informada, fixos = montar_amostras(pasta, planilha)
    info_mama = {}
    for linha in planilha or []:
        for c in ("idade", "fabricante"):
            if linha.get(c):
                info_mama.setdefault(linha["mama"], {})[c] = linha[c]
    tem_detector = all((Path(args.pasta_deteccao) / f).is_file() for f in ("detector_lesao.pth", "detector_lesao.json"))
    tem_densidade = all((Path(args.pasta_densidade) / f).is_file() for f in ("densidade.pth", "densidade.json"))
    detectar = bool(completas) and not args.sem_deteccao and tem_detector
    if completas and args.sem_deteccao:
        print("   -> detecção do modelo desligada (--sem_deteccao).")
    elif completas and not tem_detector:
        print(f"⚠️  detector não encontrado em {args.pasta_deteccao}: o modelo não vai procurar lesões.")
    if not recortes and not detectar and not caixas_usuario:
        raise SystemExit("❌ Nada para classificar: nenhum recorte, nenhuma lesão marcada e nenhuma "
                         "mamografia completa para detecção.")
    print(f"\n🧬 {len(recortes)} amostra(s) com recorte e {len(completas)} mamografia(s) completa(s).")

    print("\n🧠 Carregando modelos...")
    m_recorte = carregar(MultiTaskResNet50(), args.modelo_recorte, device)
    m_multivista = carregar(MultiViewResNet50(), args.modelo_multivista, device)
    modelos_mc = [carregar(ClassificadorMammoCLIP(), c, device) for c in pesos_mc]
    pesos_ctx = [Path(args.pasta_pesos) / f"contexto_dobra_{k}.pth" for k in range(1, N_DOBRAS + 1)]
    if completas and all(c.is_file() for c in pesos_ctx):      # contexto: só nas mamografias completas
        CONTEXTO["modelos_ctx"] = [carregar(ClassificadorMammoCLIP(), c, device) for c in pesos_ctx]
        print("   patologia nas mamografias completas: recorte + contexto (10 redes)")
    detector = DetectorLesao(args.pasta_deteccao, device) if detectar else None
    limiar_det = args.limiar_deteccao if args.limiar_deteccao is not None else (detector.limiar if detector else None)
    if detector:
        print(f"   detector de lesões: score mínimo {limiar_det:.3f}")
    m_dens = ClassificadorDensidade(args.pasta_densidade, device) if completas and tem_densidade else None
    if completas and not tem_densidade:
        print(f"⚠️  classificador de densidade não encontrado em {args.pasta_densidade}: densidade não estimada.")

    divergentes = {}
    if recortes:
        n_antes = len(recortes)
        with torch.no_grad():
            pares, divergentes = parear_por_achado(recortes, mama_de, fixos, m_recorte, device)
        print(f"   pareamento CC + MLO pelo achado previsto: {pares} par(es); {n_antes} recorte(s)/grupo(s) -> "
              f"{len(recortes)} lesão(ões)")

    print("\n🔎 Diagnosticando...")
    resultados, probs_mama = [], {}
    with torch.no_grad():
        for a, v in progresso(sorted(recortes.items()), "Recortes"):
            r = dict(diagnosticar(a, v, m_recorte, m_multivista, modelos_mc, device, args.limiar), _mama=mama_de[a])
            if a in divergentes:                                   # aviso, sem mudar a prioridade
                r["avisos_qualidade"] = "; ".join(x for x in (r.get("avisos_qualidade"), divergentes[a]) if x)
                if r.get("qualidade") == "ok":
                    r["qualidade"] = "atenção"
            resultados.append(r)
        for a, info in progresso(sorted(completas.items()), "Mamografias completas"):
            if m_dens and mama_de[a] not in dens_informada:      # densidade informada dispensa a estimativa
                try:
                    probs_mama.setdefault(mama_de[a], []).append(m_dens.estimar(ler_imagem(info["caminho"])[0]))
                except Exception as e:
                    print(f"   ⚠️  {a}: falha ao estimar a densidade ({e})")
            if detectar or caixas_usuario.get(a):
                for r in diagnosticar_completa(a, info["caminho"], detector, limiar_det, caixas_usuario.get(a),
                                               m_recorte, modelos_mc, device, args.limiar):
                    resultados.append(dict(r, _mama=mama_de[a]))

    # densidade por mama: a informada pelo usuário (prioridade) ou a inferida (média das vistas)
    dens_modelo = {m: int(np.mean(ps, axis=0).argmax()) + 1 for m, ps in probs_mama.items()}
    for r in resultados:
        chaves = (r["amostra"], r.get("_imagem"), r["_mama"])
        d = next((dens_informada[k] for k in chaves if k in dens_informada), None)
        origem = "informada pelo usuário" if d else ""
        if d is None and r["_mama"] in dens_modelo:
            d, origem = dens_modelo[r["_mama"]], "inferida pelo modelo"
        r.update(mama=r["_mama"], densidade=d or "", densidade_origem=origem,
                 aviso_densidade=texto_densidade(d) if d else "", versao_modelo=versao_modelo,
                 calibracao=calibracao_usada)
        info = info_mama.get(r["_mama"], {})                     # idade/fabricante da planilha têm prioridade
        for c in ("idade", "fabricante"):
            if info.get(c):
                r[c] = info[c]
    print(f"   densidade: {len(dens_informada)} informada(s) pelo usuário, {len(dens_modelo)} mama(s) inferida(s) "
          f"pelo modelo")

    with open(saida, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel (Windows) lê os acentos
        w = csv.DictWriter(f, fieldnames=CAMPOS_SAIDA, extrasaction="ignore")
        w.writeheader()
        w.writerows(resultados)

    print(f"\n💾 Resultado salvo em: {saida}")
    if not os.environ.get("MARIA_DASHBOARD"):            # pelo dashboard, quem registra é o próprio dashboard
        try:
            import auditoria  # noqa: E402
            auditoria.registrar(None, "classificação (linha de comando)",
                                {"pasta": str(pasta), "lesoes": len(resultados), "versao_modelo": versao_modelo,
                                 "resultado": str(saida), "sha256_resultado": auditoria.sha256_arquivo(saida)})
        except Exception as e:
            print(f"⚠️  não consegui gravar a auditoria: {e}")
    print("\n" + "=" * 132)
    print(f"{'AMOSTRA':<28}{'PRIORIDADE':<14}{'P(CAL.)':<9}{'QUALIDADE':<11}{'ORIGEM DA LESÃO':<24}{'PATOLOGIA':<13}"
          f"{'ACHADO':<19}{'DENSIDADE'}")
    print("-" * 132)
    for r in resultados:
        dens = f"{r['densidade']} ({r['densidade_origem']})" if r["densidade"] else "—"
        nome = r["amostra"] if len(r["amostra"]) <= 27 else r["amostra"][:26] + "…"
        print(f"{nome:<28}{r['prioridade'][:13]:<14}{r['p_maligno_calibrada']:<9}{r['qualidade']:<11}"
              f"{r['origem_lesao'][:23]:<24}{r['patologia'][:12]:<13}{r['achado'][:18]:<19}{dens}")
    print("=" * 132)


if __name__ == "__main__":
    main()
