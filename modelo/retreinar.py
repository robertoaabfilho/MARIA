"""
================================================================================
 MARIA — RETREINO com as anotações do usuário (densidade, detector de lesões, achado)
================================================================================
SÓ FUNCIONA NA MÁQUINA QUE TEM OS DADOS COMPLETOS DE TREINO (CBIS-DDSM e CDD-CESM preparados):
o modelo novo é ajustado com os casos do usuário MISTURADOS aos dados originais (sem eles o modelo
"esqueceria" o que aprendeu) e é comparado com o atual num TESTE FIXO (pacientes nunca usados no treino).

Regras de segurança:
  - parte do treino: ajuste fino a partir do modelo ATUAL (não do zero);
  - 20% das mamas anotadas pelo usuário ficam de fora do treino, para conferir o resultado;
  - o modelo novo só vira CANDIDATO se não piorar no teste fixo (e nos casos do usuário guardados);
  - MODO SOMBRA: o candidato fica em modelo/candidato/ e roda em paralelo nas classificações seguintes, SEM mudar
    o resultado; o administrador compara os dois nos casos confirmados e aprova (--promover) ou descarta
    (--descartar). Só então o modelo em uso muda;
  - a versão anterior vai para modelo/historico/<data>/ e pode ser restaurada (--restaurar).
  - PATOLOGIA (benigno/maligno) NÃO é retreinada. ACHADO só usa casos com exame de confirmação
    (biópsia, ressonância ou outro padrão-ouro) informado.

O que cada modelo aproveita do banco de anotações (pasta anotacoes/ do MARIA):
  densidade : mamografias completas de mamas com "densidade_confirmada"
  detector  : mamografias completas em que TODAS as caixas do modelo foram revisadas
              (lesao_real = sim / não) + as lesões marcadas pelo usuário
  achado    : lesões com "achado_confirmado" E "exame_confirmacao" (lesões marcadas como
              falso positivo são ignoradas)

Uso:
    python3 retreinar.py --verificar                       (JSON: dados disponíveis? quantos casos?)
    python3 retreinar.py --modelos densidade detector achado [--rapido]
    python3 retreinar.py --versoes                         (lista versões guardadas)
    python3 retreinar.py --restaurar 20261119_101500       (volta àquela versão)
    python3 retreinar.py --candidatos                      (JSON: candidatos em modo sombra)
    python3 retreinar.py --promover densidade              (candidato aprovado -> modelo em uso)
    python3 retreinar.py --descartar densidade             (apaga o candidato)
Caminhos dos dados: modelo/retreino_config.json (opcional) ou os padrões em CAMINHOS_PADRAO.
================================================================================
"""
import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
from pathlib import Path

for _fluxo in (sys.stdout, sys.stderr):        # Windows: log com acentos/emojis em arquivo ou pipe
    try:
        _fluxo.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
ANOTACOES = RAIZ / "anotacoes"
HISTORICO = AQUI / "historico"
CANDIDATO = AQUI / "candidato"
CAMINHOS_PADRAO = {
    "dados_deteccao": "~/MARIA/dados/preparados/deteccao",          # mamografias reduzidas + caixas (CBIS e CDD)
    "cbis": "~/CBIS_DDSM",                               # CBIS-DDSM organizado + planilhas
    "cdd_xlsx": "~/MARIA/dados/cdd_cesm/Radiology-manual-annotations.xlsx",
    "cache": "~/MARIA/dados/preparados/cache_retreino",                   # caches criados pelo retreino
}
ARQUIVOS_MODELO = {
    "densidade": ["modelo_densidade/densidade.pth", "modelo_densidade/densidade.json"],
    "detector": ["modelo_deteccao/detector_lesao.pth", "modelo_deteccao/detector_lesao.json"],
    "achado": ["modelo_achado/resnet50_multitask_mama_do_zero.pth"],
}
MINIMO = {"densidade": 10, "detector": 10, "achado": 10}   # casos mínimos para liberar o retreino
VERSOES_GUARDADAS = 3                                     # por modelo (as mais antigas são apagadas)


def caminhos():
    cfg = dict(CAMINHOS_PADRAO)
    arq = AQUI / "retreino_config.json"
    if arq.is_file():
        cfg.update(json.loads(arq.read_text(encoding="utf-8")))
    return {k: Path(v).expanduser() for k, v in cfg.items()}


# =============================================================================
# banco de anotações
# =============================================================================

def ler_anotacoes():
    import pandas as pd
    arq = ANOTACOES / "anotacoes.csv"
    if not arq.is_file():
        return pd.DataFrame()
    a = pd.read_csv(arq, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for c in ("id", "amostra", "mama", "origem_lesao", "imagem", "lesao_real", "achado_confirmado",
              "exame_confirmacao", "densidade_confirmada", "x0", "y0", "x1", "y1"):
        if c not in a.columns:                       # banco antigo/importado sem alguma coluna
            a[c] = ""
    for c in ("x0", "y0", "x1", "y1"):
        a[c] = pd.to_numeric(a.get(c, ""), errors="coerce")
    return a


def casos_disponiveis(a):
    """Separa o banco de anotações nos casos que cada modelo pode usar."""
    import pandas as pd
    vazio = {"densidade": pd.DataFrame(), "detector": {}, "achado": pd.DataFrame()}
    if a.empty:
        return vazio
    acr = {"A": 1, "B": 2, "C": 3, "D": 4, "1": 1, "2": 2, "3": 3, "4": 4}
    caixa = ["marcada pelo usuário", "detectada pelo modelo", "nenhuma lesão encontrada"]

    # densidade: por mama (última informada), aplicada a cada mamografia completa dessa mama
    dens = a[a["densidade_confirmada"].str.upper().isin(acr)].groupby("mama")["densidade_confirmada"].last()
    comp = a[a["origem_lesao"].isin(caixa) & (a["imagem"] != "")].drop_duplicates("imagem")
    comp = comp[comp["mama"].isin(dens.index)][["mama", "imagem"]].copy()
    comp["densidade"] = comp["mama"].map(lambda m: acr[dens[m].upper()])

    # detector: imagens com todas as caixas do modelo revisadas
    det = {}
    for img, g in a[a["origem_lesao"].isin(caixa) & (a["imagem"] != "")].groupby("imagem"):
        modelo = g[g["origem_lesao"] == "detectada pelo modelo"]
        usuario = g[g["origem_lesao"] == "marcada pelo usuário"]
        if len(modelo) and not modelo["lesao_real"].isin(["sim", "não"]).all():
            continue                      # revisão incompleta: uma lesão real viraria "fundo"
        if not len(modelo) and not len(usuario):
            continue
        reais = pd.concat([usuario[usuario["lesao_real"] != "não"], modelo[modelo["lesao_real"] == "sim"]])
        det[img] = {"mama": g["mama"].iloc[0],
                    "caixas": reais[["x0", "y0", "x1", "y1"]].dropna().to_numpy(dtype="float32").tolist()}

    # achado: confirmado por exame, e a lesão não é falso positivo
    nomes = {"massa/nódulo": 0, "massa": 0, "nódulo": 0, "microcalcificação": 1, "calcificação": 1}
    ach = a[(a["achado_confirmado"].str.lower().isin(nomes)) & (a["exame_confirmacao"] != "")
            & (a["lesao_real"] != "não") & (a["imagem"] != "")].copy()
    ach["y"] = ach["achado_confirmado"].str.lower().map(nomes)
    return {"densidade": comp, "detector": det, "achado": ach}


def separar_usuario(mamas, frac=0.2, semente=42):
    """Mamas do usuário guardadas para conferência (determinístico)."""
    mamas = sorted(set(mamas))
    random.Random(semente).shuffle(mamas)
    n = int(round(len(mamas) * frac)) if len(mamas) >= 5 else 0
    return set(mamas[:n])


def verificar():
    c = caminhos()
    faltando = []
    for chave, rel in (("dados_deteccao", "anotacoes.csv"), ("dados_deteccao", "anotacoes_cdd.csv"),
                       ("dados_deteccao", "imagens"), ("cbis", "cbis_ddsm_mapeamento_completo.csv"),
                       ("cbis", "cbis_ddsm_organizado"), ("cdd_xlsx", "")):
        p = c[chave] / rel if rel else c[chave]
        if not p.exists():
            faltando.append(str(p))
    for modulo, pacote in (("sklearn", "scikit-learn"), ("openpyxl", "openpyxl")):   # só a máquina de treino usa
        if importlib.util.find_spec(modulo) is None:
            faltando.append(f"biblioteca Python '{pacote}' (instale com: pip install {pacote})")
    try:
        import torch
        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False
    casos = casos_disponiveis(ler_anotacoes())
    n = {"densidade": int(casos["densidade"]["mama"].nunique()) if len(casos["densidade"]) else 0,
         "detector": len(casos["detector"]), "achado": int(len(casos["achado"]))}
    return {"disponivel": not faltando, "faltando": faltando, "gpu": gpu, "casos": n, "minimo": MINIMO}


# =============================================================================
# utilidades de imagem e modelo
# =============================================================================

def maria():
    sys.path.insert(0, str(AQUI))
    import maria as M  # noqa: E402
    return M


def imagem_1536(M, caminho):
    """Mamografia completa -> uint8 com lado maior 1536 (o mesmo preparo do detector e da densidade)."""
    from PIL import Image
    import numpy as np
    arr, _ = M.ler_imagem(caminho)
    H, W = arr.shape
    f = 1536 / max(H, W)
    img = Image.fromarray(M.normalizar_percentis(arr)).resize((round(W * f), round(H * f)), Image.LANCZOS)
    return np.asarray(img), f


def guardar_versao(modelos, carimbo):
    destino = HISTORICO / carimbo
    for m in modelos:
        for rel in ARQUIVOS_MODELO[m]:
            (destino / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(AQUI / rel, destino / rel)
    # mantém só as VERSOES_GUARDADAS mais recentes de cada modelo
    for m in ARQUIVOS_MODELO:
        com = sorted(p for p in HISTORICO.glob("*") if (p / ARQUIVOS_MODELO[m][0]).is_file())
        for velho in com[:-VERSOES_GUARDADAS]:
            for rel in ARQUIVOS_MODELO[m]:
                (velho / rel).unlink(missing_ok=True)
    return destino


def versoes():
    out = []
    for p in sorted(HISTORICO.glob("*"), reverse=True):
        ms = [m for m, rels in ARQUIVOS_MODELO.items() if (p / rels[0]).is_file()]
        if ms:
            out.append({"versao": p.name, "modelos": ms})
    return out


def auditar(acao, detalhes):
    try:
        sys.path.insert(0, str(AQUI))
        import auditoria  # noqa: E402
        import os
        auditoria.registrar(os.environ.get("MARIA_USUARIO"), acao, detalhes)
    except Exception as e:
        print(f"⚠️  não consegui gravar a auditoria: {e}")


def restaurar(versao):
    origem = HISTORICO / versao
    if not origem.is_dir():
        raise SystemExit(f"❌ versão não encontrada: {versao}")
    carimbo = time.strftime("%Y%m%d_%H%M%S") + "_antes_de_restaurar"
    ms = [m for m, rels in ARQUIVOS_MODELO.items() if (origem / rels[0]).is_file()]
    guardar_versao(ms, carimbo)
    for m in ms:
        for rel in ARQUIVOS_MODELO[m]:
            if (origem / rel).is_file():
                shutil.copy2(origem / rel, AQUI / rel)
    print(f"✅ restaurado(s) {', '.join(ms)} da versão {versao} (o estado anterior ficou em {carimbo})")
    auditar("modelo restaurado", {"versao": versao, "modelos": ms, "estado_anterior": carimbo})


def candidatos():
    arq = CANDIDATO / "candidato.json"
    info = json.loads(arq.read_text(encoding="utf-8")) if arq.is_file() else {}
    return {m: i for m, i in info.items() if (CANDIDATO / ARQUIVOS_MODELO[m][0]).is_file()}


def salvar_candidato(m, estado, info_pth, carimbo, antes, depois, n_casos):
    """Modelo retreinado aprovado nos critérios -> modelo/candidato/ (modo sombra). Os .json de configuração vêm
    do modelo em uso (a arquitetura e o limiar não mudam no retreino)."""
    import torch
    for rel in ARQUIVOS_MODELO[m]:
        (CANDIDATO / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel != info_pth:
            shutil.copy2(AQUI / rel, CANDIDATO / rel)
    torch.save(estado, CANDIDATO / info_pth)
    info = candidatos()
    info[m] = {"carimbo": carimbo, "criado_em": time.strftime("%Y-%m-%d %H:%M"), "casos_usuario": n_casos,
               "antes": antes, "depois": depois}
    (CANDIDATO / "candidato.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def promover(m):
    info = candidatos()
    if m not in info:
        raise SystemExit(f"❌ não há candidato de {m}")
    carimbo = time.strftime("%Y%m%d_%H%M%S") + "_antes_de_promover"
    guardar_versao([m], carimbo)
    for rel in ARQUIVOS_MODELO[m]:
        shutil.copy2(CANDIDATO / rel, AQUI / rel)
    descartar(m, silencioso=True)
    try:
        sys.path.insert(0, str(AQUI))
        import manifesto  # noqa: E402
        manifesto.gerar(f"candidato de {m} ({info[m]['carimbo']}) aprovado após o modo sombra")
    except Exception as e:
        print(f"⚠️  não consegui atualizar o manifesto: {e}")
    print(f"✅ {m}: candidato {info[m]['carimbo']} em uso (o anterior ficou em historico/{carimbo})")
    auditar("candidato aprovado (modelo trocado)", {"modelo": m, "candidato": info[m]["carimbo"],
                                                     "estado_anterior": carimbo})


def descartar(m, silencioso=False):
    info = candidatos()
    for rel in ARQUIVOS_MODELO[m]:
        (CANDIDATO / rel).unlink(missing_ok=True)
    c = info.pop(m, {})
    if info:
        (CANDIDATO / "candidato.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        shutil.rmtree(CANDIDATO, ignore_errors=True)
    if not silencioso:
        print(f"🗑️ candidato de {m} descartado")
        auditar("candidato descartado", {"modelo": m, "candidato": c.get("carimbo", "")})


# =============================================================================
# DENSIDADE
# =============================================================================

def retreinar_densidade(casos, c, device, rapido, log):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image
    from sklearn.metrics import cohen_kappa_score
    M = maria()
    cfg = json.loads((AQUI / "modelo_densidade/densidade.json").read_text(encoding="utf-8"))
    tam = cfg["tam"]

    # rótulos de base (CBIS 1-4, CDD A-D)
    rot = {}
    for f in ("mass_case_description_train_set", "calc_case_description_train_set",
              "mass_case_description_test_set", "calc_case_description_test_set"):
        x = pd.read_csv(c["cbis"] / f"{f}.csv")
        x.columns = [k.strip().replace(" ", "_") for k in x.columns]
        for r in x.itertuples():
            if 1 <= int(r.breast_density) <= 4:
                rot[f"{r.patient_id}_{r.left_or_right_breast}_{r.image_view}"] = int(r.breast_density)
    cdd = pd.read_excel(c["cdd_xlsx"], sheet_name="all")
    for nome, v in zip(cdd.loc[cdd.Type == "DM", "Image_name"], cdd.loc[cdd.Type == "DM", "Breast density (ACR)"]):
        if str(v).strip() in "ABCD" and str(v).strip():
            rot[f"CDD_{str(nome).strip()}"] = "ABCD".index(str(v).strip()) + 1
    base = pd.concat([pd.read_csv(c["dados_deteccao"] / "anotacoes.csv"),
                      pd.read_csv(c["dados_deteccao"] / "anotacoes_cdd.csv")]).drop_duplicates("chave")
    base = base[base.chave.isin(rot)]
    if rapido:
        base = pd.concat([base[base.divisao == "treino"].sample(200, random_state=0),
                          base[base.divisao == "teste"].sample(150, random_state=0)])

    def carregar(chave):
        return np.asarray(Image.open(c["dados_deteccao"] / "imagens" / f"{chave}.png").convert("L")
                          .resize((tam, tam), Image.BILINEAR))

    log(f"densidade: carregando {len(base)} imagens de base e {len(casos)} do usuário…")
    x_tr = [(carregar(k), rot[k] - 1) for k in base[base.divisao == "treino"].chave]
    x_te = [(carregar(k), rot[k] - 1) for k in base[base.divisao == "teste"].chave]
    guardadas = separar_usuario(casos["mama"])
    u_tr, u_va = [], []
    for r in casos.itertuples():
        try:
            u8, _ = imagem_1536(M, ANOTACOES / r.imagem)
        except Exception as e:
            log(f"   ⚠️ {r.imagem}: {e}")
            continue
        item = (np.asarray(Image.fromarray(u8).resize((tam, tam), Image.BILINEAR)), int(r.densidade) - 1)
        (u_va if r.mama in guardadas else u_tr).append(item)
    log(f"densidade: treino = {len(x_tr)} de base + {len(u_tr)} do usuário (x5) | teste fixo {len(x_te)} | "
        f"usuário guardado {len(u_va)}")

    def tensor(u8, aug):
        x = torch.from_numpy(u8.copy()).float().div_(255).unsqueeze(0)
        if aug:
            if random.random() < 0.5:
                x = x.flip(2)
            if random.random() < 0.5:
                x = x.flip(1)
            x = TF.rotate(x, random.uniform(-10, 10))
        return TF.normalize(x.expand(3, -1, -1), M.MEDIA, M.DESVIO)

    def novo_modelo():
        m = M.modelo_densidade(cfg.get("arquitetura", "resnet50"))     # mesma rede do modelo em uso
        m.load_state_dict(torch.load(AQUI / "modelo_densidade/densidade.pth", map_location=device))
        return m.to(device)

    @torch.no_grad()
    def prever(m, itens):
        m.eval()
        ps = []
        for i in range(0, len(itens), 32):
            x = torch.stack([tensor(u8, False) for u8, _ in itens[i:i + 32]]).to(device)
            ps.append(((F.softmax(m(x), 1) + F.softmax(m(x.flip(3)), 1)) / 2).cpu())
        return torch.cat(ps).numpy() if ps else np.zeros((0, 4))

    def avaliar(m):
        out = {}
        for nome, itens in (("teste_fixo", x_te), ("usuario_guardado", u_va)):
            if itens:
                y = np.array([t for _, t in itens]); p = prever(m, itens).argmax(1)
                k = float(cohen_kappa_score(y, p, weights="linear")) if len(set(y) | set(p)) > 1 else None
                out[nome] = {"n": len(itens), "kappa": None if k is None or np.isnan(k) else k,
                             "acerto": float((p == y).mean())}
        return out

    atual = novo_modelo()
    antes = avaliar(atual)
    log(f"densidade: modelo atual {json.dumps(antes)}")
    modelo = novo_modelo()
    treino = x_tr + u_tr * 5
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-5, weight_decay=1e-4)
    epocas = 1 if rapido else 4
    for ep in range(1, epocas + 1):
        modelo.train()
        random.shuffle(treino)
        for i in range(0, len(treino) - 15, 16):
            lote = treino[i:i + 16]
            x = torch.stack([tensor(u8, True) for u8, _ in lote]).to(device)
            y = torch.tensor([t for _, t in lote], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                perda = F.cross_entropy(modelo(x).float(), y, label_smoothing=0.1)
            opt.zero_grad(); perda.backward(); opt.step()
        log(f"densidade: época {ep}/{epocas} concluída")
    depois = avaliar(modelo)
    log(f"densidade: modelo novo  {json.dumps(depois)}")
    aceita = (depois["teste_fixo"]["kappa"] or 0) >= (antes["teste_fixo"]["kappa"] or 0) - 0.01 and (
        "usuario_guardado" not in antes or depois["usuario_guardado"]["acerto"] >= antes["usuario_guardado"]["acerto"])
    return modelo.state_dict(), antes, depois, aceita, {"pth": "modelo_densidade/densidade.pth"}


# =============================================================================
# DETECTOR
# =============================================================================

def retreinar_detector(casos, c, device, rapido, log):
    import numpy as np
    import pandas as pd
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops import box_iou
    from PIL import Image
    M = maria()
    cfg = json.loads((AQUI / "modelo_deteccao/detector_lesao.json").read_text(encoding="utf-8"))
    lado, limiar = cfg["lado"], cfg["limiar"]

    cb = pd.read_csv(c["dados_deteccao"] / "anotacoes.csv").assign(fonte="cbis")
    cd = pd.read_csv(c["dados_deteccao"] / "anotacoes_cdd.csv")
    base = pd.concat([cb, cd], ignore_index=True)
    if rapido:
        ch_tr = base[base.divisao == "treino"].chave.drop_duplicates().sample(120, random_state=0)
        ch_te = base[base.divisao == "teste"].chave.drop_duplicates().sample(120, random_state=0)
        base = base[base.chave.isin(set(ch_tr) | set(ch_te))]

    def carregar_base(div):
        out = []
        for k, g in base[base.divisao == div].groupby("chave"):
            img = np.asarray(Image.open(c["dados_deteccao"] / "imagens" / f"{k}.png").convert("L"))
            cx = g[["x0", "y0", "x1", "y1"]].dropna().to_numpy(np.float32).reshape(-1, 4)
            out.append({"img": img, "caixas": cx, "normal": bool((g.patologia == "NORMAL").all()) if "patologia" in g else False})
        return out

    log("detector: carregando imagens de base…")
    b_tr, b_te = carregar_base("treino"), carregar_base("teste")
    guardadas = separar_usuario([v["mama"] for v in casos.values()])
    u_tr, u_va = [], []
    for rel, info in casos.items():
        try:
            u8, f = imagem_1536(M, ANOTACOES / rel)
        except Exception as e:
            log(f"   ⚠️ {rel}: {e}")
            continue
        item = {"img": u8, "caixas": np.array(info["caixas"], np.float32).reshape(-1, 4) * f, "normal": not info["caixas"]}
        (u_va if info["mama"] in guardadas else u_tr).append(item)
    log(f"detector: treino = {len(b_tr)} de base + {len(u_tr)} do usuário (x5) | teste fixo {len(b_te)} | "
        f"usuário guardado {len(u_va)}")

    def novo_modelo():
        m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=lado, max_size=lado,
                                       box_detections_per_img=20, box_score_thresh=0.01)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        m.load_state_dict(torch.load(AQUI / "modelo_deteccao/detector_lesao.pth", map_location=device))
        return m.to(device)

    def tensor(item, aug):
        img = torch.from_numpy(item["img"].copy()).float().div_(255.0)
        cx = torch.from_numpy(item["caixas"].copy()).reshape(-1, 4)
        h, w = img.shape
        if aug:
            if random.random() < 0.5:
                img = img.flip(1); cx[:, [0, 2]] = w - cx[:, [2, 0]]
            if random.random() < 0.5:
                img = img.flip(0); cx[:, [1, 3]] = h - cx[:, [3, 1]]
        return img.unsqueeze(0).expand(3, -1, -1), {"boxes": cx, "labels": torch.ones(len(cx), dtype=torch.int64)}

    def acerto(pred, gt):
        if len(pred) == 0 or len(gt) == 0:
            return torch.zeros(len(pred), len(gt), dtype=torch.bool)
        iou = box_iou(pred, gt) >= 0.2
        cx, cy = (pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2
        dentro = (cx[:, None] >= gt[None, :, 0]) & (cx[:, None] <= gt[None, :, 2]) & \
                 (cy[:, None] >= gt[None, :, 1]) & (cy[:, None] <= gt[None, :, 3])
        return iou | dentro

    @torch.no_grad()
    def avaliar(m, itens):
        m.eval()
        les, achadas, fp_les, fp_nor, n_les_img, n_nor = 0, 0, 0, 0, 0, 0
        for i in range(0, len(itens), 4):
            lote = itens[i:i + 4]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                res = m([tensor(it, False)[0].to(device) for it in lote])
            for it, r in zip(lote, res):
                cx = r["boxes"].float().cpu()[r["scores"].float().cpu() >= limiar]
                gt = torch.from_numpy(it["caixas"]).reshape(-1, 4)
                a = acerto(cx, gt)
                fp = int((~a.any(1)).sum()) if len(cx) else 0
                if len(gt):
                    les += len(gt); achadas += int(a.any(0).sum()); fp_les += fp; n_les_img += 1
                else:
                    fp_nor += fp; n_nor += 1
        return {"lesoes": les, "sensibilidade": achadas / max(1, les), "fp_por_imagem_com_lesao": fp_les / max(1, n_les_img),
                "fp_por_exame_normal": (fp_nor / n_nor) if n_nor else None, "exames_normais": n_nor}

    atual = novo_modelo()
    antes = {"teste_fixo": avaliar(atual, b_te)}
    if u_va:
        antes["usuario_guardado"] = avaliar(atual, u_va)
    log(f"detector: modelo atual {json.dumps(antes)}")
    del atual
    modelo = novo_modelo()
    treino = b_tr + u_tr * 5
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=1e-4)
    epocas = 1 if rapido else 3
    for ep in range(1, epocas + 1):
        modelo.train()
        random.shuffle(treino)
        for i in range(0, len(treino) - 3, 4):
            imgs, alvos = zip(*[tensor(it, True) for it in treino[i:i + 4]])
            imgs = [x.to(device) for x in imgs]
            alvos = [{k: v.to(device) for k, v in a.items()} for a in alvos]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                perda = sum(modelo(imgs, alvos).values())
            opt.zero_grad(); perda.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step()
        log(f"detector: época {ep}/{epocas} concluída")
    depois = {"teste_fixo": avaliar(modelo, b_te)}
    if u_va:
        depois["usuario_guardado"] = avaliar(modelo, u_va)
    log(f"detector: modelo novo  {json.dumps(depois)}")
    a, d = antes["teste_fixo"], depois["teste_fixo"]
    fp_ok = a["fp_por_exame_normal"] is None or d["fp_por_exame_normal"] <= a["fp_por_exame_normal"] * 1.10 + 0.05
    aceita = d["sensibilidade"] >= a["sensibilidade"] - 0.01 and fp_ok and (
        "usuario_guardado" not in antes or depois["usuario_guardado"]["sensibilidade"]
        >= antes["usuario_guardado"]["sensibilidade"] - 0.01)
    return modelo.state_dict(), antes, depois, aceita, {"pth": "modelo_deteccao/detector_lesao.pth"}


# =============================================================================
# ACHADO
# =============================================================================

def cache_recortes_achado(c, log):
    """Recortes de lesão do CBIS-DDSM (224 px, cinza) com o tipo de achado, criados uma vez só."""
    import numpy as np
    import pandas as pd
    import pydicom
    from PIL import Image
    M = maria()
    arq = c["cache"] / "recortes_achado_224.npz"
    if arq.is_file():
        z = np.load(arq, allow_pickle=True)
        return z["img"], z["y"], z["divisao"]
    log("achado: criando o cache de recortes do CBIS-DDSM (só na primeira vez, alguns minutos)…")
    df = pd.read_csv(c["cbis"] / "cbis_ddsm_mapeamento_completo.csv")
    df = df[df.SeriesDescription.isin(["ROI mask images", "cropped images"])]
    imgs, ys, divs = [], [], []
    for pid, g in df.groupby("PatientID"):
        arquivos = [p for r in g.itertuples()
                    for p in (c["cbis"] / "cbis_ddsm_organizado" / r.Subset).glob(f"*/{r.SeriesInstanceUID}/*.dcm")]
        dims = []
        for p in arquivos:
            try:
                d = pydicom.dcmread(str(p), stop_before_pixels=True)
                dims.append((int(d.Rows) * int(d.Columns), p))
            except Exception:
                pass
        if not dims:
            continue
        menor = min(dims)[1]                                   # o recorte é a menor imagem da série
        try:
            arr, eh_dicom = M.ler_imagem(menor)
        except Exception:
            continue
        rgb = M.entrada_achado(arr, eh_dicom).convert("L").resize((224, 224), Image.BILINEAR)
        imgs.append(np.asarray(rgb)); ys.append(1 if pid.startswith("Calc") else 0)
        divs.append("teste" if "Test" in pid else "treino")
    arq.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arq, img=np.stack(imgs), y=np.array(ys), divisao=np.array(divs))
    log(f"achado: cache criado com {len(imgs)} recortes")
    return np.stack(imgs), np.array(ys), np.array(divs)


def retreinar_achado(casos, c, device, rapido, log):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    M = maria()
    imgs, ys, divs = cache_recortes_achado(c, log)
    tr = [(imgs[i], int(ys[i])) for i in np.where(divs == "treino")[0]]
    te = [(imgs[i], int(ys[i])) for i in np.where(divs == "teste")[0]]
    if rapido:
        random.Random(0).shuffle(tr); random.Random(0).shuffle(te)
        tr, te = tr[:300], te[:200]
    guardadas = separar_usuario(casos["mama"])
    u_tr, u_va = [], []
    for r in casos.itertuples():
        try:
            arr, eh_dicom = M.ler_imagem(ANOTACOES / r.imagem)
            if r.origem_lesao != "recorte fornecido" and not np.isnan(r.x0):
                H, W = arr.shape
                m = M.MARGEM_RECORTE
                arr = arr[max(0, int(r.y0) - m):min(H, int(r.y1) + m), max(0, int(r.x0) - m):min(W, int(r.x1) + m)]
            u8 = np.asarray(M.entrada_achado(arr, eh_dicom).convert("L").resize((224, 224), Image.BILINEAR))
        except Exception as e:
            log(f"   ⚠️ {r.amostra}: {e}")
            continue
        (u_va if r.mama in guardadas else u_tr).append((u8, int(r.y)))
    log(f"achado: treino = {len(tr)} de base + {len(u_tr)} do usuário (x5) | teste fixo {len(te)} | "
        f"usuário guardado {len(u_va)}")

    def tensor(u8, aug):
        img = Image.fromarray(u8).convert("RGB")
        if aug:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            img = TF.rotate(img, random.uniform(-15, 15))
        return M.BASE_ACHADO(img)

    def novo_modelo():
        return M.carregar(M.MultiTaskResNet50(), AQUI / "modelo_achado/resnet50_multitask_mama_do_zero.pth", device)

    @torch.no_grad()
    def avaliar(m):
        m.eval()
        out = {}
        for nome, itens in (("teste_fixo", te), ("usuario_guardado", u_va)):
            if not itens:
                continue
            ps = []
            for i in range(0, len(itens), 64):
                x = torch.stack([tensor(u8, False) for u8, _ in itens[i:i + 64]]).to(device)
                ps.append(F.softmax(m(x), 1)[:, 1].cpu())
            p = torch.cat(ps).numpy(); y = np.array([t for _, t in itens])
            out[nome] = {"n": len(itens), "acerto": float(((p >= 0.5) == y).mean()),
                         "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None}
        return out

    antes = avaliar(novo_modelo())
    log(f"achado: modelo atual {json.dumps(antes)}")
    modelo = novo_modelo()
    treino = tr + u_tr * 5
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-5, weight_decay=1e-4)
    epocas = 1 if rapido else 3
    for ep in range(1, epocas + 1):
        modelo.train()
        random.shuffle(treino)
        for i in range(0, len(treino) - 15, 16):
            lote = treino[i:i + 16]
            x = torch.stack([tensor(u8, True) for u8, _ in lote]).to(device)
            y = torch.tensor([t for _, t in lote], device=device)
            perda = F.cross_entropy(modelo(x), y)
            opt.zero_grad(); perda.backward(); opt.step()
        log(f"achado: época {ep}/{epocas} concluída")
    depois = avaliar(modelo)
    log(f"achado: modelo novo  {json.dumps(depois)}")
    a, d = antes["teste_fixo"], depois["teste_fixo"]
    aceita = (d["auc"] or 0) >= (a["auc"] or 0) - 0.005 and (
        "usuario_guardado" not in antes or depois["usuario_guardado"]["acerto"] >= antes["usuario_guardado"]["acerto"])
    return modelo.state_dict(), antes, depois, aceita, {"pth": "modelo_achado/resnet50_multitask_mama_do_zero.pth"}


# =============================================================================
# principal
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Retreino do MARIA com as anotações do usuário.")
    ap.add_argument("--verificar", action="store_true", help="só confere dados e casos (imprime JSON)")
    ap.add_argument("--modelos", nargs="*", choices=["densidade", "detector", "achado"])
    ap.add_argument("--rapido", action="store_true", help="teste do processo: poucas imagens e 1 época (não use de verdade)")
    ap.add_argument("--versoes", action="store_true")
    ap.add_argument("--restaurar", default=None)
    ap.add_argument("--candidatos", action="store_true", help="lista os candidatos em modo sombra (JSON)")
    ap.add_argument("--promover", choices=list(ARQUIVOS_MODELO), help="candidato aprovado passa a ser o modelo em uso")
    ap.add_argument("--descartar", choices=list(ARQUIVOS_MODELO), help="apaga o candidato")
    args = ap.parse_args()

    if args.verificar:
        print(json.dumps(verificar(), ensure_ascii=False))
        return
    if args.versoes:
        print(json.dumps(versoes(), ensure_ascii=False))
        return
    if args.restaurar:
        restaurar(args.restaurar)
        return
    if args.candidatos:
        print(json.dumps(candidatos(), ensure_ascii=False))
        return
    if args.promover:
        promover(args.promover)
        return
    if args.descartar:
        descartar(args.descartar)
        return

    v = verificar()
    if not v["disponivel"]:
        raise SystemExit("❌ Retreino indisponível nesta máquina: faltam os dados completos de treino:\n    "
                         + "\n    ".join(v["faltando"]))
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carimbo = time.strftime("%Y%m%d_%H%M%S")
    pasta_rel = ANOTACOES / "retreinos"
    pasta_rel.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    random.seed(42); torch.manual_seed(42)
    casos = casos_disponiveis(ler_anotacoes())
    relatorio = {"carimbo": carimbo, "dispositivo": str(device), "rapido": args.rapido, "modelos": {}}
    funcoes = {"densidade": retreinar_densidade, "detector": retreinar_detector, "achado": retreinar_achado}
    log(f"retreino iniciado ({device}); casos do usuário: {json.dumps(v['casos'])}")
    for m in args.modelos or []:
        n = v["casos"][m]
        if n < MINIMO[m] and not (args.rapido and n > 0):          # o modo de teste aceita poucos casos
            log(f"{m}: {n} caso(s) do usuário; mínimo {MINIMO[m]} — pulado")
            relatorio["modelos"][m] = {"situacao": "pulado", "motivo": f"{n} caso(s); mínimo {MINIMO[m]}"}
            continue
        try:
            estado, antes, depois, aceita, info = funcoes[m](casos[m], caminhos(), device, args.rapido, log)
        except Exception as e:
            import traceback
            log(f"❌ {m}: erro no retreino: {e}\n{traceback.format_exc()}")
            relatorio["modelos"][m] = {"situacao": "erro", "motivo": str(e)}
            continue
        if aceita and not args.rapido:
            salvar_candidato(m, estado, info["pth"], carimbo, antes, depois, n)
            situacao = "candidato (modo sombra)"
            log(f"✅ {m}: modelo novo NÃO piorou no teste fixo — virou CANDIDATO em modo sombra (modelo/candidato/). "
                "Ele roda em paralelo nas próximas classificações sem mudar o resultado; aprove ou descarte na página "
                "Anotações e retreino.")
        else:
            situacao = "mantido"
            log(f"↩️ {m}: modelo atual mantido ("
                + ("modo rápido de teste" if args.rapido else "o novo piorou no teste fixo ou nos casos do usuário") + ")")
        relatorio["modelos"][m] = {"situacao": situacao, "casos_usuario": n, "antes": antes, "depois": depois}
    (pasta_rel / f"relatorio_{carimbo}.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
    auditar("retreino concluído", {"carimbo": carimbo, "rapido": args.rapido,
                                   "modelos": {m: i.get("situacao") for m, i in relatorio["modelos"].items()}})
    log(f"relatório: {pasta_rel / f'relatorio_{carimbo}.json'}")


if __name__ == "__main__":
    main()
