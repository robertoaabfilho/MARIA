"""
================================================================================
 TREINAMENTO DO MODELO MARIA COM A BASE COMPLETA CBIS-DDSM (LOCAL)
================================================================================

Este script vive na pasta ~/MARIA (junto com o modelo .pth) e TREINA DO ZERO
(ResNet-50 inicializada com pesos do ImageNet — igual ao script que gerou o
modelo original) usando a base CBIS-DDSM completa e consistente, organizada
localmente em ~/CBIS_DDSM:

    ~/MARIA/
        treinar_maria_cbis_ddsm.py             <- este script
        resnet50_multitask_mama.pth            <- modelo antigo (não usado por padrão)
        resnet50_multitask_mama_do_zero.pth    <- modelo novo (saída)

    ~/CBIS_DDSM/cbis_ddsm_organizado/
        Calc-Training/{MALIGNANT, BENIGN, BENIGN_WITHOUT_CALLBACK}/<SeriesUID>/*.dcm
        Calc-Test/{MALIGNANT, BENIGN, BENIGN_WITHOUT_CALLBACK}/<SeriesUID>/*.dcm
        Mass-Training/{MALIGNANT, BENIGN, BENIGN_WITHOUT_CALLBACK}/<SeriesUID>/*.dcm
        Mass-Test/{MALIGNANT, BENIGN, BENIGN_WITHOUT_CALLBACK}/<SeriesUID>/*.dcm

POR QUE TREINAR DO ZERO EM VEZ DE FINE-TUNING?
O modelo .pth original foi treinado com uma amostra pequena (562 imagens) em
que máscara de ROI, recorte da lesão e mamografia completa estavam misturados
sob o mesmo rótulo — não dá para replicar esse pré-processamento de forma
confiável, e o fine-tuning nele mostrou viés forte (sempre prevendo "maligno").
Por isso o padrão agora é treinar do zero com a base completa (~6.700 séries)
e um único tipo de imagem consistente por rodada (--modo_serie).
Se quiser voltar ao fine-tuning do .pth antigo mesmo assim, use
--continuar_do_existente.

O modelo é multi-tarefa (duas saídas):
  1) Patologia:      Benigno (0)        vs  Maligno (1)
     -> vem do nome da pasta (MALIGNANT / BENIGN / BENIGN_WITHOUT_CALLBACK)
  2) Tipo de achado:  Massa/Nódulo (0)  vs  Microcalcificação (1)
     -> vem do prefixo do subset: "Mass-" = massa, "Calc-" = calcificação

IMPORTANTE — leitura dos DICOMs (v2, corrigido após 1ª rodada de treino):
1) Seleção de série: no CBIS-DDSM, cada "caso" pode ter VÁRIAS séries DICOM
   irmãs (mamografia completa, recorte da lesão, máscara de ROI). Misturar
   essas séries como se fossem todas "a imagem do caso" injeta ruído sério
   nos rótulos (uma máscara de ROI é um blob preto/branco, não uma mamografia).
   Este script agora lê a tag SeriesDescription de cada série e SÓ mantém as
   que indicam mamografia completa ("full mammogram"); séries de "cropped" ou
   "ROI mask" são descartadas. Se a tag não existir/for ambígua, cai de volta
   na heurística antiga (maior resolução) e avisa no log.
2) Janelamento (VOI LUT) e inversão MONOCHROME1: pixel_array bruto de DICOM
   mamográfico tem 12-14 bits de profundidade e depende de RescaleSlope/
   Intercept e VOI LUT para ficar com contraste correto; sem isso, o min-max
   simples "achata" a textura da lesão. Além disso, imagens MONOCHROME1 têm
   as cores invertidas e precisam ser corrigidas. Ambos os ajustes foram
   adicionados na função `dicom_para_imagem_pil`.

Pré-requisitos (instalar antes de rodar):
    pip install torch torchvision pydicom pillow pandas scikit-learn tqdm

Como rodar (a partir da pasta ~/MARIA):
    cd ~/MARIA
    python3 treinar_maria_cbis_ddsm.py
================================================================================
"""

import os
import random
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
except ImportError:
    raise SystemExit(
        "Falta instalar o pydicom. Rode: pip install pydicom"
    )

try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    # Em versões mais novas do pydicom o caminho do import mudou
    from pydicom.pixels import apply_voi_lut


# =============================================================================
# 1. CONFIGURAÇÃO — AJUSTE OS CAMINHOS AQUI SE NECESSÁRIO
# =============================================================================

@dataclass
class Config:
    raiz_base: str = os.path.expanduser("~/CBIS_DDSM/cbis_ddsm_organizado")
    modelo_existente: str = os.path.expanduser("~/MARIA/resnet50_multitask_mama.pth")
    modelo_saida: str = os.path.expanduser("~/MARIA/resnet50_multitask_mama_do_zero.pth")

    subsets_treino = ("Calc-Training", "Mass-Training")
    subsets_teste = ("Calc-Test", "Mass-Test")
    pastas_patologia = {
        "MALIGNANT": 1,
        "BENIGN": 0,
        "BENIGN_WITHOUT_CALLBACK": 0,
    }

    tamanho_imagem: int = 224
    batch_size: int = 16
    epocas: int = 15
    lr: float = 1e-4          # mesmo valor usado no treino original bem-sucedido (0.0001)
    weight_decay: float = 1e-4  # regularização L2, ajuda a reduzir overfitting
    dropout: float = 0.4        # dropout antes das cabeças de classificação
    congelar_backbone: bool = False  # True = só treina as camadas finais (mais rápido, menos preciso)
    num_workers: int = 4
    semente: int = 42

    # Pesos das duas tarefas na soma da loss (ajuste se quiser priorizar uma)
    peso_loss_patologia: float = 1.0
    peso_loss_achado: float = 1.0

    # Se True, roda uma avaliação no teste ANTES de treinar (só faz sentido quando
    # treinar_do_zero=False, ou seja, continuando de pesos já treinados).
    avaliar_antes_de_treinar: bool = True

    # "completa" = mamografia inteira | "recorte" = recorte da lesão (cropped image)
    # "recorte" é o padrão da literatura para classificação benigno/maligno no CBIS-DDSM.
    modo_serie: str = "recorte"

    # Se True, IGNORA modelo_existente e inicia a ResNet-50 com os pesos do ImageNet
    # (igual ao treino original bem-sucedido). Use isso porque o .pth existente foi
    # treinado com uma amostra pequena e com tipos de imagem misturados (máscara,
    # recorte e mamografia completa juntos) — não é uma base confiável para continuar.
    treinar_do_zero: bool = True

    # Se True, para logo após a avaliação baseline (não roda as épocas de treino).
    # Útil para testar rapidamente qual modo_serie bate com o modelo original.
    somente_baseline: bool = False

    # Fração do conjunto de TREINO separada como validação (usada para early
    # stopping e escolha do melhor checkpoint). O conjunto de teste (Calc-Test/
    # Mass-Test) fica de fora do treino inteiro e só é avaliado UMA VEZ no final,
    # para dar um número honesto (usar o teste pra escolher a época é uma forma
    # sutil de vazamento que infla o resultado reportado).
    validacao_fracao: float = 0.15

    # Se True, calcula peso por classe (inverso da frequência) para as duas
    # loss functions — compensa o desbalanceamento real da base (bem mais
    # casos benignos que malignos em Calc-Training, por exemplo).
    usar_pesos_classe: bool = True

    # Treino em duas fases: nas primeiras `epocas_backbone_congelado` épocas,
    # só as camadas finais (fc_patologia/fc_achado) treinam; depois disso o
    # backbone inteiro é descongelado. Ajuda a estabilizar o início do
    # fine-tuning em vez de já sair ajustando a rede inteira de uma vez.
    epocas_backbone_congelado: int = 2


CFG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. VARREDURA DA BASE — monta a lista de (caminho_dicom, patologia, achado)
# =============================================================================

# Modo de seleção de série: "completa" mantém mamografia inteira (descarta recorte/máscara);
# "recorte" mantém o recorte da lesão / cropped image (descarta mamografia completa/máscara).
# Controlado por CFG.modo_serie (ajustável via --modo_serie na linha de comando).
PALAVRAS_MASCARA = ("mask", "roi mask")


def escolher_arquivo_da_serie(pasta_serie: Path, modo: str):
    """Examina CADA ARQUIVO .dcm da pasta individualmente (não a pasta como um
    todo), classificando pela SeriesDescription do próprio arquivo. Isso evita
    o bug de misturar 'ler a descrição de um arquivo' com 'escolher outro
    arquivo por resolução' — no CBIS-DDSM é comum o recorte da lesão e a
    máscara de ROI estarem na MESMA pasta de série, com resoluções parecidas,
    então sem essa correlação por arquivo corre-se o risco real de pegar a
    máscara (um blob preto/branco) pensando que é o recorte da lesão.

    Retorna (caminho, classificacao, patient_id) onde classificacao é
    'bateu_modo' (achou um arquivo com descrição batendo o modo pedido),
    'fallback_maior_area' (nenhum arquivo tinha descrição útil; usou o de
    maior resolução) ou (None, None, None) (a única descrição encontrada foi
    de máscara, ou a pasta está vazia). patient_id vem da tag PatientID do
    DICOM e é usado depois para separar treino/validação POR PACIENTE, não
    por imagem — evita vazamento (mesma pessoa aparecendo nos dois lados)."""
    candidatos_modo = []       # arquivos cuja descrição bate com o modo pedido: (area, arquivo, patient_id)
    candidatos_sem_descricao = []  # arquivos sem SeriesDescription útil (fallback)
    achou_apenas_mascara = False

    for arquivo in pasta_serie.iterdir():
        if not arquivo.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(arquivo), stop_before_pixels=True)
            linhas = int(getattr(ds, "Rows", 0))
            colunas = int(getattr(ds, "Columns", 0))
            area = linhas * colunas
            descricao = str(getattr(ds, "SeriesDescription", "")).lower()
            patient_id = str(getattr(ds, "PatientID", "")) or str(pasta_serie.parent.name)

            if any(p in descricao for p in PALAVRAS_MASCARA):
                achou_apenas_mascara = True
                continue  # nunca serve, para nenhum dos dois modos

            eh_recorte = "crop" in descricao
            eh_completa = "full" in descricao

            bate_modo = (modo == "completa" and eh_completa) or (modo == "recorte" and eh_recorte)
            eh_do_outro_modo = (modo == "completa" and eh_recorte) or (modo == "recorte" and eh_completa)

            if bate_modo:
                candidatos_modo.append((area, arquivo, patient_id))
            elif eh_do_outro_modo:
                continue  # é do modo oposto, não serve aqui
            else:
                candidatos_sem_descricao.append((area, arquivo, patient_id))
        except Exception:
            continue

    if candidatos_modo:
        candidatos_modo.sort(key=lambda x: x[0], reverse=True)
        _, caminho, patient_id = candidatos_modo[0]
        return caminho, "bateu_modo", patient_id

    if candidatos_sem_descricao:
        candidatos_sem_descricao.sort(key=lambda x: x[0], reverse=True)
        _, caminho, patient_id = candidatos_sem_descricao[0]
        return caminho, "fallback_maior_area", patient_id

    return None, None, None


def montar_lista_arquivos(raiz_base: str, subsets, pastas_patologia, modo_serie: str):
    """Percorre raiz_base/<subset>/<PATOLOGIA>/<SeriesUID>/ e monta a lista
    de exemplos (caminho_dicom, rotulo_patologia, rotulo_achado).

    Para cada pasta de série, `escolher_arquivo_da_serie` já garante que o
    arquivo escolhido é, ele mesmo, o que bate com `modo_serie` (não apenas
    "algum arquivo da pasta bateu")."""
    raiz = Path(raiz_base)
    exemplos = []
    faltando_pasta = []
    contagem = {"bateu_modo": 0, "fallback_maior_area": 0, "sem_candidato": 0}

    for subset in subsets:
        pasta_subset = raiz / subset
        if not pasta_subset.exists():
            faltando_pasta.append(str(pasta_subset))
            continue

        # "Calc-Training" -> achado = calcificação (1) | "Mass-Training" -> achado = massa (0)
        rotulo_achado = 1 if subset.startswith("Calc") else 0

        for nome_pasta_pat, rotulo_pat in pastas_patologia.items():
            pasta_pat = pasta_subset / nome_pasta_pat
            if not pasta_pat.exists():
                continue
            for pasta_serie in pasta_pat.iterdir():
                if not pasta_serie.is_dir():
                    continue

                caminho_dicom, classificacao, patient_id = escolher_arquivo_da_serie(pasta_serie, modo_serie)
                if caminho_dicom is None:
                    contagem["sem_candidato"] += 1
                    continue

                contagem[classificacao] += 1
                exemplos.append((str(caminho_dicom), rotulo_pat, rotulo_achado, patient_id))

    if faltando_pasta:
        print("⚠️  Atenção: as seguintes pastas de subset não foram encontradas e foram ignoradas:")
        for p in faltando_pasta:
            print(f"    - {p}")

    rotulo_modo = "mamografia completa" if modo_serie == "completa" else "recorte da lesão"
    print(
        f"   (modo: {rotulo_modo} | arquivo com descrição batendo o modo: {contagem['bateu_modo']} | "
        f"sem descrição útil, usado por fallback (maior resolução): {contagem['fallback_maior_area']} | "
        f"pastas sem nenhum candidato válido (só tinha máscara, ou vazia): {contagem['sem_candidato']})"
    )

    return exemplos


# =============================================================================
# 3. DATASET — carrega e converte DICOM -> tensor normalizado
# =============================================================================

def dicom_para_imagem_pil(caminho: str) -> Image.Image:
    """Lê um DICOM e converte para uma imagem PIL RGB de 8 bits, aplicando:
      1) VOI LUT (janelamento) — usa a curva de contraste definida no próprio
         DICOM em vez de um min-max cru, preservando a textura da lesão.
      2) Correção de MONOCHROME1 — nesse modo o DICOM guarda os valores
         invertidos (maior valor = mais escuro); sem inverter, a imagem sai
         com preto e branco trocados.
    """
    ds = pydicom.dcmread(caminho)
    pixel_array = ds.pixel_array

    # Aplica VOI LUT (janelamento de contraste) quando disponível
    try:
        pixel_array = apply_voi_lut(pixel_array, ds)
    except Exception:
        pass  # se não houver VOI LUT no arquivo, segue com o array original

    pixel_array = pixel_array.astype(np.float32)

    # MONOCHROME1 = valores altos representam PRETO (invertido do usual) -> inverte
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # Normaliza para 0-255 (8 bits) para virar uma imagem PIL comum
    pixel_array -= pixel_array.min()
    maximo = pixel_array.max()
    if maximo > 0:
        pixel_array /= maximo
    pixel_array = (pixel_array * 255).astype(np.uint8)

    return Image.fromarray(pixel_array).convert("RGB")


class DatasetCBISDDSM(Dataset):
    def __init__(self, exemplos, transformacoes):
        self.exemplos = exemplos
        self.transformacoes = transformacoes

    def __len__(self):
        return len(self.exemplos)

    def __getitem__(self, idx):
        caminho, rotulo_pat, rotulo_achado, _patient_id = self.exemplos[idx]
        try:
            imagem = dicom_para_imagem_pil(caminho)
        except Exception as e:
            # Em caso de DICOM corrompido, retorna uma imagem preta para não travar o treino
            print(f"⚠️  Falha ao ler {caminho}: {e}. Usando imagem em branco no lugar.")
            imagem = Image.new("RGB", (CFG.tamanho_imagem, CFG.tamanho_imagem))

        imagem = self.transformacoes(imagem)
        return imagem, rotulo_pat, rotulo_achado


def criar_transformacoes(treino: bool):
    if treino:
        return transforms.Compose([
            # RandomResizedCrop no lugar de Resize fixo: simula a lesão aparecendo
            # um pouco mais perto/longe e deslocada no quadro (zoom + crop variável).
            transforms.RandomResizedCrop(
                CFG.tamanho_imagem, scale=(0.75, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # recorte de lesão não tem "lado certo para cima"
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((CFG.tamanho_imagem, CFG.tamanho_imagem)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# 4. MODELO — mesma arquitetura multi-tarefa já usada no MARIA
# =============================================================================

class MultiTaskResNet50(nn.Module):
    def __init__(self, pesos_imagenet: bool = False, dropout: float = 0.4):
        super(MultiTaskResNet50, self).__init__()
        pesos = models.ResNet50_Weights.DEFAULT if pesos_imagenet else None
        self.resnet = models.resnet50(weights=pesos)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Dropout antes de cada cabeça: reduz overfitting (o modelo não pode
        # depender de nenhum neurônio específico, precisa de padrões mais gerais)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_patologia = nn.Linear(num_ftrs, 2)  # Benigno vs Maligno
        self.fc_achado = nn.Linear(num_ftrs, 2)     # Nódulo/Massa vs Calcificação

    def forward(self, x):
        features = self.dropout(self.resnet(x))
        saida_patologia = self.fc_patologia(features)
        saida_achado = self.fc_achado(features)
        return saida_patologia, saida_achado


# =============================================================================
# 5. LOOPS DE TREINO E VALIDAÇÃO
# =============================================================================

def rodar_epoca(modelo, loader, otimizador, criterio_pat, criterio_ach, device, treino: bool):
    modelo.train() if treino else modelo.eval()

    perda_total = 0.0
    acertos_pat, acertos_ach, total = 0, 0, 0

    contexto = torch.enable_grad() if treino else torch.no_grad()
    with contexto:
        barra = tqdm(loader, desc="Treino" if treino else "Avaliação", leave=False)
        for imagens, rot_pat, rot_ach in barra:
            imagens = imagens.to(device)
            rot_pat = rot_pat.to(device)
            rot_ach = rot_ach.to(device)

            if treino:
                otimizador.zero_grad()

            saida_pat, saida_ach = modelo(imagens)
            perda_pat = criterio_pat(saida_pat, rot_pat)
            perda_ach = criterio_ach(saida_ach, rot_ach)
            perda = CFG.peso_loss_patologia * perda_pat + CFG.peso_loss_achado * perda_ach

            if treino:
                perda.backward()
                otimizador.step()

            perda_total += perda.item() * imagens.size(0)
            acertos_pat += (saida_pat.argmax(1) == rot_pat).sum().item()
            acertos_ach += (saida_ach.argmax(1) == rot_ach).sum().item()
            total += imagens.size(0)

            barra.set_postfix(perda=perda.item())

    return {
        "perda": perda_total / total,
        "acc_patologia": acertos_pat / total,
        "acc_achado": acertos_ach / total,
    }


def calcular_pesos_classe(exemplos, indice_rotulo: int, num_classes: int = 2):
    """Calcula peso por classe (inverso da frequência) a partir da lista de
    exemplos (caminho, rotulo_patologia, rotulo_achado). indice_rotulo=1 para
    patologia, 2 para achado. Classes mais raras recebem peso maior."""
    contagens = [0] * num_classes
    for exemplo in exemplos:
        contagens[exemplo[indice_rotulo]] += 1

    total = sum(contagens)
    pesos = [total / (num_classes * max(c, 1)) for c in contagens]
    return torch.tensor(pesos, dtype=torch.float32), contagens


# =============================================================================
# 6. PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning do modelo MARIA com a base CBIS-DDSM completa")
    parser.add_argument("--raiz_base", default=CFG.raiz_base)
    parser.add_argument("--modelo_existente", default=CFG.modelo_existente)
    parser.add_argument("--modelo_saida", default=CFG.modelo_saida)
    parser.add_argument("--epocas", type=int, default=CFG.epocas)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument(
        "--modo_serie", choices=["completa", "recorte"], default=CFG.modo_serie,
        help="'completa' = mamografia inteira | 'recorte' = imagem recortada da lesão",
    )
    parser.add_argument(
        "--somente_baseline", action="store_true", default=CFG.somente_baseline,
        help="Roda só a avaliação baseline (sem fine-tuning) e para. Útil para testar --modo_serie rápido.",
    )
    parser.add_argument(
        "--treinar_do_zero", dest="treinar_do_zero", action="store_true", default=CFG.treinar_do_zero,
        help="Ignora --modelo_existente e inicia a ResNet-50 com pesos do ImageNet.",
    )
    parser.add_argument(
        "--continuar_do_existente", dest="treinar_do_zero", action="store_false",
        help="Faz fine-tuning a partir de --modelo_existente em vez de treinar do zero.",
    )
    args = parser.parse_args()

    CFG.raiz_base = args.raiz_base
    CFG.modelo_existente = args.modelo_existente
    CFG.modelo_saida = args.modelo_saida
    CFG.epocas = args.epocas
    CFG.batch_size = args.batch_size
    CFG.lr = args.lr
    CFG.modo_serie = args.modo_serie
    CFG.somente_baseline = args.somente_baseline
    CFG.treinar_do_zero = args.treinar_do_zero

    set_seed(CFG.semente)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    # ---- 1. Montar listas de arquivos ----
    print(f"\n📂 Varrendo a base CBIS-DDSM (treino) [modo_serie={CFG.modo_serie}]...")
    exemplos_treino = montar_lista_arquivos(CFG.raiz_base, CFG.subsets_treino, CFG.pastas_patologia, CFG.modo_serie)
    print(f"   -> {len(exemplos_treino)} imagens de treino encontradas.")

    print(f"\n📂 Varrendo a base CBIS-DDSM (teste) [modo_serie={CFG.modo_serie}]...")
    exemplos_teste = montar_lista_arquivos(CFG.raiz_base, CFG.subsets_teste, CFG.pastas_patologia, CFG.modo_serie)
    print(f"   -> {len(exemplos_teste)} imagens de teste encontradas.")

    if len(exemplos_treino) == 0:
        raise SystemExit(
            "❌ Nenhuma imagem de treino encontrada. Confira o caminho em --raiz_base "
            f"(atual: {CFG.raiz_base})."
        )

    # ---- 1.5 Separar validação do treino POR PACIENTE (o teste fica de fora até o final) ----
    # Importante: split por imagem sorteia recortes aleatoriamente e pode colocar
    # imagens do MESMO paciente no treino e na validação — a validação fica
    # artificialmente boa (o modelo "reconhece" o paciente), sem refletir o que
    # acontece com pacientes nunca vistos. Por isso agrupamos por PatientID.
    from sklearn.model_selection import GroupShuffleSplit

    grupos_pacientes = [e[3] for e in exemplos_treino]
    splitter = GroupShuffleSplit(n_splits=1, test_size=CFG.validacao_fracao, random_state=CFG.semente)
    indices_treino, indices_validacao = next(splitter.split(exemplos_treino, groups=grupos_pacientes))
    exemplos_treino_final = [exemplos_treino[i] for i in indices_treino]
    exemplos_validacao = [exemplos_treino[i] for i in indices_validacao]

    pacientes_treino = set(grupos_pacientes[i] for i in indices_treino)
    pacientes_validacao = set(grupos_pacientes[i] for i in indices_validacao)
    sobreposicao = pacientes_treino & pacientes_validacao
    print(
        f"\n🔀 Separado do treino por PACIENTE: {len(exemplos_treino_final)} imagens / "
        f"{len(pacientes_treino)} pacientes para treino final; {len(exemplos_validacao)} imagens / "
        f"{len(pacientes_validacao)} pacientes para validação."
    )
    print(f"   Pacientes em comum entre treino e validação: {len(sobreposicao)} (ideal: 0)")
    print("   O conjunto de teste (Calc-Test/Mass-Test) fica reservado e só é avaliado no final.")

    # ---- 2. Datasets e DataLoaders ----
    dataset_treino = DatasetCBISDDSM(exemplos_treino_final, criar_transformacoes(treino=True))
    dataset_validacao = DatasetCBISDDSM(exemplos_validacao, criar_transformacoes(treino=False))
    dataset_teste = DatasetCBISDDSM(exemplos_teste, criar_transformacoes(treino=False))

    loader_treino = DataLoader(
        dataset_treino, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"),
    )
    loader_validacao = DataLoader(
        dataset_validacao, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"),
    )
    loader_teste = DataLoader(
        dataset_teste, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- 3. Modelo: do zero (ImageNet) ou fine-tuning do .pth existente ----
    if CFG.treinar_do_zero:
        print("\n🧠 Iniciando ResNet-50 com pesos do ImageNet (treino do zero, igual ao treino original).")
        modelo = MultiTaskResNet50(pesos_imagenet=True, dropout=CFG.dropout).to(device)
    else:
        modelo = MultiTaskResNet50(pesos_imagenet=False, dropout=CFG.dropout).to(device)
        if os.path.exists(CFG.modelo_existente):
            print(f"\n🧠 Carregando pesos existentes de: {CFG.modelo_existente}")
            modelo.load_state_dict(torch.load(CFG.modelo_existente, map_location=device))
            print("✔ Pesos carregados. Vamos continuar o treino a partir daqui (fine-tuning).")
        else:
            print(f"⚠️  Modelo existente não encontrado em {CFG.modelo_existente}. Iniciando com pesos aleatórios.")

    # ---- 3.5 Treino em duas fases: backbone congelado no início ----
    if CFG.epocas_backbone_congelado > 0:
        print(
            f"🧊 Backbone congelado pelas primeiras {CFG.epocas_backbone_congelado} época(s) "
            "(só fc_patologia/fc_achado treinam no início)."
        )
        for param in modelo.resnet.parameters():
            param.requires_grad = False

    # ---- 4. Pesos de classe (compensa desbalanceamento benigno/maligno) ----
    if CFG.usar_pesos_classe:
        pesos_pat, contagens_pat = calcular_pesos_classe(exemplos_treino_final, indice_rotulo=1)
        pesos_ach, contagens_ach = calcular_pesos_classe(exemplos_treino_final, indice_rotulo=2)
        print(
            f"⚖️  Distribuição patologia no treino: benigno={contagens_pat[0]} | maligno={contagens_pat[1]} "
            f"-> pesos na loss: {pesos_pat.tolist()}"
        )
        print(
            f"⚖️  Distribuição achado no treino: massa={contagens_ach[0]} | calcificação={contagens_ach[1]} "
            f"-> pesos na loss: {pesos_ach.tolist()}"
        )
        criterio_pat = nn.CrossEntropyLoss(weight=pesos_pat.to(device))
        criterio_ach = nn.CrossEntropyLoss(weight=pesos_ach.to(device))
    else:
        criterio_pat = nn.CrossEntropyLoss()
        criterio_ach = nn.CrossEntropyLoss()

    # ---- 5. Otimizador e scheduler ----
    parametros_treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.AdamW(parametros_treinaveis, lr=CFG.lr, weight_decay=CFG.weight_decay)
    # Reduz o LR pela metade se a perda de validação não melhorar por 2 épocas seguidas —
    # ajuda a estabilizar o treino quando ele começa a "decorar" o treino (overfitting).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=2)

    if CFG.somente_baseline:
        print("🛑 --somente_baseline ativado: parando aqui, sem rodar as épocas de treino.")
        return

    # ---- 6. Loop de treino (usa VALIDAÇÃO para early stopping; teste fica reservado) ----
    melhor_acc_patologia = -1.0
    epocas_sem_melhora = 0
    PACIENCIA_EARLY_STOPPING = 4  # para o treino se patologia não melhorar por N épocas seguidas
    print(f"\n🚀 Iniciando treino por até {CFG.epocas} época(s) (early stopping: paciência {PACIENCIA_EARLY_STOPPING})...\n")

    for epoca in range(1, CFG.epocas + 1):
        # Descongela o backbone assim que passar da fase inicial
        if CFG.epocas_backbone_congelado > 0 and epoca == CFG.epocas_backbone_congelado + 1:
            print(f"🔓 Descongelando o backbone inteiro a partir da época {epoca}.")
            for param in modelo.resnet.parameters():
                param.requires_grad = True
            # Recria o otimizador para incluir os parâmetros recém-descongelados
            otimizador = torch.optim.AdamW(modelo.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=2)

        print(f"===== Época {epoca}/{CFG.epocas} =====")
        metricas_treino = rodar_epoca(modelo, loader_treino, otimizador, criterio_pat, criterio_ach, device, treino=True)
        print(
            f"Treino     -> perda: {metricas_treino['perda']:.4f} | "
            f"acc patologia: {metricas_treino['acc_patologia']*100:.2f}% | "
            f"acc achado: {metricas_treino['acc_achado']*100:.2f}%"
        )

        metricas_val = rodar_epoca(modelo, loader_validacao, otimizador, criterio_pat, criterio_ach, device, treino=False)
        print(
            f"Validação  -> perda: {metricas_val['perda']:.4f} | "
            f"acc patologia: {metricas_val['acc_patologia']*100:.2f}% | "
            f"acc achado: {metricas_val['acc_achado']*100:.2f}%"
        )
        acc_patologia_atual = metricas_val["acc_patologia"]
        scheduler.step(metricas_val["perda"])

        lr_atual = otimizador.param_groups[0]["lr"]
        print(f"   (learning rate atual: {lr_atual:.2e})")

        # Salva o melhor modelo com base na acurácia de PATOLOGIA na VALIDAÇÃO
        # (nunca no teste — o teste só entra no final, uma única vez).
        if acc_patologia_atual > melhor_acc_patologia:
            melhor_acc_patologia = acc_patologia_atual
            epocas_sem_melhora = 0
            os.makedirs(os.path.dirname(CFG.modelo_saida), exist_ok=True)
            torch.save(modelo.state_dict(), CFG.modelo_saida)
            print(f"💾 Novo melhor modelo salvo em: {CFG.modelo_saida} (acc patologia validação: {acc_patologia_atual*100:.2f}%)")
        else:
            epocas_sem_melhora += 1
            print(f"   (sem melhora em patologia há {epocas_sem_melhora}/{PACIENCIA_EARLY_STOPPING} época(s))")
            if epocas_sem_melhora >= PACIENCIA_EARLY_STOPPING:
                print(
                    f"\n🛑 Early stopping: patologia não melhora há {PACIENCIA_EARLY_STOPPING} épocas seguidas. "
                    "Parando para não continuar decorando o treino."
                )
                print()
                break

        print()

    # ---- 7. Avaliação final ÚNICA no teste, com o melhor checkpoint salvo ----
    print("🔒 Carregando o melhor checkpoint (por validação) para a avaliação final no teste...")
    modelo.load_state_dict(torch.load(CFG.modelo_saida, map_location=device))
    if len(exemplos_teste) > 0:
        metricas_teste_final = rodar_epoca(modelo, loader_teste, otimizador, criterio_pat, criterio_ach, device, treino=False)
        print(
            f"\n📊 RESULTADO FINAL NO TESTE (nunca visto durante o treino nem usado para escolher a época):\n"
            f"   acc patologia: {metricas_teste_final['acc_patologia']*100:.2f}% | "
            f"acc achado: {metricas_teste_final['acc_achado']*100:.2f}% | "
            f"perda: {metricas_teste_final['perda']:.4f}"
        )

    print("\n🎉 Treinamento concluído!")
    print(f"   Melhor acurácia de patologia (validação, usada para escolher a época): {melhor_acc_patologia*100:.2f}%")
    if len(exemplos_teste) > 0:
        print(f"   Acurácia de patologia no TESTE (resultado final a reportar): {metricas_teste_final['acc_patologia']*100:.2f}%")
    print(f"   Modelo final salvo em: {CFG.modelo_saida}")


if __name__ == "__main__":
    main()
