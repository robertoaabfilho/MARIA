"""
================================================================================
 TREINO COM FUSÃO DE DUAS VISTAS (CC + MLO) DA MESMA LESÃO
================================================================================
Em vez de classificar uma única imagem por vez, esse script pareia as DUAS
vistas de mamografia da mesma lesão — CC (craniocaudal) e MLO (mediolateral
oblíqua) — e alimenta as duas juntas num modelo com dois ramos (mesma ResNet-50
com pesos compartilhados, tipo "Siamese"), concatenando as duas antes das
camadas finais de classificação. A ideia: a mesma lesão pode ficar mais clara
numa vista do que na outra, então ver as duas juntas tende a ajudar,
principalmente em patologia (benigno vs maligno).

COMO PAREAR AS VISTAS:
O PatientID de cada DICOM do CBIS-DDSM já guarda o padrão completo, por
exemplo "Mass-Training_P_01504_LEFT_CC_1": paciente P_01504, mama esquerda,
vista CC, lesão nº 1. Agrupamos por (paciente, lado, nº da lesão) e juntamos
a imagem CC com a MLO correspondente. Quando só uma vista existe pra uma
lesão, a mesma imagem é usada nos dois ramos (fallback), pra não perder esses
casos — mas a maioria das lesões no CBIS-DDSM tem as duas vistas disponíveis.

Como rodar (a partir de ~/MARIA):
    source ~/tcia_env/bin/activate
    cd ~/MARIA
    python3 treinar_maria_multivista.py
================================================================================
"""

import argparse
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from tqdm import tqdm

from treinar_maria_cbis_ddsm import (
    CFG,
    dicom_para_imagem_pil,
    criar_transformacoes,
    escolher_arquivo_da_serie,
    calcular_pesos_classe,
    set_seed,
)

PADRAO_PATIENT_ID = re.compile(r"P_(\d+)_(LEFT|RIGHT)_(CC|MLO)(?:_(\d+))?")


def parse_patient_id(patient_id_raw: str):
    """Extrai (numero_paciente, lado, vista, numero_lesao) do PatientID do
    CBIS-DDSM, ex: 'Mass-Training_P_01504_LEFT_CC_1' -> ('01504','LEFT','CC','1').
    Retorna None se não bater com o padrão esperado."""
    m = PADRAO_PATIENT_ID.search(patient_id_raw or "")
    if not m:
        return None
    numero_paciente, lado, vista, numero_lesao = m.groups()
    return numero_paciente, lado, vista, (numero_lesao or "0")


def montar_pares_cc_mlo(raiz_base: str, subsets, pastas_patologia):
    """Percorre a base e agrupa os recortes por LESÃO (paciente+lado+nº da
    lesão), juntando a imagem CC com a MLO correspondente. Retorna uma lista
    de (caminho_cc, caminho_mlo, rotulo_patologia, rotulo_achado, numero_paciente)."""
    raiz = Path(raiz_base)
    grupos = {}
    nao_reconhecidos = 0

    for subset in subsets:
        pasta_subset = raiz / subset
        if not pasta_subset.exists():
            continue
        rotulo_achado = 1 if subset.startswith("Calc") else 0

        for nome_pasta_pat, rotulo_pat in pastas_patologia.items():
            pasta_pat = pasta_subset / nome_pasta_pat
            if not pasta_pat.exists():
                continue
            for pasta_serie in pasta_pat.iterdir():
                if not pasta_serie.is_dir():
                    continue

                caminho, _classificacao, patient_id_raw = escolher_arquivo_da_serie(pasta_serie, "recorte")
                if caminho is None:
                    continue

                parsed = parse_patient_id(patient_id_raw)
                if parsed is None:
                    nao_reconhecidos += 1
                    continue

                numero_paciente, lado, vista, numero_lesao = parsed
                chave_lesao = (numero_paciente, lado, numero_lesao)

                if chave_lesao not in grupos:
                    grupos[chave_lesao] = {
                        "numero_paciente": numero_paciente,
                        "rot_pat": rotulo_pat,
                        "rot_ach": rotulo_achado,
                    }
                grupos[chave_lesao][vista] = str(caminho)

    pares = []
    duas_vistas = 0
    so_uma_vista = 0

    for chave_lesao, dados in grupos.items():
        cc = dados.get("CC")
        mlo = dados.get("MLO")

        if cc and mlo:
            duas_vistas += 1
        elif cc or mlo:
            so_uma_vista += 1
        else:
            continue

        # Fallback: se só uma vista existe, duplica ela pros dois ramos do modelo
        cc_final = cc or mlo
        mlo_final = mlo or cc

        pares.append((cc_final, mlo_final, dados["rot_pat"], dados["rot_ach"], dados["numero_paciente"]))

    print(
        f"   Lesões pareadas: {duas_vistas} com as duas vistas (CC+MLO) | "
        f"{so_uma_vista} só com uma vista (duplicada como fallback) | "
        f"{nao_reconhecidos} descartadas (PatientID fora do padrão esperado)"
    )
    return pares


class DatasetMultiVista(Dataset):
    def __init__(self, pares, transformacoes, tamanho: int):
        self.pares = pares
        self.transformacoes = transformacoes
        self.tamanho = tamanho

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        caminho_cc, caminho_mlo, rot_pat, rot_ach, _numero_paciente = self.pares[idx]

        try:
            img_cc = dicom_para_imagem_pil(caminho_cc)
        except Exception:
            img_cc = Image.new("RGB", (self.tamanho, self.tamanho))

        try:
            img_mlo = dicom_para_imagem_pil(caminho_mlo)
        except Exception:
            img_mlo = Image.new("RGB", (self.tamanho, self.tamanho))

        tensor_cc = self.transformacoes(img_cc)
        tensor_mlo = self.transformacoes(img_mlo)
        return tensor_cc, tensor_mlo, rot_pat, rot_ach


class MultiViewResNet50(nn.Module):
    """Duas vistas (CC e MLO) passam pela MESMA ResNet-50 (pesos
    compartilhados — 'Siamese'), e as duas saídas de 2048 dimensões são
    concatenadas (4096) antes das camadas finais de classificação."""

    def __init__(self, pesos_imagenet: bool = False, dropout: float = 0.5):
        super().__init__()
        pesos = models.ResNet50_Weights.DEFAULT if pesos_imagenet else None
        self.backbone = models.resnet50(weights=pesos)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.fc_patologia = nn.Linear(num_ftrs * 2, 2)
        self.fc_achado = nn.Linear(num_ftrs * 2, 2)

    def forward(self, x_cc, x_mlo):
        feat_cc = self.backbone(x_cc)
        feat_mlo = self.backbone(x_mlo)
        feat = torch.cat([feat_cc, feat_mlo], dim=1)
        feat = self.dropout(feat)
        saida_patologia = self.fc_patologia(feat)
        saida_achado = self.fc_achado(feat)
        return saida_patologia, saida_achado


def rodar_epoca(modelo, loader, otimizador, criterio_pat, criterio_ach, device, treino: bool):
    modelo.train() if treino else modelo.eval()

    perda_total = 0.0
    acertos_pat, acertos_ach, total = 0, 0, 0

    contexto = torch.enable_grad() if treino else torch.no_grad()
    with contexto:
        barra = tqdm(loader, desc="Treino" if treino else "Avaliação", leave=False)
        for tensor_cc, tensor_mlo, rot_pat, rot_ach in barra:
            tensor_cc = tensor_cc.to(device)
            tensor_mlo = tensor_mlo.to(device)
            rot_pat = rot_pat.to(device)
            rot_ach = rot_ach.to(device)

            if treino:
                otimizador.zero_grad()

            saida_pat, saida_ach = modelo(tensor_cc, tensor_mlo)
            perda_pat = criterio_pat(saida_pat, rot_pat)
            perda_ach = criterio_ach(saida_ach, rot_ach)
            perda = CFG.peso_loss_patologia * perda_pat + CFG.peso_loss_achado * perda_ach

            if treino:
                perda.backward()
                otimizador.step()

            perda_total += perda.item() * tensor_cc.size(0)
            acertos_pat += (saida_pat.argmax(1) == rot_pat).sum().item()
            acertos_ach += (saida_ach.argmax(1) == rot_ach).sum().item()
            total += tensor_cc.size(0)

            barra.set_postfix(perda=perda.item())

    return {
        "perda": perda_total / total,
        "acc_patologia": acertos_pat / total,
        "acc_achado": acertos_ach / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--modelo_saida", default=str(Path(CFG.modelo_saida).parent / "resnet50_multivista.pth"))
    args = parser.parse_args()

    set_seed(CFG.semente)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    print("\n📂 Pareando lesões (CC + MLO) — TREINO...")
    pares_treino = montar_pares_cc_mlo(CFG.raiz_base, CFG.subsets_treino, CFG.pastas_patologia)
    print(f"   -> {len(pares_treino)} lesões de treino.")

    print("\n📂 Pareando lesões (CC + MLO) — TESTE...")
    pares_teste = montar_pares_cc_mlo(CFG.raiz_base, CFG.subsets_teste, CFG.pastas_patologia)
    print(f"   -> {len(pares_teste)} lesões de teste.")

    if len(pares_treino) == 0:
        raise SystemExit("❌ Nenhuma lesão de treino encontrada.")

    # ---- Separar validação do treino POR PACIENTE ----
    from sklearn.model_selection import GroupShuffleSplit

    grupos_pacientes = [p[4] for p in pares_treino]
    splitter = GroupShuffleSplit(n_splits=1, test_size=CFG.validacao_fracao, random_state=CFG.semente)
    indices_treino, indices_validacao = next(splitter.split(pares_treino, groups=grupos_pacientes))
    pares_treino_final = [pares_treino[i] for i in indices_treino]
    pares_validacao = [pares_treino[i] for i in indices_validacao]

    pacientes_treino = set(grupos_pacientes[i] for i in indices_treino)
    pacientes_validacao = set(grupos_pacientes[i] for i in indices_validacao)
    print(
        f"\n🔀 Separado por PACIENTE: {len(pares_treino_final)} lesões / {len(pacientes_treino)} pacientes treino | "
        f"{len(pares_validacao)} lesões / {len(pacientes_validacao)} pacientes validação | "
        f"sobreposição: {len(pacientes_treino & pacientes_validacao)} (ideal: 0)"
    )

    # ---- Datasets e DataLoaders ----
    dataset_treino = DatasetMultiVista(pares_treino_final, criar_transformacoes(treino=True), CFG.tamanho_imagem)
    dataset_validacao = DatasetMultiVista(pares_validacao, criar_transformacoes(treino=False), CFG.tamanho_imagem)
    dataset_teste = DatasetMultiVista(pares_teste, criar_transformacoes(treino=False), CFG.tamanho_imagem)

    loader_treino = DataLoader(dataset_treino, batch_size=args.batch_size, shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"))
    loader_validacao = DataLoader(dataset_validacao, batch_size=args.batch_size, shuffle=False,
                                   num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"))
    loader_teste = DataLoader(dataset_teste, batch_size=args.batch_size, shuffle=False,
                               num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"))

    # ---- Modelo ----
    print("\n🧠 Iniciando MultiViewResNet50 com pesos do ImageNet (backbone compartilhado entre CC e MLO).")
    modelo = MultiViewResNet50(pesos_imagenet=True, dropout=CFG.dropout).to(device)

    print(f"🧊 Backbone congelado pelas primeiras {CFG.epocas_backbone_congelado} época(s).")
    for param in modelo.backbone.parameters():
        param.requires_grad = False

    # ---- Pesos de classe ----
    exemplos_para_pesos = [(None, p[2], p[3]) for p in pares_treino_final]  # reaproveita calcular_pesos_classe (índices 1 e 2)
    pesos_pat, contagens_pat = calcular_pesos_classe(exemplos_para_pesos, indice_rotulo=1)
    pesos_ach, contagens_ach = calcular_pesos_classe(exemplos_para_pesos, indice_rotulo=2)
    print(f"⚖️  Patologia: benigno={contagens_pat[0]} | maligno={contagens_pat[1]} -> pesos {pesos_pat.tolist()}")
    print(f"⚖️  Achado: massa={contagens_ach[0]} | calcificação={contagens_ach[1]} -> pesos {pesos_ach.tolist()}")
    criterio_pat = nn.CrossEntropyLoss(weight=pesos_pat.to(device))
    criterio_ach = nn.CrossEntropyLoss(weight=pesos_ach.to(device))

    # ---- Otimizador e scheduler ----
    parametros_treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.AdamW(parametros_treinaveis, lr=args.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=2)

    # ---- Loop de treino ----
    melhor_acc_patologia = -1.0
    epocas_sem_melhora = 0
    PACIENCIA = 4
    print(f"\n🚀 Iniciando treino por até {args.epocas} época(s) (early stopping: paciência {PACIENCIA})...\n")

    for epoca in range(1, args.epocas + 1):
        if CFG.epocas_backbone_congelado > 0 and epoca == CFG.epocas_backbone_congelado + 1:
            print(f"🔓 Descongelando o backbone a partir da época {epoca}.")
            for param in modelo.backbone.parameters():
                param.requires_grad = True
            otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=CFG.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=2)

        print(f"===== Época {epoca}/{args.epocas} =====")
        metricas_treino = rodar_epoca(modelo, loader_treino, otimizador, criterio_pat, criterio_ach, device, treino=True)
        print(f"Treino     -> perda: {metricas_treino['perda']:.4f} | acc patologia: {metricas_treino['acc_patologia']*100:.2f}% | acc achado: {metricas_treino['acc_achado']*100:.2f}%")

        metricas_val = rodar_epoca(modelo, loader_validacao, otimizador, criterio_pat, criterio_ach, device, treino=False)
        print(f"Validação  -> perda: {metricas_val['perda']:.4f} | acc patologia: {metricas_val['acc_patologia']*100:.2f}% | acc achado: {metricas_val['acc_achado']*100:.2f}%")

        scheduler.step(metricas_val["perda"])
        print(f"   (learning rate atual: {otimizador.param_groups[0]['lr']:.2e})")

        if metricas_val["acc_patologia"] > melhor_acc_patologia:
            melhor_acc_patologia = metricas_val["acc_patologia"]
            epocas_sem_melhora = 0
            os.makedirs(os.path.dirname(args.modelo_saida), exist_ok=True)
            torch.save(modelo.state_dict(), args.modelo_saida)
            print(f"💾 Novo melhor modelo salvo em: {args.modelo_saida} (acc patologia validação: {melhor_acc_patologia*100:.2f}%)")
        else:
            epocas_sem_melhora += 1
            print(f"   (sem melhora há {epocas_sem_melhora}/{PACIENCIA} época(s))")
            if epocas_sem_melhora >= PACIENCIA:
                print(f"\n🛑 Early stopping na época {epoca}.\n")
                break

        print()

    print("🔒 Avaliação final única no teste...")
    modelo.load_state_dict(torch.load(args.modelo_saida, map_location=device))
    metricas_teste = rodar_epoca(modelo, loader_teste, otimizador, criterio_pat, criterio_ach, device, treino=False)
    print(
        f"\n📊 RESULTADO FINAL NO TESTE:\n"
        f"   acc patologia: {metricas_teste['acc_patologia']*100:.2f}% | acc achado: {metricas_teste['acc_achado']*100:.2f}%"
    )
    print(f"\n🎉 Concluído! Modelo salvo em: {args.modelo_saida}")


if __name__ == "__main__":
    main()
