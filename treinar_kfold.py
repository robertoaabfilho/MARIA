"""
================================================================================
 VALIDAÇÃO CRUZADA K-FOLD (5 dobras, agrupada por paciente) — modelo recorte
================================================================================
LOO (leave-one-out) de verdade — treinar uma vez por paciente deixado de fora
— é inviável no prazo (seriam ~2.400 treinos completos só pro modelo recorte).
Este script faz a versão prática equivalente: divide os PACIENTES de treino
em 5 grupos (dobras/"folds"), e treina 5 modelos independentes, cada um
usando 4 grupos pra treinar e o 5º pra validar — assim cada paciente é usado
para validação exatamente uma vez, em algum dos 5 treinos.

IMPORTANTE: usa só o conjunto de TREINO (Calc-Training + Mass-Training) pra
fazer as 5 dobras — o conjunto de teste oficial (Calc-Test + Mass-Test)
continua reservado e não participa deste script, preservando o número
oficial do projeto (70,87% / 95,03%) como a avaliação final única.

O que este script mostra: se a acurácia variar MUITO entre as 5 dobras, é
sinal de que o resultado depende de sorte no split (pouco confiável). Se
variar pouco (desvio padrão baixo), é sinal de que o resultado é estável —
reforça a confiança no número oficial reportado.

Como rodar (a partir de ~/MARIA):
    source ~/tcia_env/bin/activate
    cd ~/MARIA
    python3 treinar_kfold.py

Aviso de tempo: treina 5 modelos completos (mesma metodologia do treino
principal — early stopping, ~10-15 épocas cada). Espere de 1 a 2 horas de
GPU, dependendo do hardware.
================================================================================
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

# ---- Torna a pasta modelo/ (ao lado deste arquivo) importável, igual ao maria.py ----
PASTA_SCRIPT = Path(__file__).resolve().parent
PASTA_MODELO = PASTA_SCRIPT / "modelo"
if PASTA_MODELO.is_dir() and str(PASTA_MODELO) not in sys.path:
    sys.path.insert(0, str(PASTA_MODELO))

from treinar_maria_cbis_ddsm import (
    CFG,
    set_seed,
    montar_lista_arquivos,
    criar_transformacoes,
    calcular_pesos_classe,
    DatasetCBISDDSM,
    MultiTaskResNet50,
    rodar_epoca,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epocas", type=int, default=15)
    parser.add_argument("--modo_serie", choices=["completa", "recorte"], default="recorte")
    parser.add_argument("--pasta_saida", default=str(PASTA_SCRIPT / "kfold_resultados"))
    args = parser.parse_args()

    set_seed(CFG.semente)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    pasta_saida = Path(args.pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # ---- 1. Montar a lista de exemplos de TREINO (o teste oficial fica de fora) ----
    print(f"\n📂 Varrendo a base CBIS-DDSM (treino) [modo_serie={args.modo_serie}]...")
    exemplos = montar_lista_arquivos(CFG.raiz_base, CFG.subsets_treino, CFG.pastas_patologia, args.modo_serie)
    print(f"   -> {len(exemplos)} imagens de treino encontradas (serão divididas em {args.n_folds} dobras).")

    grupos_pacientes = [e[3] for e in exemplos]

    # ---- 2. Gerar as dobras (GroupKFold: cada paciente cai numa única dobra) ----
    gkf = GroupKFold(n_splits=args.n_folds)
    indices = list(range(len(exemplos)))

    resultados_por_dobra = []

    for numero_dobra, (idx_treino, idx_val) in enumerate(gkf.split(indices, groups=grupos_pacientes), start=1):
        print(f"\n{'='*70}")
        print(f"===== DOBRA {numero_dobra}/{args.n_folds} =====")
        print(f"{'='*70}")

        exemplos_treino_dobra = [exemplos[i] for i in idx_treino]
        exemplos_val_dobra = [exemplos[i] for i in idx_val]

        pacientes_treino = set(grupos_pacientes[i] for i in idx_treino)
        pacientes_val = set(grupos_pacientes[i] for i in idx_val)
        print(
            f"   {len(exemplos_treino_dobra)} imagens / {len(pacientes_treino)} pacientes treino | "
            f"{len(exemplos_val_dobra)} imagens / {len(pacientes_val)} pacientes validação | "
            f"sobreposição: {len(pacientes_treino & pacientes_val)} (ideal: 0)"
        )

        # ---- Datasets e DataLoaders da dobra ----
        dataset_treino = DatasetCBISDDSM(exemplos_treino_dobra, criar_transformacoes(treino=True))
        dataset_val = DatasetCBISDDSM(exemplos_val_dobra, criar_transformacoes(treino=False))

        loader_treino = DataLoader(
            dataset_treino, batch_size=CFG.batch_size, shuffle=True,
            num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"),
        )
        loader_val = DataLoader(
            dataset_val, batch_size=CFG.batch_size, shuffle=False,
            num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"),
        )

        # ---- Modelo do zero (mesma metodologia do treino principal) ----
        modelo = MultiTaskResNet50(pesos_imagenet=True, dropout=CFG.dropout).to(device)

        for param in modelo.resnet.parameters():
            param.requires_grad = False

        pesos_pat, contagens_pat = calcular_pesos_classe(exemplos_treino_dobra, indice_rotulo=1)
        pesos_ach, contagens_ach = calcular_pesos_classe(exemplos_treino_dobra, indice_rotulo=2)
        criterio_pat = nn.CrossEntropyLoss(weight=pesos_pat.to(device))
        criterio_ach = nn.CrossEntropyLoss(weight=pesos_ach.to(device))

        otimizador = torch.optim.AdamW(
            [p for p in modelo.parameters() if p.requires_grad], lr=CFG.lr, weight_decay=CFG.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=2)

        melhor_acc_patologia = -1.0
        melhor_acc_achado = -1.0
        epocas_sem_melhora = 0
        paciencia = 4
        caminho_modelo_dobra = pasta_saida / f"modelo_dobra_{numero_dobra}.pth"

        for epoca in range(1, args.epocas + 1):
            if CFG.epocas_backbone_congelado > 0 and epoca == CFG.epocas_backbone_congelado + 1:
                print(f"🔓 Descongelando o backbone a partir da época {epoca}.")
                for param in modelo.resnet.parameters():
                    param.requires_grad = True
                otimizador = torch.optim.AdamW(modelo.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=2)

            metricas_treino = rodar_epoca(modelo, loader_treino, otimizador, criterio_pat, criterio_ach, device, treino=True)
            metricas_val = rodar_epoca(modelo, loader_val, otimizador, criterio_pat, criterio_ach, device, treino=False)
            scheduler.step(metricas_val["perda"])

            print(
                f"   Época {epoca:2d} -> treino: pat {metricas_treino['acc_patologia']*100:.2f}% / "
                f"ach {metricas_treino['acc_achado']*100:.2f}% | "
                f"val: pat {metricas_val['acc_patologia']*100:.2f}% / ach {metricas_val['acc_achado']*100:.2f}%"
            )

            if metricas_val["acc_patologia"] > melhor_acc_patologia:
                melhor_acc_patologia = metricas_val["acc_patologia"]
                melhor_acc_achado = metricas_val["acc_achado"]
                epocas_sem_melhora = 0
                torch.save(modelo.state_dict(), caminho_modelo_dobra)
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= paciencia:
                    print(f"   🛑 Early stopping na época {epoca}.")
                    break

        print(f"   ✔ Dobra {numero_dobra} concluída — melhor patologia: {melhor_acc_patologia*100:.2f}% | achado: {melhor_acc_achado*100:.2f}%")
        resultados_por_dobra.append({
            "dobra": numero_dobra,
            "acc_patologia": melhor_acc_patologia,
            "acc_achado": melhor_acc_achado,
            "n_val": len(exemplos_val_dobra),
        })

        # Libera memória da GPU antes da próxima dobra
        del modelo, otimizador, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 3. Resumo final: média ± desvio padrão entre as dobras ----
    import statistics

    accs_pat = [r["acc_patologia"] for r in resultados_por_dobra]
    accs_ach = [r["acc_achado"] for r in resultados_por_dobra]

    media_pat = statistics.mean(accs_pat) * 100
    desvio_pat = statistics.stdev(accs_pat) * 100 if len(accs_pat) > 1 else 0.0
    media_ach = statistics.mean(accs_ach) * 100
    desvio_ach = statistics.stdev(accs_ach) * 100 if len(accs_ach) > 1 else 0.0

    print(f"\n{'='*70}")
    print("📊 RESUMO DA VALIDAÇÃO CRUZADA K-FOLD")
    print(f"{'='*70}")
    for r in resultados_por_dobra:
        print(f"   Dobra {r['dobra']}: patologia {r['acc_patologia']*100:.2f}% | achado {r['acc_achado']*100:.2f}% (n={r['n_val']})")
    print(f"\n   Patologia: {media_pat:.2f}% ± {desvio_pat:.2f}% (média ± desvio padrão entre as {args.n_folds} dobras)")
    print(f"   Achado:    {media_ach:.2f}% ± {desvio_ach:.2f}%")
    print(
        f"\n   Comparar com o número oficial do teste reservado: 70,87% patologia / 95,03% achado. "
        "Se a média do K-fold ficar próxima disso (dentro do desvio padrão), reforça a confiança no resultado."
    )


if __name__ == "__main__":
    main()
