# Model card — MARIA

**Finalidade:** apoio à priorização da leitura de mamografias por radiologista. **Não diagnostica.**
**Situação:** protótipo de pesquisa, sem registro na ANVISA. **Versão do modelo:** ver `modelo/MANIFESTO.json`
(gravada em cada resultado).

## Componentes
| Componente | Arquitetura | Treino | Saída |
|---|---|---|---|
| Patologia | 5 × EfficientNet-B2 (codificador Mammo-CLIP), recorte em tela de 512 px; no treino, cabeças auxiliares de achado e de atributos da lesão (forma, margem, tipo e distribuição da calcificação, sutileza), descartadas no uso | CBIS-DDSM (treino + teste oficial), validação cruzada por paciente | P(maligno) bruta e calibrada (Platt), incerteza |
| Achado | ResNet-50 (ImageNet) + ResNet-50 multivista CC+MLO | CBIS-DDSM (treino) | massa/nódulo × microcalcificação |
| Detector | Faster R-CNN ResNet-50 FPN v2 (COCO) | CBIS-DDSM + CDD-CESM (inclui exames normais) | caixas das lesões |
| Densidade | EfficientNet-B2 (codificador Mammo-CLIP, pré-treino em mamografias), 768 px | CBIS-DDSM + CDD-CESM | BI-RADS 1–4 |

## Desempenho (pacientes fora do treino)
- Patologia: acurácia 76,0%, AUC 0,841 (validação cruzada sem vazamento, 2.781 lesões; por dobra 0,842 ± 0,031);
  AUC 0,90 (densidade 1) a 0,74 (densidade 4). Externo (CDD-CESM, digital): AUC 0,815 (0,85–0,88 em
  massas/calcificações); com limiar 0,5, sensibilidade 85% e especificidade 66%. Subgrupos mais fracos: < 40 anos
  e assimetria/distorção (AUC 0,73).
- Calibração (CBIS-DDSM, validação cruzada): ECE 3,9% → 1,3%. Não transfere para o CDD-CESM sem recalibração.
- Achado: acurácia 93,2%, AUC 0,982 (teste oficial CBIS-DDSM, 704 lesões).
- Detector (limiar 0,50): 59% das lesões em filme, 55% em digital; 0,93 caixa falsa por exame normal
  (1,11 em mama densa, 0,51 não densa).
- Densidade: densa × não densa 88,0%; kappa 0,70 (filme 0,71; digital 0,65); densidade 4 reconhecida em 75% (50% em digital, 38 casos).

## Dados e vieses conhecidos
- CBIS-DDSM: filme digitalizado da década de 1990, EUA; sem idade nem fabricante; só exames com lesão.
- CDD-CESM: mamografia digital (baixa energia), Egito, 2 equipamentos, 326 pacientes.
- **Nenhum dado brasileiro.** Equipamentos e população do SUS não representados.
- Patologia treinada em lesões já suspeitas (≈ 42% malignas): a probabilidade não é de rastreamento.

## Uso não recomendado
Triagem automática sem leitura do radiologista; população de rastreamento sem recalibração; imagens marcadas
"não usar"; mulheres < 40 anos e achados como assimetria/distorção (desempenho menor); decisão de conduta
baseada só no MARIA.
