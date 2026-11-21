# Model card — MARIA

**Finalidade:** apoio à priorização da leitura de mamografias por radiologista. **Não diagnostica.**
**Situação:** protótipo de pesquisa, sem registro na ANVISA. **Versão do modelo:** ver `modelo/MANIFESTO.json`
(gravada em cada resultado).

## Componentes
| Componente | Arquitetura | Treino | Saída |
|---|---|---|---|
| Patologia | 5 × EfficientNet-B2 (codificador Mammo-CLIP) no recorte da lesão (tela 512 px; no treino, cabeças auxiliares de achado e de atributos da lesão, descartadas no uso) + 5 × EfficientNet-B2 de CONTEXTO (janela da mamografia em volta da lesão), em média, nas lesões da mamografia completa | Recorte: CBIS-DDSM (treino + teste oficial) + CDD-CESM; contexto: CBIS-DDSM; validação cruzada por paciente | P(maligno) bruta e calibrada (Platt, uma por entrada), incerteza |
| Achado | ResNet-50 (ImageNet) + ResNet-50 multivista CC+MLO | CBIS-DDSM (treino) | massa/nódulo × microcalcificação |
| Detector | Faster R-CNN ResNet-50 FPN v2 (COCO) | CBIS-DDSM + CDD-CESM (inclui exames normais) | caixas das lesões |
| Densidade | EfficientNet-B2 (codificador Mammo-CLIP, pré-treino em mamografias), 768 px | CBIS-DDSM + CDD-CESM | BI-RADS 1–4 |
| Prioridade da mama (opcional) | GMIC (NYU, 5 redes publicadas; mamografia inteira) + maior P bruta do MARIA na mama: média dos logits padronizados, regressão logística | GMIC: ~200 mil exames de rastreamento da NYU (pelos autores; sem retreino); escala e limites: metade A do CMMD | escore da mama (não é probabilidade), prioridade alta/média/baixa, mapa de saliência |

## Desempenho (pacientes fora do treino)
- Patologia (recorte, validação cruzada sem vazamento no CBIS-DDSM, 2.781 lesões): acurácia 76,1%, AUC 0,841
  (por dobra 0,841 ± 0,027); AUC 0,90 (densidade 1) a 0,74 (densidade 4).
- Externo, CMMD (digital, China, 1.872 mamas com biópsia; caminho completo sem marcação, SEM o GMIC: detector +
  recorte + contexto, maior P por mama): AUC 0,755 (com o GMIC: 0,814 na metade B, abaixo); prioridade alta (P cal. ≥ 0,37) com sensibilidade 79% e especificidade
  53%. Mais fraco em calcificações (0,70) e < 50 anos (0,66–0,70). Escolha do modelo nas mesmas caixas: recorte
  com CDD-CESM +0,015 [+0,002; +0,026] sobre o anterior; média com o contexto +0,014 [+0,005; +0,023]. O contexto
  sozinho não serve (sensibilidade de 36% na prioridade alta). O CDD-CESM deixou de ser teste externo da patologia
  (entrou no treino); lá, antes, a média recorte + contexto também foi melhor ou igual ao recorte.
- Prioridade da mama, híbrido GMIC + MARIA (CMMD, metade B dos pacientes, 932 mamas, nada ajustado nela): AUC 0,814
  (MARIA sozinho 0,758; GMIC 0,789); prioridade alta com sensibilidade 81% e especificidade 62,5% (MARIA sozinho
  78% / 55%). Calcificações: GMIC 0,88 × MARIA 0,70; massas: MARIA 0,76 × GMIC 0,72. Nas 1.872 mamas, GMIC
  sozinho − MARIA = +0,023 [−0,000; +0,045] (incerto); combinação − GMIC = +0,025 [+0,014; +0,036].
  Incluindo as mamas SEM achado dos mesmos exames (protocolo do meta-repositório da NYU; metade B, 1.298 mamas,
  366 sem achado): MARIA 0,761, GMIC 0,842, híbrido 0,842; prioridade alta indevida em mama sem achado: 48% (MARIA)
  × 27% (híbrido). Nesse protocolo o nosso GMIC reproduz o publicado (0,831 em todas as mamas; artigo: 0,831).
  Conferência no CDD-CESM (Egito; tudo congelado; 248 mamas com lesão, 135 malignas; MARIA anterior, sem CDD no
  treino, nas caixas do RADIOLOGISTA): MARIA 0,879, GMIC 0,748, híbrido 0,868 (híbrido − MARIA −0,011 [−0,041;
  +0,017]: empate); prioridade alta do híbrido com o limite do CMMD: sens. 87%, espec. 69%. Com a lesão já bem
  localizada, o GMIC não acrescenta; o ganho do CMMD vem de lesões que o detector não acha (calcificações).
- Calibração (CBIS-DDSM, validação cruzada): ECE 3,8% → 1,4% (recorte) e 5,1% → 1,7% (recorte + contexto). No
  CMMD (70% de câncer) a probabilidade fica abaixo da proporção real; o limite da prioridade alta do recorte +
  contexto (0,37) foi escolhido no CMMD para ~80% de sensibilidade (metade dos pacientes; conferido na outra).
- Achado: acurácia 93,2%, AUC 0,982 (teste oficial CBIS-DDSM, 704 lesões).
- Detector (limiar 0,50): 59% das lesões em filme, 55% em digital; 0,93 caixa falsa por exame normal
  (1,11 em mama densa, 0,51 não densa).
- Densidade: densa × não densa 88,0%; kappa 0,70 (filme 0,71; digital 0,65); densidade 4 reconhecida em 75% (50% em digital, 38 casos).

## Dados e vieses conhecidos
- CBIS-DDSM: filme digitalizado da década de 1990, EUA; sem idade nem fabricante; só exames com lesão.
- CDD-CESM: mamografia digital (baixa energia), Egito, 2 equipamentos, 326 pacientes.
- CMMD: mamografia digital, China, 1.775 pacientes, só casos biopsiados (70% malignos); sem marcação da lesão,
  vista ou fabricante; usado só como teste externo (e para escolher o limite da prioridade alta).
- GMIC: treinado pelos autores em rastreamento da NYU (EUA); entra como publicado, sem retreino; o híbrido foi
  escolhido e medido em casos biopsiados (CMMD), não em rastreamento com exames normais.
- **Nenhum dado brasileiro.** Equipamentos e população do SUS não representados.
- Patologia treinada em lesões já suspeitas (≈ 42% malignas): a probabilidade não é de rastreamento.

## Uso não recomendado
Triagem automática sem leitura do radiologista; população de rastreamento sem recalibração; imagens marcadas
"não usar"; mulheres < 50 anos, calcificações e achados como assimetria/distorção (desempenho menor); decisão de conduta
baseada só no MARIA.
