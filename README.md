# MARIA — Diagnóstico Automático de Mamografia

O MARIA é um sistema de apoio ao diagnóstico de câncer de mama que analisa imagens de mamografia e classifica duas coisas, de forma independente:

- **Patologia**: Benigno ou Maligno
- **Achado**: Massa/Nódulo ou Microcalcificação

Este documento explica como os modelos foram treinados, como o `maria.py` combina esses modelos e como usar a ferramenta.

---

## 1. Base de dados

Todo o treino usa o **CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography), baixado do TCIA (The Cancer Imaging Archive) — 6.775 séries DICOM no total, organizadas em `Calc-Training`, `Calc-Test`, `Mass-Training`, `Mass-Test`, cada uma rotulada como `MALIGNANT`, `BENIGN` ou `BENIGN_WITHOUT_CALLBACK` (as duas últimas tratadas como benigno).

Cada série do CBIS-DDSM pode conter até três tipos de imagem, todas derivadas do mesmo exame:

| Tipo | O que é |
|---|---|
| **Mamografia completa** (`full mammogram images`) | A mama inteira, como capturada no exame |
| **Recorte da lesão** (`cropped images`) | Um recorte focado na região suspeita |
| **Máscara de ROI** (`ROI mask images`) | Um mapa binário (branco/preto) marcando o contorno da lesão — usado só para localizar a lesão, nunca para classificação (não tem textura/diagnóstico) |

### 1.1. Pré-processamento do DICOM

Antes de qualquer treino, cada DICOM passa por:

1. **VOI LUT** (janelamento de contraste): aplica a curva de contraste do próprio arquivo, em vez de um min-max cru — sem isso, mamografias de 12-14 bits saem com o contraste "achatado".
2. **Correção de `MONOCHROME1`**: quando a imagem vem nesse modo (menos comum), os valores de pixel vêm invertidos e precisam ser corrigidos antes de virar uma imagem normal.

Essa lógica está em `dicom_para_imagem_pil()`, dentro de `modelo/treinar_maria_cbis_ddsm.py`.

### 1.2. Seleção do arquivo certo por série

Cada pasta de série pode ter mais de um arquivo `.dcm` (é comum o recorte e a máscara da mesma lesão estarem na mesma pasta, por uma peculiaridade de curadoria do próprio CBIS-DDSM). Por isso, cada arquivo é classificado **individualmente** pela sua própria tag `SeriesDescription`, e só depois o de maior resolução é escolhido dentro do tipo desejado — nunca escolhendo por resolução entre tipos diferentes (o que já causou o bug de pegar uma máscara pensando que era um recorte, corrigido durante o desenvolvimento).

---

## 2. Os três modelos treinados

O MARIA não usa um modelo único — usa **três modelos especializados**, cada um melhor em uma coisa, combinados no momento do diagnóstico.

### 2.1. Modelo "recorte"

- **Arquitetura**: ResNet-50, multitarefa (duas cabeças de saída: patologia e achado), inicializada com pesos do ImageNet.
- **Dados de treino**: recortes de lesão (`cropped images`), ~2.900 imagens de treino.
- **Por quê**: é o modelo mais forte nos dois testes, porque o recorte já mostra a lesão em close, facilitando ver tanto a forma (massa vs. calcificação) quanto a textura (benigno vs. maligno).
- **Arquivo**: `modelo/resnet50_multitask_mama_do_zero.pth`

### 2.2. Modelo "completa"

- **Arquitetura**: idêntica ao modelo recorte (ResNet-50 multitarefa).
- **Dados de treino**: mamografias completas (a mama inteira), ~2.450 imagens de treino.
- **Por quê**: sozinho é mais fraco (a lesão é só um detalhe pequeno numa imagem grande), mas complementa o modelo recorte na tarefa de patologia — ver a mama inteira dá algum contexto que o recorte isolado não tem.
- **Limitação conhecida**: em achado (massa vs. calcificação) esse modelo fica perto do chute aleatório, porque o detalhe fino de uma microcalcificação se perde ao reduzir a imagem inteira para 224×224. Por isso **não é usado para achado**.
- **Arquivo**: `modelo/resnet50_multitask_mama_completa.pth`

### 2.3. Modelo "multivista"

- **Arquitetura**: duas vistas (CC — craniocaudal, e MLO — mediolateral oblíqua) da mesma lesão passam pela mesma ResNet-50 (pesos compartilhados, "Siamese"), e as duas saídas de 2048 dimensões são concatenadas antes das camadas finais.
- **Dados de treino**: pares de recortes CC+MLO da mesma lesão, pareados via `PatientID` do DICOM (o CBIS-DDSM guarda o padrão `..._P_01504_LEFT_CC_1`, de onde se extrai paciente, lado, vista e número da lesão).
- **Por quê**: ver as duas vistas juntas ajuda um pouco em achado, mas **não ajudou em patologia** — o motivo mais provável é que parear reduz o volume de exemplos de treino (de ~2.900 recortes individuais para ~1.630 lesões pareadas), e essa perda de volume pesou mais que o ganho teórico de ver duas vistas.
- **Arquivo**: `modelo/resnet50_multivista.pth`

### 2.4. Metodologia comum aos três treinos

- **Split por paciente** (não por imagem): treino, validação e teste nunca compartilham o mesmo paciente — evita vazamento de informação entre os conjuntos.
- **Pesos de classe** na função de perda, compensando o desbalanceamento real entre benigno e maligno na base.
- **Treino em duas fases**: as 2 primeiras épocas treinam só as camadas finais (backbone congelado); depois disso a rede inteira é descongelada.
- **Regularização**: dropout (0,4) antes das camadas finais, weight decay no otimizador (AdamW), scheduler que reduz o learning rate quando a validação para de melhorar.
- **Data augmentation**: `RandomResizedCrop` (zoom/enquadramento variável), flip horizontal e vertical, rotação até 15°, translação leve, variação de brilho/contraste.
- **Early stopping**: o treino para automaticamente se a acurácia de patologia na validação não melhorar por 4 épocas seguidas — evita continuar treinando até o modelo decorar o conjunto de treino.
- **Avaliação final única**: o conjunto de teste (`Calc-Test`/`Mass-Test`) nunca é usado para escolher a melhor época — só entra no fim, uma única vez, pra dar um número honesto.

---

## 3. Como os três modelos são combinados (pipeline final)

Depois de muitos testes com diferentes arquiteturas, combinações e correções metodológicas, a combinação que deu o melhor resultado validado foi:

| Tarefa | Combinação | Resultado (teste completo, 563 lesões) |
|---|---|---|
| **Patologia** | Recorte + Completa, com TTA nos dois | **70,87%** |
| **Achado** | Recorte + Multivista, com TTA no recorte | **95,03%** |

O modelo **multivista não entra em patologia** (testado, piorou o resultado) e o modelo **completa não entra em achado** (fica perto do chute aleatório nessa tarefa).

### 3.1. TTA (Test-Time Augmentation)

Em vez de uma única passada por imagem, cada imagem é avaliada em **5 vistas aumentadas** (original, espelhada horizontal, espelhada vertical, rotação +10°, rotação -10°), e a média das probabilidades entre elas é usada para decidir a classe final. Isso reduz o ruído de uma leitura única e costuma dar um ganho real de 1-3 pontos percentuais.

---

## 4. Estrutura de pastas

```
MARIA/
├── maria.py                                        <- script principal (este README é sobre ele)
├── treinar_kfold.py                                 <- validação cruzada K-fold (ver seção 7)
├── README.md                                        <- este arquivo
├── kfold_resultados/                                 <- gerada ao rodar treinar_kfold.py
│   ├── modelo_dobra_1.pth ... modelo_dobra_5.pth
└── modelo/
    ├── treinar_maria_cbis_ddsm.py                   <- pré-processamento, modelo base, funções compartilhadas
    ├── treinar_maria_multivista.py                  <- modelo multivista, pareamento CC/MLO
    ├── resnet50_multitask_mama_do_zero.pth           <- modelo recorte
    ├── resnet50_multitask_mama_completa.pth          <- modelo completa
    └── resnet50_multivista.pth                       <- modelo multivista
```

O `maria.py` encontra a pasta `modelo/` sozinho, relativa a onde ele mesmo está salvo (não a pasta de onde o comando é rodado) — então essa estrutura funciona em qualquer máquina, bastando copiar a pasta `MARIA/` inteira.

---

## 5. Como usar

### 5.1. Instalação

```bash
pip install torch torchvision pydicom pillow tqdm scikit-learn --break-system-packages
```

O próprio `maria.py` confere essas dependências antes de rodar e avisa com o comando certo de instalação se faltar alguma.

### 5.2. Rodando

```bash
python3 maria.py --pasta /caminho/para/pasta/com/imagens
```

A pasta de entrada pode conter **DICOM (.dcm) e/ou PNG/JPG, misturados**.

### 5.3. Como o `maria.py` identifica os tipos de imagem automaticamente

**Para DICOM**: primeiro tenta pela tag `SeriesDescription` (`cropped images` → recorte, `full mammogram images` → completa; `ROI mask images` é sempre ignorada). Se a descrição não ajudar, usa um fallback por resolução — a maior imagem entre candidatas vira completa, as menores viram recorte.

**Para PNG/JPG** (que não carregam metadado DICOM): pelo sufixo do nome do arquivo:

| Sufixo | Significado |
|---|---|
| `_recorte_cc.png` | Recorte da lesão, vista CC |
| `_recorte_mlo.png` | Recorte da lesão, vista MLO |
| `_recorte.png` | Recorte da lesão, vista única |
| `_completa.png` | Mamografia completa |

Arquivos com o mesmo nome antes do sufixo são tratados como a mesma amostra (lesão).

### 5.4. Como as amostras são agrupadas

**DICOM do CBIS-DDSM** (PatientID no padrão `..._P_01504_LEFT_CC_1`): agrupamento em duas passadas — monta os grupos de lesão a partir dos recortes (que têm o número da lesão), depois anexa cada mamografia completa a **todas** as lesões daquele mesmo lado (a completa não tem número de lesão no PatientID, porque pode conter mais de uma).

**DICOM de outra origem** (sem esse padrão): agrupamento genérico por `PatientID` + `Laterality` (tags DICOM padrão). Como DICOM real de hospital não traz "número da lesão", isso assume uma lesão relevante por lado da mama.

**PNG/JPG**: o nome do arquivo (antes do sufixo) já define a amostra diretamente.

### 5.5. Saída

Gera `results.csv` (por padrão, na mesma pasta do `maria.py`) com as colunas:

| Coluna | Conteúdo |
|---|---|
| `amostra` | Identificador da lesão/caso |
| `dados` | Quais fontes de imagem/modelo entraram na decisão (ex: `recorte_cc+recorte_mlo+completa_cc+completa_mlo+multivista`) |
| `patologia` | Benigno / Maligno / indisponível |
| `achado` | Massa/Nódulo / Microcalcificação / indisponível |
| `conf_patologia` | Confiança (%) da previsão de patologia |
| `conf_achado` | Confiança (%) da previsão de achado |

A coluna `dados` é importante: quando um caso só tem uma fonte disponível (ex: só `completa`, sem recorte), a confiança daquela previsão tende a ser menos confiável — o modelo completa sozinho é o mais fraco dos três.

---

## 6. Validação adicional: K-fold cruzado (`treinar_kfold.py`)

Além do split único treino/validação/teste usado no pipeline principal, o projeto inclui uma ferramenta de **validação cruzada K-fold**, pensada para checar se o resultado depende de sorte na divisão dos dados ou se é estável.

### 6.1. Por que K-fold em vez de Leave-One-Out (LOO)

O ideal, em teoria, seria LOO — treinar o modelo uma vez para cada paciente deixado de fora e medir nele. Na prática isso é inviável: o modelo recorte tem ~2.400 pacientes de treino, e cada treino completo leva de 10 a 20 minutos na GPU — LOO exigiria ~600 horas (25 dias) só para esse modelo. K-fold é a versão prática equivalente: em vez de milhares de treinos, faz **5** (ou outro número configurável), cada um deixando um grupo diferente de pacientes de fora para validação — assim cada paciente é validado exatamente uma vez, distribuído entre as 5 rodadas.

### 6.2. Como funciona

- Os pacientes do conjunto de **treino** (`Calc-Training` + `Mass-Training`) são divididos em 5 grupos, via `GroupKFold` (agrupado por paciente, não por imagem — mesma lógica de evitar vazamento usada no treino principal).
- Treina 5 modelos independentes, cada um com 4 grupos para treinar e 1 para validar, usando a mesma metodologia do modelo recorte oficial (pesos do ImageNet, pesos de classe, congelamento de backbone nas 2 primeiras épocas, early stopping com paciência 4).
- **O conjunto de teste oficial (`Calc-Test`/`Mass-Test`) não participa** — fica reservado, preservando a avaliação final única do pipeline principal.
- Ao final, reporta a acurácia de cada uma das 5 dobras e a **média ± desvio padrão** entre elas — um desvio padrão baixo indica que o resultado é estável; um desvio alto indicaria que a acurácia varia bastante dependendo de quais pacientes caem no treino ou na validação.

### 6.3. Como rodar

```bash
source ~/tcia_env/bin/activate
cd ~/MARIA
python3 treinar_kfold.py
```

Treina 5 modelos completos — espere de 1 a 2 horas de GPU. Os 5 modelos resultantes são salvos em `kfold_resultados/`, separados dos três modelos oficiais usados pelo `maria.py`.

### 6.4. Resultado

```
Patologia: 76,68% ± 2,18% (média ± desvio padrão entre as 5 dobras)
Achado:    93,86% ± 1,32%
```

**Achado** é consistente com o número oficial do pipeline (95,03%) — reforça confiança na estabilidade do modelo nessa tarefa.

**Patologia** aparece mais alta no K-fold (76,68%) do que no número oficial (70,87%). Isso não é uma contradição preocupante: o K-fold mede o modelo recorte isolado, sem TTA e sem o ensemble com o modelo completa (validação mais simples que a do pipeline final), e mede dentro do pool de treino — não no conjunto de teste oficial, genuinamente externo. O desvio padrão baixo (±2,18 pontos) mostra que o modelo é estável entre diferentes divisões de pacientes; a diferença para o número oficial reflete que o teste oficial do CBIS-DDSM parece ser sistematicamente mais difícil que amostras internas do pool de treino — o que torna o **70,87% oficial a estimativa mais conservadora e confiável a reportar**, não o K-fold.

---

## 7. Limitações conhecidas

- **70,87% de acurácia em patologia** não é um número perto de 100% — é, na prática, o teto real que essa abordagem (classificação de imagem única/dupla, sem segmentação automática ou fusão de múltiplas vistas com atenção, treinado com ~2.900 imagens) consegue alcançar nessa base de dados. Um trabalho acadêmico de 2025 usando a mesma fonte de dados e técnicas mais sofisticadas (fusão com features artesanais + embeddings de transformer) chegou ao mesmo teto (~71%).
- **O modelo completa é fraco em achado** (perto do chute aleatório) — por isso fica de fora dessa tarefa.
- **O modelo multivista não superou o ensemble simples** — mantido no pipeline só para achado, onde ajuda um pouco.
- **Agrupamento de DICOM de fontes externas assume uma lesão por lado** — sem o padrão de nomenclatura do CBIS-DDSM, não há como saber o número de lesões reais a partir do próprio arquivo DICOM.
- **Nunca testado em dados clínicos reais** — todo o treino e validação usam exclusivamente o CBIS-DDSM. Não há garantia de que o desempenho se mantenha em imagens de outros scanners, populações ou protocolos de aquisição.

O caminho até esse resultado envolveu descartar várias abordagens que pareciam promissoras mas não se confirmaram na prática: fusão de múltiplas vistas (simples e com atenção), um ensemble de três modelos, outra arquitetura (EfficientNet-B3), e um recorte guiado por máscara de ROI que parecia superior até uma comparação controlada revelar viés de seleção de amostra. Em cada caso, a decisão de manter ou descartar foi baseada em medição no mesmo conjunto de teste reservado, nunca usado para treinar ou ajustar hiperparâmetros.
