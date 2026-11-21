# SOUP — software de terceiros (IEC 62304, 8.1.2)

Gerado em 2026-11-19 a partir do ambiente de desenvolvimento (Python 3.14.10, Linux).

| Biblioteca | Versão validada | Uso no MARIA | Licença | Riscos e controle |
|---|---|---|---|---|
| torch | 2.15.0 | Inferência e treino das redes | Apache-2.0 AND Apache-2.0 WITH LLVM-exce | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| torchvision | 0.29.2 | Arquiteturas ResNet-50 e Faster R-CNN | BSD | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| efficientnet_pytorch | 0.7.1 | Arquitetura EfficientNet-B2 (Mammo-CLIP) | Apache | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| pydicom | 3.0.2 | Leitura/escrita DICOM e anonimização | ver pacote | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| pynetdicom | 3.0.4 | Nó DICOM (C-STORE/C-ECHO) | ver pacote | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| pillow | 13.0.0 | Leitura de PNG/JPG, desenho das caixas | MIT-CMU | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| numpy | 2.5.4 | Cálculo numérico | BSD-3-Clause AND 0BSD AND MIT AND Zlib A | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| pandas | 3.1.0 | Tabelas (dashboard) | BSD-3-Clause | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| streamlit | 1.67.0 | Interface (dashboard) | Apache-2.0 | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| altair | 6.3.0 | Gráficos do dashboard | Copyright (c) 2015-2025, Vega-Altair Dev | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| plotly | 7.1.0 | Imagens com zoom e legenda das caixas (dashboard) | MIT | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| tqdm | 4.70.1 | Barra de progresso | MPL-2.0 AND MIT | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| scikit-learn | 1.9.1 | Retreino e geração de métricas (só máquina de treino) | BSD-3-Clause | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| openpyxl | 3.1.5 | Leitura da planilha do CDD-CESM (só máquina de treino) | MIT | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| scipy | 1.18.1 | Corte da mama no preparo do GMIC (opcional) | BSD-3-Clause | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| opencv-python-headless | 5.1.0.95 | Redimensionamento no preparo do GMIC (opcional) | Apache-2.0 | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |
| pytest | 9.2.0 | Testes automatizados (desenvolvimento) | MIT | Versão exata em config/requirements-validado.txt; mudança de versão exige rodar `python -m pytest testes` e a regressão dos exemplos |

Pesos pré-treinados de terceiros: Mammo-CLIP (codificador de imagem, autores do artigo Ghosh et al., MICCAI 2024), ResNet-50 ImageNet e Faster R-CNN COCO (torchvision). Os pesos ajustados pelo projeto estão registrados por hash em modelo/MANIFESTO.json.

**GMIC (opcional)** — código e 5 pesos publicados pela NYU (github.com/nyukat/GMIC, versão `594029e`; Shen et al., Medical Image Analysis 2021), licença **GNU AGPLv3**. Não é distribuído com o MARIA: é baixado à parte por `config/baixar_gmic.py` para `modelo/modulo_gmic/`, e os pesos são conferidos por SHA-256 (`modelo/gmic_maria.py`, `HASHES`) a cada execução; se não conferirem, o MARIA não usa o GMIC. Usado só na prioridade da mama (híbrido); sem retreino. Riscos: mudança de comportamento com bibliotecas novas (controle: teste lento REQ_PRI_04 e regressão); obrigações da AGPLv3 ao distribuir ou oferecer pela rede uma versão modificada do GMIC.

Anomalias conhecidas relevantes: nenhuma identificada nas versões acima durante os testes do projeto. Antes de liberar uma versão, consultar as notas de versão e os avisos de segurança (CVE) de cada biblioteca.
