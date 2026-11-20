# MARIA — Modelo de Análise Radiomamográfica por Inteligência Artificial

Protótipo de pesquisa que **prioriza a leitura de mamografias** para o radiologista: indica regiões suspeitas,
estima a probabilidade de malignidade (calibrada), o tipo de achado e a densidade mamária, e avisa quando a imagem
não deve ser usada. **Não emite diagnóstico** — a decisão é sempre do radiologista.

> ⚠️ **Protótipo de pesquisa, sem registro na ANVISA. Não usar para decisão clínica.**

## O que faz

| Saída | Como |
|---|---|
| **Prioridade de leitura** (alta / média / baixa) | P(maligno) calibrada; fila ordenada para o radiologista |
| **Patologia** (benigno × maligno) | 5 redes Mammo-CLIP no recorte da lesão + 5 de contexto (janela da mamografia) nas lesões da mamografia completa |
| **Localização** | detector Faster R-CNN; o médico também pode marcar a lesão na tela |
| **Achado** (massa × microcalcificação) | ResNet-50 + multivista (CC + MLO) |
| **Densidade** (BI-RADS A–D) | Mammo-CLIP, 768 px |
| **Explicabilidade e segurança** | mapa de calor (Grad-CAM), incerteza entre as redes, controle de qualidade ("não usar") |

Também inclui: dashboard (Streamlit) com login, auditoria e anonimização (LGPD); revisão, banco de anotações,
calibração local e retreino; nó DICOM para receber do PACS e devolver o resultado no mesmo estudo; documentação de
qualidade e regulatória em [`docs/`](docs/README.md).

## Desempenho medido (pacientes fora do treino)

- **Patologia, recortes** (validação cruzada no CBIS-DDSM, 2.781 lesões): AUC 0,841.
- **Validação externa no CMMD** (mamografia digital, China, 1.872 mamas confirmadas por biópsia; caminho completo
  sem marcação do médico): AUC 0,755 por mama; prioridade alta com 79% de sensibilidade e 53% de especificidade.
- **Densidade**: kappa 0,70 (densa × não densa 88%).

Limitações, subgrupos e detalhes: página "Sobre o modelo" do dashboard e [`docs/qualidade/model_card.md`](docs/qualidade/model_card.md).
**Nenhum dado brasileiro** foi usado — veja o [protocolo proposto](docs/dados/protocolo_dados_brasileiros.md).

## Instalação

1. Baixe este repositório (Code → Download ZIP, ou `git clone`).
2. **Baixe os pesos dos modelos** (~730 MB, grandes demais para o repositório): em
   [Releases](https://github.com/robertoaabfilho/MARIA/releases), baixe o arquivo `MARIA_pesos_*.zip` e
   descompacte-o **dentro da pasta do projeto** (ele cria `modelo/modelo_*/` e `exemplos/1_dicom_filme_cbis/`).
3. Confira os pesos: `python modelo/manifesto.py --verificar` (a versão deve bater com a do `modelo/MANIFESTO.json`).
4. Rode:
   - **Windows:** duplo clique em `config\iniciar_dashboard.bat` (instala o Python e as bibliotecas se faltar);
   - **Linux/macOS:** `./config/iniciar_dashboard.sh`.

Guia completo (em português): [`COMO_USAR.txt`](COMO_USAR.txt). Testes: `python -m pytest testes -q`.

## Licenças

- **Código**: MIT (arquivo [`LICENSE`](LICENSE)).
- **Pesos dos modelos** (Release): os de **patologia e densidade** derivam do codificador do **Mammo-CLIP**
  (Ghosh et al., MICCAI 2024), licenciado sob **CC BY-NC-SA 4.0**; por isso esses pesos são distribuídos sob a
  mesma licença — **uso não comercial**, com citação, e obras derivadas sob a mesma licença. Os pesos de achado e
  detector partem de modelos pré-treinados do torchvision (ImageNet/COCO).
- **Imagens de exemplo**: de bases públicas, com citação obrigatória (abaixo).

## Bases de dados e citações

- **CBIS-DDSM** — Sawyer-Lee R. et al. *Curated Breast Imaging Subset of DDSM* [Dataset]. The Cancer Imaging Archive,
  2016. DOI: 10.7937/K9/TCIA.2016.7O02S9CY (CC BY 3.0).
- **CDD-CESM** — Khaled R. et al. *Categorized Digital Database for Low energy and Subtracted Contrast Enhanced
  Spectral Mammography images* [Dataset]. The Cancer Imaging Archive, 2021. DOI: 10.7937/29kw-ae92 (CC BY 4.0);
  Khaled et al., *Scientific Data* 9, 2022. DOI: 10.1038/s41597-022-01238-0.
- **CMMD** — Cui C. et al. *The Chinese Mammography Database (CMMD)* [Dataset]. The Cancer Imaging Archive, 2021.
  DOI: 10.7937/tcia.eqde-4b16 (CC BY 4.0). Usado só como teste externo.
- **Mammo-CLIP** — Ghosh S. et al. *Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and
  Robustness in Mammography*. MICCAI 2024 (código e pesos: CC BY-NC-SA 4.0).
- The Cancer Imaging Archive — Clark K. et al. *J Digit Imaging* 26(6):1045–1057, 2013.
