# Requisitos de software e rastreabilidade

Cada requisito aponta para onde está implementado e como é verificado. Testes automáticos: `testes/`
(rode `python -m pytest testes -q`; o teste de integração exige `MARIA_TESTES_LENTOS=1`).

| ID | Requisito | Implementação | Verificação |
|---|---|---|---|
| REQ_ENT_01 | Ler a planilha do médico em CSV com "," ou ";", UTF-8 ou cp1252, nomes de coluna flexíveis, vírgula decimal | maria.py `ler_planilha` | test_REQ_ENT_01 |
| REQ_ENT_01b | Formato padrão de 11 colunas (paciente, idade, arquivos, lado, vista, caixa x/y início/fim, densidade) com valores normalizados; cabeçalhos do CBIS-DDSM aceitos | maria.py `ler_planilha`; dashboard `padronizar_planilha` | test_REQ_ENT_01b, 01c, 06 |
| REQ_ENT_04 | Juntar CC + MLO (multivista) só quando a mama tem exatamente um recorte de cada vista com o mesmo achado previsto pelo modelo; achados diferentes ficam separados; a coluna opcional "lesao" força o par (o "abnormality id" do CBIS não identifica a lesão entre vistas) | maria.py `montar_amostras`, `parear_por_achado` | test_REQ_ENT_04, 04b |
| REQ_ENT_07 | Obter a vista (CC/MLO) do DICOM pelo ViewPosition ou, se vazio, pelo PatientOrientation | maria.py `vista_dicom`; dashboard `gerar_planilha_da_pasta` | test_REQ_ENT_07 |
| REQ_ENT_02 | Ignorar caixas inválidas (x1 ≤ x0 ou y1 ≤ y0) com aviso | maria.py `ler_planilha` | test_REQ_ENT_02 |
| REQ_ENT_03 | Reconhecer DICOM pelos metadados e PNG/JPG pelo nome ou pela planilha | maria.py `montar_amostras` | regressão dos exemplos |
| REQ_DET_01 | Detectar lesões em mamografia completa e comparar com as marcadas pelo médico | maria.py `DetectorLesao`, `diagnosticar_completa`, `coincide` | test_REQ_DET_01; FROC (desempenho.json) |
| REQ_MAR_01 | Permitir que o radiologista marque a lesão na tela, em pixels da imagem original | dashboard: "Marcar lesões na imagem" | verificação visual registrada (alinhamento da caixa com a lesão) |
| REQ_SAI_01 | Fornecer P(maligno) calibrada, monotônica e entre 0 e 1 | maria.py `calibrar`, calibracao.json | test_REQ_SAI_01; ECE em validação cruzada |
| REQ_PAT_01 | Nas lesões da mamografia completa, combinar a patologia do recorte com a de contexto (janela da mamografia em volta da lesão, igual à do treino) | maria.py `janela_contexto`, `prob_maligno_contexto` | test_REQ_PAT_01, 03; regressão |
| REQ_PAT_02 | Calibração e limite de prioridade próprios para cada entrada (recorte / recorte + contexto); rótulo Maligno = prioridade alta no limiar automático | maria.py `calibrar`, `prioridade_de`, `resultado` | test_REQ_PAT_02 |
| REQ_SAI_02 | Fornecer prioridade de leitura (alta/média/baixa) e não avaliar imagens inadequadas | maria.py `prioridade_de` | test_REQ_SAI_02 |
| REQ_PRI_01 | Prioridade da mama pelo híbrido GMIC + MARIA: escore monotônico nas duas entradas, limites do calibracao.json (escolhidos numa metade do CMMD, conferidos na outra) | maria.py `escore_mama`, `prioridade_hibrida`; calibracao.json `hibrido_gmic` | test_REQ_PRI_01; conferência na metade B do CMMD e no CDD-CESM |
| REQ_PRI_02 | A prioridade da mama vale para todas as linhas da mama; a da lesão fica em `prioridade_lesao`; "não avaliar" é mantido; sem GMIC (ou só recortes), a prioridade é a da lesão | maria.py `aplicar_hibrido` | test_REQ_PRI_02; regressão |
| REQ_PRI_03 | Usar o GMIC só se o código e os 5 pesos conferirem (SHA-256), e pôr a mama na orientação do GMIC qualquer que seja o lado | gmic_maria.py `conferir`, `parede_a_direita` | test_REQ_PRI_03 |
| REQ_PRI_04 | O GMIC roda numa mamografia completa e o mapa de saliência volta no tamanho da imagem; espelhar a imagem não muda o resultado | gmic_maria.py `GMIC.prever` | test_REQ_PRI_04 (lento) |
| REQ_SAI_03 | Indicar quando não usar (modalidade, cor, contraste, recorte pequeno) e quando ter atenção | maria.py `checar_qualidade` | test_REQ_SAI_03 |
| REQ_SAI_04 | Sinalizar resultado incerto (discordância entre as redes ou probabilidade intermediária) | maria.py `resultado` | test_REQ_SAI_04 |
| REQ_SAI_05 | Mostrar onde o modelo olhou (mapa de calor por lesão) | maria.py `mapa_calor` | verificação visual registrada |
| REQ_SUB_01 | Registrar idade e fabricante e analisar por densidade, idade, achado, fabricante e vista | maria.py `metadados_imagem`; dashboard (subgrupos) | regressão; desempenho.json (subgrupos externos) |
| REQ_CAL_01 | Calibrar probabilidade e limiar com casos locais confirmados, com mínimo de casos | dashboard `ajustar_platt`, `limiar_para_sensibilidade`; maria.py (calibracao_local.json) | test_REQ_CAL_01 |
| REQ_DEN_01 | Carregar o modelo de densidade na rede indicada pelo densidade.json (ResNet-50 ou Mammo-CLIP), sem chaves sobrando ou faltando | maria.py `modelo_densidade`, `ClassificadorDensidade` | test_REQ_DEN_01 |
| REQ_RAS_01 | Identificar a versão do modelo (hash) em cada resultado e conferir contra o manifesto | modelo/manifesto.py; maria.py | test_REQ_RAS_01 |
| REQ_RET_01 | Retreinar só com casos válidos (caixas revisadas; achado com exame de confirmação) e só substituir se não piorar | modelo/retreinar.py | test_REQ_RET_01; relatórios de retreino |
| REQ_LGPD_01 | Acesso só com login; senhas com hash; perfis; bloqueio após tentativas; sessão com tempo limite | modelo/acesso.py; dashboard `tela_de_acesso` | test_REQ_LGPD_01; teste do dashboard (login) |
| REQ_LGPD_02 | Registrar quem fez o quê, de forma à prova de adulteração | modelo/auditoria.py | test_REQ_LGPD_02 |
| REQ_LGPD_03 | Pseudonimizar identificadores fora do atendimento (banco, exportação) | modelo/anonimizacao.py `pseudonimizar` | test_REQ_LGPD_03 |
| REQ_LGPD_04 | Anonimizar DICOM copiados para o banco/exportação sem alterar a imagem | modelo/anonimizacao.py `anonimizar_dicom` | test_REQ_LGPD_04 |
| REQ_INT_01 | Receber estudos do PACS (C-STORE), só de AE autorizados, e devolver o resultado no mesmo estudo | integracao/no_dicom.py | test_integracao_pacs (lento) |
| REQ_INT_02 | Gerar fila priorizada para o sistema do serviço e apagar os dados recebidos | integracao/no_dicom.py | test_integracao_pacs (lento) |
