# Documentação do MARIA — conformidade e qualidade

| Pasta / arquivo | Conteúdo |
|---|---|
| regulatorio/classificacao_SaMD.md | Enquadramento como SaMD (RDC 657/2022), classe de risco (RDC 751/2022), próximos passos |
| qualidade/plano_ciclo_de_vida_software.md | IEC 62304: classe de segurança, processos, configuração, verificação, liberação |
| qualidade/requisitos_e_rastreabilidade.md | Requisitos com ID ↔ implementação ↔ teste |
| qualidade/gestao_de_risco.md | ISO 14971: perigos, controles implementados, risco residual |
| qualidade/SOUP.md | Bibliotecas de terceiros e versões validadas |
| qualidade/model_card.md | Descrição do modelo, dados, desempenho, limitações |
| seguranca/seguranca_informacao_e_LGPD.md | Controles de segurança, LGPD (base legal, anonimização, acesso, auditoria, retenção) |
| dados/protocolo_dados_brasileiros.md | Protocolo de coleta de dados reais brasileiros multicêntricos |
| integracao/integracao_PACS_SISCAN.md | PACS/DICOM, RIS, SISCAN (antigo SISMAMA) |

## Matriz de conformidade (situação em 19/11/2026)

Legenda: ✅ implementado e testado · 🟡 parcial / depende de ação externa · ❌ não atende

| # | Requisito | Situação | O que existe | O que falta |
|---|---|---|---|---|
| 1 | Encontrar lesões ou o radiologista marcá-las | ✅ | Detector Faster R-CNN; marcação na tela (dashboard) e na planilha; comparação médico × modelo | Segmentação (contorno) da lesão, se exigida |
| 2 | Dados do mundo real (SUS, vários fabricantes, rótulo confiável) | 🟡 | Validação externa no CMMD (digital, mamas com biópsia; caminho completo com a prioridade da mama pelo híbrido GMIC + MARIA: AUC 0,814 na metade dos pacientes fora de qualquer escolha, prioridade alta com 81% de sensibilidade e 62,5% de especificidade; sem o GMIC, 0,758) e conferência no CDD-CESM; calibração local da probabilidade e do limiar com casos confirmados; desempenho local por fabricante; protocolo de coleta | **Os dados brasileiros em si** (depende de projeto aprovado no CEP e parcerias); ajuste/validação com eles |
| 3 | Priorizar (não diagnosticar), calibrado, explicável, indicar quando não usar | ✅ | Prioridade de leitura e fila (prioridade da mama pelo híbrido GMIC + MARIA); P calibrada (ECE 3,8% → 1,4%); calibração local da lesão e da mama com IC 95%; incerteza entre as redes; mapa de calor Grad-CAM; controle de qualidade "não usar/atenção" | Calibração definitiva com dados locais; avaliação de qualidade de imagem mais completa (posicionamento, artefatos) |
| 4 | Subgrupos: densidade, idade, lesão, fabricante | 🟡 | Saída com idade e fabricante; análise por subgrupo no dashboard; validação externa por idade, equipamento, densidade e achado; desempenho local por subgrupo | Bases com mais equipamentos e faixas etárias (o CBIS-DDSM não tem idade nem fabricante) |
| 5 | Diretrizes de SaMD (RDC 657/2022, RDC 751/2022) | 🟡 | Análise preliminar: SaMD, hipótese classe III (registro) | Parecer regulatório; registro na ANVISA |
| 6 | ISO 13485, IEC 62304, ISO 14971, segurança da informação | 🟡 | Git + manifesto de hashes; versão do modelo em cada resultado; 38 testes automatizados; controle de mudança do modelo (modo sombra com aprovação do administrador); monitoramento do desempenho local com alerta de queda; requisitos e rastreabilidade; gestão de risco; SOUP; plano de ciclo de vida | Sistema da qualidade implantado e auditado; certificação; revisão independente dos documentos |
| 7 | LGPD | ✅/🟡 | Login com perfis; auditoria com hash encadeado; pseudonimização e anonimização DICOM no banco e nas exportações; minimização no nó DICOM; retenção; documento de base legal | Aprovação do encarregado (DPO) e RIPD formal da instituição; criptografia do disco (infraestrutura) |
| 8 | Integração PACS/DICOM, sistemas do serviço, SISMAMA | 🟡 | Nó DICOM: recebe do PACS e devolve imagens de resultado no mesmo estudo; fila priorizada em CSV para o RIS | Teste com o PACS real do serviço; DICOM SR estruturado; HL7/FHIR com o RIS; SISCAN (sem interface pública conhecida) |
