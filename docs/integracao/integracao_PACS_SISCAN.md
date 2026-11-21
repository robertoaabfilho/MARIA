# Integração com o fluxo do serviço: PACS/DICOM, RIS e SISCAN

## 1. PACS (implementado: nó DICOM)
`integracao/no_dicom.py` é um nó DICOM (AE "MARIA"):
1. O PACS (ou o mamógrafo) **envia** os estudos para o MARIA (C-STORE). Só AE titles da lista
   `ae_permitidos` são aceitos.
2. Quando o estudo termina de chegar, o MARIA processa e **devolve ao PACS**, no **mesmo estudo** da paciente,
   uma série "MARIA - apoio à priorização (não é diagnóstico)" com imagens DICOM Secondary Capture (mamografia
   com as caixas e a prioridade + imagem-resumo). O radiologista vê o resultado **no próprio visualizador do
   PACS**, ao abrir o exame.
3. As imagens recebidas são apagadas após o processamento.

Configuração: copie `integracao/config_dicom.exemplo.json` para `config_dicom.json`, ajuste AE title, porta,
PACS de destino e AE autorizados; cadastre o nó "MARIA" no PACS (AE, IP, porta). Teste a conexão com
`python integracao/no_dicom.py --teste-eco`. Rodar: `python integracao/no_dicom.py`.
Opções: `limiar` (null = automático), `sem_mapas`, `sem_gmic` (sem o GMIC, a prioridade é só a da lesão; com o GMIC instalado, a série devolvida e a fila usam a prioridade da mama, híbrido GMIC + MARIA).

**Pendências:** teste com o PACS real; **DICOM SR** (relatório estruturado — ex.: modelo de CAD de mamografia)
para que o PACS/RIS leia os achados como dados, e não só como imagem; DICOM com **TLS**; envio automático por
regra de roteamento do PACS (modalidade MG).

## 2. Sistemas de informação do serviço (RIS / lista de trabalho)
O nó grava `integracao/saida/fila_priorizada.csv` (accession, ID da paciente, estudo, maior prioridade,
quantidade de lesões, qualidade, versão do modelo). O RIS pode ler esse arquivo para **ordenar a lista de
trabalho** do radiologista. **Pendência:** integração por **HL7 v2** (ORU) ou **FHIR** (DiagnosticReport /
Observation) conforme o RIS do serviço.

## 3. SISMAMA → SISCAN
O **SISMAMA foi substituído pelo SISCAN** (Sistema de Informação do Câncer, instituído pela Portaria GM/MS nº
3.394/2013), que integra o rastreamento do câncer de mama e do colo do útero. O registro da mamografia no SISCAN
é feito pelo serviço (requisição e laudo, com categoria BI-RADS). ⚠️ Não conhecemos interface pública de
integração automática (API) do SISCAN; a integração costuma ser manual pelo sistema web ou via sistemas
próprios homologados — confirmar com o DATASUS/coordenação estadual.

**Papel do MARIA:** o MARIA **não preenche o laudo** nem a categoria BI-RADS (é decisão do radiologista). Ele
contribui com a **priorização** antes do laudo. Os dados do laudo continuam sendo registrados no SISCAN pelo
fluxo normal do serviço.
