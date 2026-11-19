# Plano de ciclo de vida do software (base: IEC 62304)

> Documento inicial do projeto. Para um produto regulado, ele precisa ser aprovado e mantido dentro do sistema
> da qualidade (ISO 13485), com revisões registradas.

## 1. Identificação
- **Software:** MARIA — Modelo de Análise Radiomamográfica por Inteligência Artificial.
- **Componentes:** `modelo/maria.py` (motor de inferência), `maria_dashboard.py` (interface), `modelo/retreinar.py`
  (retreino controlado), `integracao/no_dicom.py` (integração PACS), módulos `acesso`, `auditoria`,
  `anonimizacao`, `manifesto`.
- **Pesos dos modelos:** identificados pela **versão do modelo** (hash SHA-256 de pesos + calibração,
  `modelo/MANIFESTO.json`), gravada em cada linha de resultado.

## 2. Classe de segurança do software
Um defeito pode contribuir para atraso no diagnóstico de câncer (dano grave). Existe controle de risco externo
ao software — o **radiologista lê todos os exames** e decide —, que reduz a probabilidade do dano, mas o
projeto adota, de forma conservadora, a **classe C** até haver justificativa aprovada para classe B.
Consequência: requisitos, arquitetura, verificação de unidades e de integração e rastreabilidade completos.

## 3. Processos e como o projeto os atende

| Processo (IEC 62304) | Como é feito hoje | Evidência |
|---|---|---|
| Planejamento | Este plano | docs/qualidade |
| Requisitos de software | Lista com IDs | requisitos_e_rastreabilidade.md |
| Arquitetura | Componentes e fluxos | model_card.md, página "Sobre o modelo" (diagrama), integracao/ |
| Implementação | Código em Python, revisado por análise estática (pyflakes) | git |
| Verificação | Testes automatizados (pytest) ligados aos requisitos + regressão dos exemplos (previsões idênticas às da versão anterior) | `python -m pytest testes` |
| Integração | Teste de ponta a ponta do nó DICOM com PACS simulado | testes/test_integracao_pacs.py |
| Liberação | Gerar manifesto, rodar testes, registrar commit e versão | seção 5 |
| Manutenção / problemas | Registro de problemas e correções no histórico do git (mensagens de commit) | git log |
| Gestão de configuração | Git para código e documentos; hashes para pesos e exemplos; versões exatas das bibliotecas | .gitignore, MANIFESTO.json, config/requirements-validado.txt |
| Gestão de risco | ISO 14971 | gestao_de_risco.md |
| SOUP | Lista com versões e uso | SOUP.md |

## 4. Gestão de configuração
- **Código e documentos:** repositório git (`git log` mostra quem mudou o quê e por quê).
- **Pesos, calibração e exemplos:** grandes demais para o git; ficam fora dele, mas **registrados por hash** em
  `modelo/MANIFESTO.json`. O `maria.py` confere os hashes a cada execução e avisa se algo mudou.
- **Bibliotecas:** versões mínimas em `config/requirements.txt` e **exatas validadas** em
  `config/requirements-validado.txt`.
- **Dados de pacientes, senhas e chaves:** nunca versionados (`.gitignore`).

## 5. Procedimento de liberação de versão
1. `python -m pytest testes -q` sem falhas (inclua o teste de integração: `MARIA_TESTES_LENTOS=1`).
2. Rodar os exemplos e conferir com a versão anterior (`config/rodar_exemplos`).
3. `python modelo/manifesto.py --gerar --descricao "<o que mudou>"`.
4. Atualizar documentos afetados (requisitos, riscos, model card, COMO_USAR).
5. Commit com mensagem descrevendo a mudança e a nova versão do modelo; etiqueta (`git tag vX.Y`).

## 6. Mudanças do modelo em campo (retreino e calibração local)
Feitas apenas por administrador, auditadas, com critérios automáticos de aceitação (teste fixo) e versão
anterior guardada. Para uso assistencial, precisam de plano de controle de mudanças aprovado (ver regulatório).
