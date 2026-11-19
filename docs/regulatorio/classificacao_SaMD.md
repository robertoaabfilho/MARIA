# Enquadramento regulatório do MARIA como Software como Dispositivo Médico (SaMD)

> **Análise preliminar, feita pela equipe de desenvolvimento — não substitui parecer de especialista em
> assuntos regulatórios nem consulta à ANVISA.** Confira sempre o texto vigente das normas no site da ANVISA
> (as resoluções citadas podem ter sido alteradas). Pontos de menor certeza estão marcados com ⚠️.

## 1. O MARIA é dispositivo médico?

**Sim, se for usado em pacientes.** A RDC nº 657/2022 (ANVISA) trata da regularização de *software como
dispositivo médico* (SaMD): software com finalidade médica (diagnóstico, prevenção, monitoramento, tratamento)
que funciona como dispositivo médico por si só, sem ser parte de um equipamento. O MARIA:

- recebe mamografias de pacientes e produz informação usada na **decisão diagnóstica** (prioridade de leitura,
  probabilidade de malignidade, tipo de achado, localização da lesão e densidade);
- não é software meramente administrativo, de bem-estar ou de comunicação (exceções da norma).

Enquanto for **protótipo de pesquisa**, sem uso para decidir condutas, ele pode ser usado em estudo aprovado
por Comitê de Ética (CEP/CONEP), com o resultado sem influência no cuidado. **Uso assistencial exige
regularização na ANVISA.**

## 2. Finalidade de uso proposta (o que será declarado)

> *Software de apoio à **priorização** da leitura de mamografias por médico radiologista: indica regiões
> suspeitas, estima a probabilidade de malignidade calibrada, o tipo de achado e a densidade mamária, e
> sinaliza imagens inadequadas. **Não emite diagnóstico**; a decisão é sempre do radiologista.*

A finalidade é o principal fator da classe de risco. Declarar "priorização/apoio" (e não "diagnóstico
autônomo") e manter o radiologista na decisão são controles de risco importantes, mas **não tornam o software
de baixo risco** por si só: um falso negativo ainda pode atrasar o diagnóstico de câncer.

## 3. Classe de risco (RDC nº 751/2022)

A RDC 751/2022 trata da classificação de risco, dos regimes de notificação e registro e da rotulagem dos
dispositivos médicos, com **classes I (menor risco) a IV (maior risco)**: classes I e II seguem o regime de
**notificação**; classes III e IV, o de **registro**.

⚠️ A regra de classificação para software (regra 11 do anexo da RDC 751/2022, alinhada à regra equivalente do
regulamento europeu) atribui, em linhas gerais: software que fornece informação usada para decisões
diagnósticas ou terapêuticas → **classe II**; se essas decisões puderem causar **deterioração grave** do estado
de saúde ou intervenção cirúrgica → **classe III**; se puderem causar **morte ou deterioração irreversível** →
**classe IV**. **Confira a redação exata no texto vigente.**

**Análise para o MARIA:** um erro (falso negativo na prioridade ou no detector) pode atrasar o diagnóstico de
câncer de mama, com risco de progressão da doença. Isso aponta para **classe III**, podendo ser discutida a
**classe IV** dependendo de como a ANVISA avaliar o impacto do atraso. **Hipótese de trabalho do projeto:
classe III (registro).** Consequências:

| Exigência (classe III/IV) | Situação no projeto |
|---|---|
| Regime de **registro** na ANVISA (não basta notificação) | não iniciado |
| **Boas Práticas de Fabricação** certificadas (⚠️ RDC 665/2022) — sistema da qualidade tipo ISO 13485 | estrutura inicial em docs/qualidade (não certificado) |
| Dossiê técnico: descrição, arquitetura, **validação clínica**, gestão de risco, cibersegurança, rotulagem e instruções de uso | parcial (docs/) — falta validação clínica com dados brasileiros |
| Validação clínica representativa da população e dos equipamentos de uso | **não atende**: treino em filme digitalizado dos EUA; teste externo em 1 base do Egito; falta base brasileira multicêntrica (docs/dados) |
| Plano de monitoramento pós-mercado (tecnovigilância) e controle de mudanças do modelo | parcial: manifesto de versões, auditoria e retreino com regras de aceitação; falta procedimento formal |

## 4. Mudanças no modelo (retreino e calibração local)

O retreino e a calibração local alteram o comportamento do software. Para um produto registrado, mudanças
assim precisam ser **previstas e controladas** (o que pode mudar, com que dados, quais critérios de aceitação,
como reverter) e, conforme o impacto, comunicadas ou submetidas à ANVISA. O projeto já tem: critérios
automáticos de aceitação no retreino, versão do modelo em cada resultado (hash), histórico de versões com
restauração, auditoria de quem retreinou ou calibrou. **Antes do uso assistencial**, o retreino e a calibração
devem ser desligados ou formalizados num plano de controle de mudanças aprovado.

## 5. Próximos passos regulatórios

1. Contratar/consultar especialista regulatório e confirmar a classe com base no texto vigente.
2. Implantar sistema da qualidade (ISO 13485) e preparar a certificação de BPF, se classe III/IV.
3. Fazer o estudo de validação clínica com dados brasileiros (docs/dados/protocolo_dados_brasileiros.md).
4. Completar o dossiê (docs/qualidade, docs/seguranca) e as instruções de uso (COMO_USAR.txt como base).
5. Definir o plano de tecnovigilância e de controle de mudanças do modelo.
