# Protocolo para dados do mundo real brasileiros (proposta)

**Problema:** o MARIA foi treinado em mamografia de filme digitalizada (CBIS-DDSM, EUA, década de 1990). Os
equipamentos do SUS são digitais e de vários fabricantes. No teste externo (CDD-CESM), o **ranqueamento se
manteve** (AUC ≈ 0,81; 0,85–0,87 em massas e calcificações), mas o **limiar 0,5 não transfere**
(especificidade 65%). Sem dados brasileiros não é possível afirmar o desempenho no uso real.

## 1. Objetivos
1. Validar o desempenho (sensibilidade, especificidade, AUC, calibração) em exames de serviços brasileiros.
2. Recalibrar a probabilidade e os limites de prioridade por serviço/fabricante.
3. Se necessário, ajustar o modelo (retreino) com dados brasileiros, mantendo um conjunto de teste independente.

## 2. Desenho
Estudo **retrospectivo, multicêntrico**, com aprovação no CEP de cada centro (Plataforma Brasil, Res. CNS
466/2012), dados **anonimizados** na origem (ferramenta do MARIA ou do PACS), com dispensa de TCLE justificada
ou TCLE, conforme o CEP.

## 3. Critérios
- **Inclusão:** mamografias (CC e MLO) de mulheres ≥ 40 anos (e um estrato de < 40 anos), de rastreamento e
  diagnóstico, de **pelo menos 3 fabricantes** (ex.: GE, Hologic, Siemens, Fujifilm, Carestream) e 4 centros
  de regiões diferentes.
- **Rótulo de referência (confiável):**
  - **maligno**: confirmação histopatológica (biópsia ou cirurgia);
  - **benigno**: histopatologia benigna **ou** seguimento com estabilidade por **≥ 2 anos** (BI-RADS 1–2 no
    seguimento);
  - **normal**: exame BI-RADS 1–2 com seguimento negativo ≥ 2 anos.
- **Lesões:** contorno ou caixa marcada por radiologista (a ferramenta de marcação do MARIA pode ser usada),
  com dupla leitura numa amostra para medir concordância.
- **Variáveis:** idade, densidade BI-RADS do laudo, fabricante e modelo do equipamento, tipo de achado,
  categoria BI-RADS, resultado anatomopatológico.

## 4. Tamanho da amostra (ordem de grandeza)
Para estimar a sensibilidade com intervalo de confiança de 95% de ±5 pontos, são necessários ≈ 250–300
cânceres; para especificidade em rastreamento, alguns milhares de exames normais. Para análise por fabricante e
por faixa etária, cada subgrupo precisa de dezenas de cânceres. **Estimativa final com estatístico.**

## 5. Divisão dos dados
Separação **por paciente e por centro**: um centro inteiro reservado como teste externo nunca usado em
calibração ou retreino. Calibração e limiar ajustados em dados de calibração, nunca no teste.

## 6. Análises
Sensibilidade e especificidade da "prioridade alta"; AUC; curva de calibração e ECE; FROC do detector; tudo
por **fabricante, densidade, faixa etária e tipo de achado**, com intervalos de confiança. O dashboard do MARIA
(desempenho local e calibração local) faz parte dessas análises a partir do banco de anotações.

## 7. Fluxo no MARIA
1. Cada centro roda o MARIA, revisa os resultados e registra a confirmação (biópsia/seguimento) na aba 3.
2. Exporta o banco **anonimizado** (.zip) e envia ao centro coordenador.
3. O centro coordenador importa, analisa, recalibra/retreina e valida no centro de teste.
