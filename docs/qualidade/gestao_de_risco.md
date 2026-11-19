# Gestão de risco (base: ISO 14971)

> Análise inicial da equipe de desenvolvimento. Probabilidade (P) e severidade (S) em escala de 1 a 5;
> risco = P × S. Aceitabilidade a ser definida pela política de risco do fabricante. Revisar com usuários
> clínicos e atualizar a cada mudança do modelo.

| # | Perigo / situação perigosa | Dano possível | P×S antes | Controles implementados | P×S depois | Observações / pendências |
|---|---|---|---|---|---|---|
| R1 | Falso negativo: lesão maligna com prioridade baixa ou não detectada | Atraso no diagnóstico de câncer | 3×5=15 | Radiologista lê todos os exames (finalidade: priorização, não diagnóstico); avisos na tela e nas imagens DICOM; limiar ajustável e calibração local por sensibilidade-alvo; incerteza sinalizada | 2×5=10 | **Risco residual alto**: validar sensibilidade com dados brasileiros; definir sensibilidade mínima aceitável |
| R2 | Falso positivo: prioridade alta indevida | Ansiedade, exames/biópsias desnecessários | 4×2=8 | Radiologista decide; P calibrada; especificidade mostrada por subgrupo; calibração local | 3×2=6 | Especificidade de 65% fora da base de treino: calibrar localmente |
| R3 | Uso em população/equipamento diferente do treino | Desempenho pior que o informado | 4×4=16 | Avisos de limitação; validação externa por subgrupo; desempenho local por fabricante; calibração local | 3×4=12 | Exige estudo multicêntrico brasileiro |
| R4 | Imagem inadequada (outra modalidade, baixa resolução, saturada, sem contraste) | Resultado sem sentido usado na decisão | 3×4=12 | Controle de qualidade: "não usar" bloqueia a prioridade; "atenção" alerta | 1×4=4 | Ampliar checagens (posicionamento, implante, artefatos) |
| R5 | Confiança excessiva na probabilidade ("automação") | Decisão sem leitura crítica | 3×4=12 | Texto "não é diagnóstico" na tela e no PACS; mapa de calor para conferência; incerteza | 2×4=8 | Treinamento dos usuários; estudo de usabilidade (IEC 62366) |
| R6 | Resultado atribuído à paciente errada | Conduta na paciente errada | 2×5=10 | Nó DICOM grava o resultado no mesmo estudo/paciente (UIDs DICOM); IDs no resultado; auditoria | 1×5=5 | Teste com o PACS real do serviço |
| R7 | Versão de modelo errada/corrompida | Desempenho diferente do validado | 2×4=8 | Manifesto com hash conferido a cada execução; versão gravada em cada resultado | 1×4=4 | — |
| R8 | Retreino ou calibração local pioram o modelo | Desempenho pior em campo | 3×4=12 | Só administrador; critérios automáticos no teste fixo; versão anterior guardada e restaurável; auditoria | 2×4=8 | Para uso assistencial: plano de controle de mudanças aprovado |
| R9 | Anotações erradas no banco (rótulo sem confirmação) | Retreino ensina erros | 3×3=9 | Achado só entra com exame de confirmação; patologia não é retreinada; detector só com imagens totalmente revisadas | 2×3=6 | — |
| R10 | Vazamento de dados de pacientes | Violação de privacidade (LGPD) | 3×4=12 | Login, perfis, auditoria, pseudonimização/anonimização fora do atendimento, retenção, nó DICOM apaga recebidos | 2×4=8 | Criptografia de disco e backup (infraestrutura); RIPD |
| R11 | Acesso não autorizado ao nó DICOM | Envio/recebimento indevido de exames | 3×3=9 | Lista de AE autorizados (recusa com 0x0124); auditoria | 2×3=6 | Restringir também por rede/firewall; TLS no DICOM |
| R12 | Falha de software (exceção, arquivo corrompido) | Resultado ausente ou incompleto | 3×2=6 | Erros por imagem não derrubam a rodada ("indisponível"); log; testes automatizados | 2×2=4 | — |

**Conclusão provisória:** o risco residual dominante é **R1/R3 (desempenho em dados brasileiros)**, que só se
reduz com validação clínica multicêntrica. Até lá, o MARIA deve ser usado apenas em pesquisa.
