# Segurança da informação e LGPD

> Dado de saúde é **dado pessoal sensível** (LGPD — Lei nº 13.709/2018, art. 5º, II). Este documento descreve
> os controles do MARIA e o que a instituição precisa definir. Deve ser revisado pelo **encarregado (DPO)**.

## 1. Base legal (a definir pela instituição)
- **Uso assistencial** (apoio ao radiologista no atendimento): tratamento para **tutela da saúde**, em
  procedimento realizado por profissionais de saúde (LGPD, art. 11, II, "f").
- **Pesquisa** (validação, retreino, publicação): **realização de estudos por órgão de pesquisa**, com
  **anonimização sempre que possível** (art. 11, II, "c"), com aprovação do Comitê de Ética (Res. CNS 466/2012)
  e Termo de Consentimento ou sua dispensa justificada.
- Elaborar o **Relatório de Impacto à Proteção de Dados (RIPD)** (art. 38) e registrar as operações (art. 37).

## 2. Controles implementados no MARIA
| Controle | Como | Onde |
|---|---|---|
| Controle de acesso | Login obrigatório; usuários locais; senha mínima de 8 caracteres com letras e números; hash scrypt com sal (a senha não é guardada); perfis usuário/administrador; bloqueio de 5 min após 5 tentativas; sessão expira após 30 min sem uso | modelo/acesso.py; dashboard |
| Mínimo privilégio | Retreino, restauração, calibração local, exportação/importação do banco, usuários e auditoria: só administrador | dashboard |
| Registro de uso (auditoria) | Login/logout/recusas, classificações (pasta, quantidade, versão do modelo, hash do resultado), anotações, exportações, retreinos, calibrações, nó DICOM; arquivo só de acréscimo com **hash encadeado** (adulteração detectável) | modelo/auditoria.py; página Administração |
| Pseudonimização | Fora do atendimento (banco de anotações, exportação, retreino) os identificadores viram códigos HMAC com chave local | modelo/anonimizacao.py |
| Anonimização de imagens | DICOM copiados para o banco/exportação: remove nome, ID, nascimento, endereço, instituição, médicos, acesso, datas e tags privadas; troca UIDs. PNG/JPG regravados sem metadados | modelo/anonimizacao.py |
| Minimização | Nó DICOM apaga as imagens recebidas após processar; banco guarda só as lesões revisadas | integracao/no_dicom.py |
| Retenção | Exclusão de resultados antigos (prazo definido pelo administrador) | página Administração |
| Segredos fora do código | usuarios.json, chave de pseudonimização, configurações locais e dados de pacientes fora do git | .gitignore |
| Integridade do modelo | Hash dos pesos conferido a cada execução | modelo/manifesto.py |
| Integração | Nó DICOM aceita só AE titles autorizados | integracao/no_dicom.py |

## 3. Responsabilidades da instituição (infraestrutura)
- **Criptografia do disco** onde ficam `resultados/`, `anotacoes/`, `auditoria/` e `integracao/` (BitLocker,
  LUKS etc.); **backup** cifrado, incluindo `config/usuarios.json` e a chave de pseudonimização.
- Rodar o dashboard e o nó DICOM **só na rede interna** (firewall), de preferência atrás de HTTPS/VPN; DICOM com
  TLS quando o PACS suportar.
- Contas nominais (sem login compartilhado); revisar usuários periodicamente; remover desligados.
- Definir prazos de retenção conforme a política de prontuário e de pesquisa.
- Atualizações de segurança do sistema operacional e das bibliotecas (ver SOUP).

## 4. Direitos dos titulares
Pedidos de acesso, correção ou exclusão: localizar pelos identificadores nos resultados (identificados) ou,
no banco pseudonimizado, recalculando o pseudônimo com a chave local (`anonimizacao.pseudonimizar`).

## 5. Limitações conhecidas
Textos gravados **dentro dos pixels** da imagem (alguns equipamentos escrevem nome/data na imagem) não são
removidos pela anonimização. Login local (sem integração com o diretório da instituição).
