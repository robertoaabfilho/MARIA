"""
Testes de verificação do MARIA. Cada teste tem o ID do requisito que verifica (docs/qualidade/03_requisitos_e_rastreabilidade.md).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import RAIZ, funcoes_do_dashboard


# ---------------- planilha e montagem ----------------
def test_REQ_ENT_01_planilha_excel_ptbr(tmp_path):
    """Planilha com ';', cp1252, cabeçalho com acento e vírgula decimal é lida corretamente."""
    import maria
    arq = tmp_path / "p.csv"
    arq.write_bytes("Mama;Densidade;Vista;Arquivo_Completa;Lesão;x0;y0;x1;y1;Idade;Fabricante\r\n"
                    "m1;b;cc;a.png;1;10;20;30,5;40;52;GE Senographe\r\n".encode("cp1252"))
    l = maria.ler_planilha(arq)[0]
    assert l["mama"] == "m1" and l["densidade"] == 2 and l["vista"] == "CC"
    assert l["caixa"] == (10.0, 20.0, 30.5, 40.0) and l["idade"] == "52" and l["fabricante"] == "GE Senographe"


def test_REQ_ENT_01b_formato_padrao_11_colunas(tmp_path):
    """Formato padrão: paciente, idade, arquivos, lado, vista, x/y_inicio/fim, densidade."""
    import maria
    arq = tmp_path / "p.csv"
    arq.write_bytes(("paciente;idade;arquivo_completa;arquivo_recorte;lado;vista;x_inicio;y_inicio;x_fim;y_fim;"
                     "densidade\r\n"
                     "P_00001;52;a.dcm;r.dcm;Esquerdo;mlo;10;20;30,5;40;C\r\n"
                     "P_00002;;b.dcm;;R;CC;;;;;\r\n").encode("cp1252"))
    l1, l2 = maria.ler_planilha(arq)
    assert l1["mama"] == "P_00001_esquerdo" and l1["vista"] == "MLO" and l1["idade"] == "52"
    assert l1["caixa"] == (10.0, 20.0, 30.5, 40.0) and l1["densidade"] == 3
    assert l2["mama"] == "P_00002_direito" and l2["caixa"] is None


def test_REQ_ENT_01c_nomes_das_bases_cbis_cdd(tmp_path):
    """Cabeçalhos originais do CBIS-DDSM são reconhecidos."""
    import maria
    arq = tmp_path / "p.csv"
    arq.write_text("patient_id,breast_density,left or right breast,image view\n"
                   "P_00005,3,RIGHT,CC\n", encoding="utf-8")
    l = maria.ler_planilha(arq)[0]
    assert (l["mama"], l["densidade"], l["vista"]) == ("P_00005_direito", 3, "CC")


def test_REQ_ENT_04_pareamento_cc_mlo_pelo_achado_previsto(monkeypatch, tmp_path):
    """CC + MLO só são juntados quando a mama tem exatamente um recorte de cada vista com o MESMO achado previsto;
    achados diferentes nas duas vistas ficam como casos separados."""
    import maria
    import torch
    from PIL import Image
    for n in ("a", "b", "c", "d", "e", "f", "g"):
        Image.new("L", (64, 64), 128).save(tmp_path / f"{n}.png")
    arq = tmp_path / "p.csv"
    arq.write_text("paciente,lado,vista,arquivo_recorte\n"
                   "P1,esquerdo,CC,a.png\nP1,esquerdo,MLO,b.png\n"                      # mesmo achado -> junta
                   "P2,direito,CC,c.png\nP2,direito,MLO,d.png\n"                         # achados diferentes -> separa
                   "P3,direito,CC,e.png\nP3,direito,CC,f.png\nP3,direito,MLO,g.png\n",  # 2 massas na CC -> separa
                   encoding="utf-8")
    previsto = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 0, "f": 0, "g": 0}      # 0 = massa, 1 = calcificação
    monkeypatch.setattr(maria, "ler_imagem", lambda c: (Path(c).stem, False))
    monkeypatch.setattr(maria, "entrada_achado", lambda arr, eh: arr)
    monkeypatch.setattr(maria, "prob_achado_recorte",
                        lambda m, nome, d: torch.tensor([1.0, 0.0] if previsto[nome] == 0 else [0.0, 1.0]))
    recortes, _, mama_de, _, _, fixos = maria.montar_amostras(tmp_path, maria.ler_planilha(arq))
    assert maria.parear_por_achado(recortes, mama_de, fixos, None, "cpu") == 1
    assert sorted(recortes["P1_esquerdo_massa"]) == ["CC", "MLO"]
    assert all(len(v) == 1 for k, v in recortes.items() if k != "P1_esquerdo_massa")
    assert len(recortes) == 6


def test_REQ_ENT_04b_coluna_lesao_forca_o_par(tmp_path):
    """Com a coluna opcional "lesao", o par CC + MLO é o que o médico informou (não passa pelo modelo)."""
    import maria
    from PIL import Image
    for n in ("a", "b"):
        Image.new("L", (64, 64), 128).save(tmp_path / f"{n}.png")
    arq = tmp_path / "p.csv"
    arq.write_text("paciente,lado,vista,arquivo_recorte,lesao\nP1,esquerdo,CC,a.png,1\nP1,esquerdo,MLO,b.png,1\n",
                   encoding="utf-8")
    recortes, *_, fixos = maria.montar_amostras(tmp_path, maria.ler_planilha(arq))
    assert sorted(recortes["P1_esquerdo_1"]) == ["CC", "MLO"] and "P1_esquerdo_1" in fixos


def test_REQ_ENT_06_dashboard_padroniza_valores():
    """O editor do dashboard aceita sinônimos e deixa vazio o que não reconhece."""
    import pandas as pd
    ns = funcoes_do_dashboard("_chave_coluna", "planilha_vazia", "padronizar_planilha", "separar_lado")
    df = ns["padronizar_planilha"](pd.DataFrame({"Patient_ID": ["P_1", "P_2"], "Side": ["L", "xyz"],
                                                 "caixa_esq": ["5,5", ""]}))
    assert (df.at[0, "paciente"], df.at[0, "lado"], df.at[0, "x_inicio"]) == ("P_1", "esquerdo", 5.5)
    assert pd.isna(df.at[1, "lado"])
    assert ns["separar_lado"]("pacienteA_esq") == ("pacienteA", "esquerdo")


def test_REQ_ENT_02_caixa_invalida_ignorada(tmp_path):
    import maria
    arq = tmp_path / "p.csv"
    arq.write_text("mama,arquivo_completa,x0,y0,x1,y1\nm1,a.png,50,50,10,10\n", encoding="utf-8")
    assert maria.ler_planilha(arq)[0]["caixa"] is None


# ---------------- detecção / marcação ----------------
def test_REQ_DET_01_coincidencia_de_caixas():
    import maria
    assert maria.coincide((0, 0, 10, 10), (2, 2, 12, 12))          # sobreposição grande
    assert maria.coincide((0, 0, 100, 100), (40, 40, 60, 60))      # uma dentro da outra
    assert not maria.coincide((0, 0, 10, 10), (50, 50, 60, 60))    # longe


# ---------------- saída: calibração, prioridade, qualidade ----------------
def test_REQ_SAI_01_calibracao_monotona_e_limitada():
    import maria
    maria.CONTEXTO["cal"] = json.loads((RAIZ / "modelo/calibracao.json").read_text(encoding="utf-8"))
    ps = [maria.calibrar(p) for p in np.linspace(0.01, 0.99, 50)]
    assert all(0 < p < 1 for p in ps) and all(b >= a for a, b in zip(ps, ps[1:]))


def test_REQ_SAI_02_prioridade_por_limites():
    import maria
    maria.CONTEXTO["cal"] = {"prioridade": {"alta": 0.5, "media": 0.2}}
    assert maria.prioridade_de(0.8, 0.01, "ok") == "alta"
    assert maria.prioridade_de(0.3, 0.01, "ok") == "média"
    assert maria.prioridade_de(0.1, 0.01, "ok") == "baixa"
    assert maria.prioridade_de(0.9, 0.01, "não usar") == "não avaliar (qualidade)"


def test_REQ_SAI_03_qualidade_detecta_imagem_inadequada():
    import maria
    uniforme = np.full((500, 500), 100.0)
    assert any(n == "não usar" for n, _ in maria.checar_qualidade(uniforme, {}, "completa"))
    pequeno = np.random.RandomState(0).rand(20, 20) * 1000
    assert any(n == "não usar" for n, _ in maria.checar_qualidade(pequeno, {}, "recorte"))
    assert any(n == "não usar" for n, _ in maria.checar_qualidade(np.random.rand(400, 400) * 1000,
                                                                     {"modalidade": "CT"}, "recorte"))
    baixa = np.random.RandomState(0).rand(600, 400) * 1000
    assert any("resolução baixa" in m for _, m in maria.checar_qualidade(baixa, {}, "completa"))


def test_REQ_SAI_04_resultado_incerto_sinalizado():
    import maria, torch
    maria.CONTEXTO["cal"] = None
    r = maria.resultado("a", "recorte fornecido", "recorte", [(0.5, 0.30)], [torch.tensor([0.6, 0.4])], 0.5)
    assert r["qualidade"] == "atenção" and "incerto" in r["avisos_qualidade"]


# ---------------- rastreabilidade ----------------
def test_REQ_RAS_01_manifesto_confere():
    import manifesto
    atual, registrada, div = manifesto.verificar()
    assert len(atual) == 12 and registrada is not None


# ---------------- LGPD: acesso, auditoria, anonimização ----------------
def test_REQ_LGPD_01_senha_com_hash_e_autenticacao(isolado):
    import acesso
    acesso.criar("ana", "Ana", "administrador", "segredo123")
    bruto = acesso.ARQ_USUARIOS.read_text(encoding="utf-8")
    assert "segredo123" not in bruto and "scrypt$" in bruto
    assert acesso.autenticar("ana", "segredo123")["perfil"] == "administrador"
    assert acesso.autenticar("ana", "errada000") is None
    with pytest.raises(ValueError):
        acesso.criar("bia", "Bia", "usuario", "curta")              # senha fraca
    with pytest.raises(ValueError):
        acesso.remover("ana")                                       # único administrador


def test_REQ_LGPD_02_auditoria_detecta_adulteracao(isolado):
    import auditoria
    for i in range(5):
        auditoria.registrar("ana", "classificação", {"i": i})
    assert auditoria.verificar()[0]
    linhas = auditoria.ARQ_AUDITORIA.read_text(encoding="utf-8").splitlines()
    linhas[2] = linhas[2].replace('"i": 2', '"i": 99')               # altera um evento
    auditoria.ARQ_AUDITORIA.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    ok, _, problema = auditoria.verificar()
    assert not ok and "linha 3" in problema
    del linhas[2]                                                    # apaga um evento
    auditoria.ARQ_AUDITORIA.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    assert not auditoria.verificar()[0]


def test_REQ_LGPD_03_pseudonimo_estavel_e_irreversivel(isolado):
    import anonimizacao
    a, b = anonimizacao.pseudonimizar("Maria Silva"), anonimizacao.pseudonimizar("Maria Silva")
    assert a == b and "Maria" not in a and a != anonimizacao.pseudonimizar("Joana")


def test_REQ_LGPD_04_dicom_anonimizado(isolado):
    import anonimizacao, pydicom
    origem = next((RAIZ / "exemplos/1_dicom_filme_cbis").glob("*recorte.dcm"))
    destino = isolado / "anon.dcm"
    original = pydicom.dcmread(origem)
    original.InstitutionName = "Hospital X"; original.PatientBirthDate = "19700101"
    original.save_as(isolado / "orig.dcm")
    anonimizacao.anonimizar_dicom(isolado / "orig.dcm", destino)
    d = pydicom.dcmread(destino)
    assert "PatientName" not in d and "InstitutionName" not in d and "PatientBirthDate" not in d
    assert d.PatientIdentityRemoved == "YES" and d.PatientID.startswith("PAC-")
    assert d.SOPInstanceUID != original.SOPInstanceUID
    assert (d.pixel_array == original.pixel_array).all()             # a imagem não muda


# ---------------- calibração local (dashboard) ----------------
def test_REQ_CAL_01_auc_e_platt_conferem_com_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    from sklearn.linear_model import LogisticRegression
    ns = funcoes_do_dashboard("auc_numpy", "ajustar_platt", "aplicar_platt", "limiar_para_sensibilidade",
                              "sens_espec")
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 400); p = np.clip(0.3 * y + rng.rand(400) * 0.7, 0.01, 0.99)
    assert abs(ns["auc_numpy"](y, p) - sk.roc_auc_score(y, p)) < 1e-9
    a, b = ns["ajustar_platt"](p, y)
    lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    assert abs(a - lr.coef_[0, 0]) < 1e-3 and abs(b - lr.intercept_[0]) < 1e-3
    lim = ns["limiar_para_sensibilidade"](p, y, 0.9)
    assert ns["sens_espec"](p, y, lim)[0] >= 0.9


# ---------------- retreino ----------------
def test_REQ_RET_01_retreino_so_usa_casos_validos(tmp_path):
    import pandas as pd, retreinar
    a = pd.DataFrame([
        {"mama": "M1", "origem_lesao": "detectada pelo modelo", "imagem": "i1.dcm", "lesao_real": "sim",
         "achado_confirmado": "Massa/Nódulo", "exame_confirmacao": "Biópsia (anatomopatológico)",
         "densidade_confirmada": "C", "x0": 1, "y0": 1, "x1": 5, "y1": 5, "amostra": "a1"},
        {"mama": "M1", "origem_lesao": "detectada pelo modelo", "imagem": "i1.dcm", "lesao_real": "",
         "achado_confirmado": "", "exame_confirmacao": "", "densidade_confirmada": "", "x0": 9, "y0": 9, "x1": 12,
         "y1": 12, "amostra": "a2"},
        {"mama": "M2", "origem_lesao": "recorte fornecido", "imagem": "r.png", "lesao_real": "",
         "achado_confirmado": "Microcalcificação", "exame_confirmacao": "", "densidade_confirmada": "",
         "x0": np.nan, "y0": np.nan, "x1": np.nan, "y1": np.nan, "amostra": "a3"}])
    c = retreinar.casos_disponiveis(a)
    assert "i1.dcm" not in c["detector"]                             # caixa sem revisão: imagem fora
    assert len(c["achado"]) == 1                                     # sem exame de confirmação: fora
    assert list(c["densidade"]["mama"]) == ["M1"]
