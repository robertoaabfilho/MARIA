"""
Testes de verificação do MARIA. Cada teste tem o ID do requisito que verifica (docs/qualidade/03_requisitos_e_rastreabilidade.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import RAIZ, funcoes_do_dashboard, precisa_da_release


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
    pares, divergentes = maria.parear_por_achado(recortes, mama_de, fixos, None, "cpu")
    assert pares == 1 and sorted(divergentes) == ["P2_direito_cc", "P2_direito_mlo"]   # só a P2 diverge
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


def test_REQ_SAI_03b_regiao_breast_com_lado_nao_gera_aviso():
    """BodyPartExamined 'RIGHT BREAST' / 'Left Breast' é mamografia: não pode gerar aviso de região."""
    import maria
    img = np.random.default_rng(0).uniform(0, 4000, (1200, 800))
    for regiao in ("BREAST", "RIGHT BREAST", "Left Breast", "MAMA"):
        assert not any("região" in m for _, m in maria.checar_qualidade(img, {"regiao": regiao}, "completa"))
    assert any("região" in m for _, m in maria.checar_qualidade(img, {"regiao": "CHEST"}, "completa"))


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
    precisa_da_release(RAIZ / "exemplos/1_dicom_filme_cbis/P_00066_LEFT_CC_recorte.dcm")
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


def test_REQ_DEN_01_densidade_carrega_rede_do_json():
    """O modelo de densidade instalado carrega (sem chaves sobrando ou faltando) na rede que o densidade.json indica,
    e devolve 4 probabilidades que somam 1."""
    import torch
    import maria
    pasta = RAIZ / "modelo/modelo_densidade"
    precisa_da_release(pasta / "densidade.pth")
    cfg = json.loads((pasta / "densidade.json").read_text(encoding="utf-8"))
    m = maria.modelo_densidade(cfg.get("arquitetura", "resnet50"))
    m.load_state_dict(torch.load(pasta / "densidade.pth", map_location="cpu"), strict=True)
    c = maria.ClassificadorDensidade(pasta, torch.device("cpu"))
    p = c.estimar(np.random.default_rng(0).uniform(0, 4000, (600, 400)).astype(np.float32))
    assert p.shape == (4,) and abs(float(p.sum()) - 1) < 1e-4


def test_REQ_PAT_01_janela_de_contexto():
    """Janela de contexto: tela 512 x 512, centrada na lesão; o fundo (sem tecido) fica preto."""
    import maria
    arr = np.zeros((3000, 2000), np.float32)
    arr[:, :1200] = np.random.default_rng(0).uniform(500, 4000, (3000, 1200))     # tecido à esquerda, fundo à direita
    img = maria.normalizar_mamografia(arr)
    assert img.dtype == np.uint8 and img[:, 1500:].max() == 0
    j = maria.janela_contexto(img, (900, 1400, 1000, 1500))
    assert j.shape == (maria.MC_TELA, maria.MC_TELA) and j.dtype == np.uint8
    assert j[:, :200].mean() > 0 and j[:, -100:].max() == 0      # tecido à esquerda da lesão, fundo à direita


def test_REQ_PAT_02_calibracao_e_limiar_por_entrada():
    """Recorte + contexto usa a calibração e o limite de prioridade próprios; com o limiar automático, 'Maligno'
    coincide com 'prioridade alta' nas duas entradas."""
    import maria
    maria.CONTEXTO["cal"] = {"patologia": {"a": 1.0, "b": 0.0}, "patologia_contexto": {"a": 1.0, "b": 0.0},
                             "prioridade": {"alta": 0.5, "media": 0.2}, "prioridade_contexto": {"alta": 0.37, "media": 0.2}}
    ctx = maria.resultado("x", maria.ORIGEM_MODELO, "", [(0.42, 0.01)], [], None, patologia_entrada=maria.ENTRADA_CONTEXTO)
    rec = maria.resultado("y", maria.ORIGEM_RECORTE, "", [(0.42, 0.01)], [], None, patologia_entrada=maria.ENTRADA_RECORTE)
    assert (ctx["prioridade"], ctx["patologia"], ctx["limiar_maligno"]) == ("alta", "Maligno", "0.37")
    assert (rec["prioridade"], rec["patologia"], rec["limiar_maligno"]) == ("média", "Benigno", "0.50")
    manual = maria.resultado("z", maria.ORIGEM_MODELO, "", [(0.42, 0.01)], [], 0.6, patologia_entrada=maria.ENTRADA_CONTEXTO)
    assert manual["patologia"] == "Benigno" and manual["prioridade"] == "alta"      # limiar manual só muda o rótulo


def test_REQ_PAT_03_pesos_de_contexto_carregam():
    """Os 5 pesos de contexto instalados carregam na rede do maria.py (sem chaves sobrando ou faltando)."""
    import torch
    import maria
    precisa_da_release(*[RAIZ / f"modelo/modelo_patologia/contexto_dobra_{k}.pth" for k in range(1, 6)])
    for k in range(1, 6):
        m = maria.ClassificadorMammoCLIP()
        m.load_state_dict(torch.load(RAIZ / f"modelo/modelo_patologia/contexto_dobra_{k}.pth", map_location="cpu"),
                          strict=True)


# ---------------- prioridade da mama: híbrido GMIC + MARIA ----------------
def _cfg_hibrido():
    return json.loads((RAIZ / "modelo/calibracao.json").read_text(encoding="utf-8"))["hibrido_gmic"]


def test_REQ_PRI_01_escore_da_mama_monotono_e_limites():
    """O escore da mama sobe com a P do GMIC e com a P do MARIA; a prioridade segue os limites do calibracao.json."""
    import maria
    cfg = _cfg_hibrido()
    grade = np.linspace(0.01, 0.99, 25)
    for fixo in (0.05, 0.5, 0.95):
        e_g = [maria.escore_mama(p, fixo, cfg) for p in grade]
        e_m = [maria.escore_mama(fixo, p, cfg) for p in grade]
        assert all(b > a for a, b in zip(e_g, e_g[1:])) and all(b > a for a, b in zip(e_m, e_m[1:]))
        assert all(0 < e < 1 for e in e_g + e_m)
    assert maria.prioridade_hibrida(cfg["alta"], cfg) == "alta"
    assert maria.prioridade_hibrida((cfg["alta"] + cfg["media"]) / 2, cfg) == "média"
    assert maria.prioridade_hibrida(cfg["media"] - 1e-6, cfg) == "baixa"


def test_REQ_PRI_02_hibrido_aplicado_por_mama():
    """A prioridade da mama vale para todas as linhas da mama; a da lesão fica em prioridade_lesao; 'não avaliar'
    continua; mama sem GMIC (só recorte) fica com a prioridade da lesão; mama sem lesão usa maria_sem_lesao."""
    import maria
    cfg = _cfg_hibrido()
    L = lambda a, mama, pri, p, origem=maria.ORIGEM_MODELO, img=None: {
        "amostra": a, "_mama": mama, "_imagem": img or a.rsplit("_", 1)[0], "prioridade": pri, "p_maligno": p,
        "origem_lesao": origem}
    res = [L("m1_cc_modelo1", "m1", "baixa", "0.9000"), L("m1_mlo_modelo1", "m1", "não avaliar (qualidade)", "0.2000"),
           {**L("m2_cc", "m2", "baixa (nada detectado)", "N/D", maria.ORIGEM_NENHUMA), "_imagem": "m2_cc"},
           L("r3", "m3", "média", "0.3000", maria.ORIGEM_RECORTE, img="")]
    gmic_img = {"m1_cc": (0.95, "mapa_m1.png"), "m1_mlo": (0.85, ""), "m2_cc": (0.10, "")}
    mama_de = {"m1_cc": "m1", "m1_mlo": "m1", "m2_cc": "m2"}
    assert maria.aplicar_hibrido(res, gmic_img, mama_de, cfg) == 2
    a, b, c, d = res
    e1 = maria.escore_mama(0.90, 0.9, cfg)
    assert a["escore_mama"] == f"{e1:.4f}" and a["prioridade"] == maria.prioridade_hibrida(e1, cfg) == "alta"
    assert a["prioridade_lesao"] == "baixa" and a["prioridade_origem"] == "mama (GMIC + MARIA)"
    assert a["mapa_gmic"] == "mapa_m1.png"
    assert b["prioridade"] == "não avaliar (qualidade)" and b["prioridade_mama"] == "alta"
    assert c["escore_mama"] == f"{maria.escore_mama(0.10, cfg['maria_sem_lesao'], cfg):.4f}"
    assert d["prioridade"] == "média" and d["prioridade_origem"] == "lesão (MARIA)" and "escore_mama" not in d


def test_REQ_PRI_03_lado_da_parede_e_modulo_gmic_conferido(tmp_path, monkeypatch):
    """A ponte acha a parede torácica (para pôr a mama na orientação do GMIC) e recusa um GMIC ausente ou alterado."""
    import gmic_maria
    img = np.zeros((300, 200), np.float32)
    img[50:250, :120] = 200                       # mama encostada à esquerda
    assert not gmic_maria.parede_a_direita(img) and gmic_maria.parede_a_direita(np.fliplr(img))
    monkeypatch.setattr(gmic_maria, "PASTA", tmp_path / "modulo_gmic")
    assert not gmic_maria.disponivel() and len(gmic_maria.conferir()) == 6
    (tmp_path / "modulo_gmic/models").mkdir(parents=True)
    (tmp_path / "modulo_gmic/models/sample_model_1.p").write_bytes(b"adulterado")
    assert "sample_model_1.p: SHA-256 diferente" in gmic_maria.conferir()


def test_REQ_ENT_07_vista_pelo_patient_orientation():
    """Sem ViewPosition, a vista vem do PatientOrientation (P\\FL / A\\FR = MLO; P\\L / A\\R = CC)."""
    import maria
    from types import SimpleNamespace as N
    assert maria.vista_dicom(N(ViewPosition="MLO", PatientOrientation=["P", "L"])) == "MLO"
    assert maria.vista_dicom(N(ViewPosition="", PatientOrientation=["P", "FL"])) == "MLO"
    assert maria.vista_dicom(N(ViewPosition="", PatientOrientation=["A", "FR"])) == "MLO"
    assert maria.vista_dicom(N(ViewPosition="", PatientOrientation=["A", "R"])) == "CC"
    assert maria.vista_dicom(N()) == "SEM_VISTA"


# ---------------------------------------------------------------- desempenho local, calibração e modo sombra
def _banco_df(linhas):
    import pandas as pd
    ns = funcoes_do_dashboard()
    cols = ns["COLUNAS_ANOTACAO"]
    return pd.DataFrame([{c: "" for c in cols} | l for l in linhas]).reindex(columns=cols, fill_value="").astype(str)


BIOPSIA, SEGUIMENTO = "Biópsia (anatomopatológico)", "Seguimento (estável por ≥ 2 anos)"


def test_REQ_LOC_01_rotulo_por_mama():
    """Mama maligna = lesão maligna confirmada ou 'mama confirmada' maligna; não maligna só com confirmação explícita
    (benigna/normal) e exame válido; sem exame válido, a mama não entra."""
    import desempenho_local as DL
    b = _banco_df([
        {"rodada": "r1", "mama": "M1", "origem_lesao": "detectada pelo modelo", "p_maligno": "0.8", "escore_mama": "0.7",
         "patologia_confirmada": "Maligno", "exame_confirmacao": BIOPSIA, "lesao_real": "sim"},
        {"rodada": "r1", "mama": "M2", "origem_lesao": "nenhuma lesão encontrada", "escore_mama": "0.1",
         "mama_confirmada": "Normal", "exame_confirmacao": SEGUIMENTO},
        {"rodada": "r1", "mama": "M3", "origem_lesao": "detectada pelo modelo", "p_maligno": "0.3",
         "mama_confirmada": "Benigna", "exame_confirmacao": "Ultrassonografia"},           # exame não vale
        {"rodada": "r1", "mama": "M4", "origem_lesao": "detectada pelo modelo", "p_maligno": "0.4",
         "patologia_confirmada": "Benigno", "exame_confirmacao": BIOPSIA, "lesao_real": "sim"},  # sem 'mama confirmada'
    ])
    m = DL.mamas_confirmadas(b).set_index("grupo")
    assert set(m.index) == {"r1|M1", "r1|M2"}
    assert m.loc["r1|M1", "y"] == 1 and m.loc["r1|M1", "p_maria"] == 0.8 and m.loc["r1|M1", "escore"] == 0.7
    assert m.loc["r1|M2", "y"] == 0 and bool(m.loc["r1|M2", "normal"]) and np.isnan(m.loc["r1|M2", "p_maria"])
    assert len(DL.lesoes_confirmadas(b)) == 2


def test_REQ_LOC_02_intervalo_de_confianca_e_estabilidade():
    import desempenho_local as DL
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400); p = y * 0.3 + rng.random(400) * 0.7
    v, inf, sup = DL.ic_bootstrap(DL.auc, y, p)
    assert inf < v < sup and sup - inf < 0.15
    v2, inf2, sup2 = DL.ic_bootstrap(DL.auc, y[:40], p[:40])
    assert sup2 - inf2 > sup - inf                     # menos casos -> intervalo mais largo
    assert abs(DL.auc(y, p) - ns_auc(y, p)) < 1e-9     # mesma AUC do dashboard
    assert DL.estabilidade(y[:8]) == "insuficiente" and DL.estabilidade(y[:25]) == "instável"
    assert DL.estabilidade(y) == "ok" and DL.pode_calibrar(y) and not DL.pode_calibrar(y[:40])


def ns_auc(y, p):
    return funcoes_do_dashboard("auc_numpy")["auc_numpy"](y, p)


def test_REQ_LOC_03_fila_de_casos_para_anotar():
    """Discordância GMIC × MARIA e mama em prioridade alta sem lesão vêm primeiro; casos já anotados saem."""
    import pandas as pd
    import desempenho_local as DL
    r = pd.DataFrame([
        {"amostra": "a", "mama": "m1", "origem_lesao": "detectada pelo modelo", "prioridade": "baixa", "p_gmic_mama": "0.9",
         "p_maligno": "0.05", "p_maligno_calibrada": "0.05", "incerteza": "0.01", "fabricante": "X", "escore_mama": "0.3"},
        {"amostra": "b", "mama": "m2", "origem_lesao": "nenhuma lesão encontrada", "prioridade": "alta",
         "p_gmic_mama": "0.5", "p_maligno": "", "p_maligno_calibrada": "", "incerteza": "", "fabricante": "X",
         "escore_mama": "0.5"},
        {"amostra": "c", "mama": "m3", "origem_lesao": "detectada pelo modelo", "prioridade": "baixa", "p_gmic_mama": "0.1",
         "p_maligno": "0.45", "p_maligno_calibrada": "0.2", "incerteza": "0.01", "fabricante": "X", "escore_mama": "0.1"},
        {"amostra": "d", "mama": "m4", "origem_lesao": "detectada pelo modelo", "prioridade": "alta", "p_gmic_mama": "0.8",
         "p_maligno": "0.9", "p_maligno_calibrada": "0.9", "incerteza": "0.01", "fabricante": "X", "escore_mama": "0.9"},
    ])
    cfg = json.loads((RAIZ / "modelo/calibracao.json").read_text(encoding="utf-8"))["hibrido_gmic"]
    f = DL.fila_para_anotar(r, {"d"}, {"X": 50}, cfg_hibrido=cfg)
    assert set(list(f["amostra"])[:2]) == {"a", "b"}
    assert "d" not in set(f["amostra"]) and "c" not in set(f["amostra"])
    assert DL.MOTIVOS["discordancia"] in f.set_index("amostra").at["a", "motivos"]
    assert DL.MOTIVOS["alta_sem_lesao"] in f.set_index("amostra").at["b", "motivos"]


def test_REQ_LOC_04_monitoramento_alerta_queda():
    import pandas as pd
    import desempenho_local as DL
    linhas = []
    for mes, acerto in (("2026-01-15", 0.95), ("2026-02-15", 0.30)):
        for i in range(60):
            y = int(i < 30)
            linhas.append({"grupo": f"{mes}{i}", "y": y, "data": mes,
                           "score": (0.9 if (y and i < 30 * acerto) else 0.1) if y else 0.1})
    t = DL.por_periodo(pd.DataFrame(linhas), "score", 0.5)
    assert list(t["periodo"]) == ["2026-01", "2026-02"]
    al = DL.alertas(t, 0.80)
    assert len(al) == 1 and al[0].startswith("2026-02")


def test_REQ_LOC_05_comparacao_do_modo_sombra():
    import desempenho_local as DL
    b = _banco_df([
        {"rodada": "r1", "mama": "M1", "imagem": "i1", "versao_sombra": "detector:20260101;achado:20260101",
         "origem_lesao": "detectada pelo modelo", "lesao_real": "sim", "sombra_detectou": "sim", "caixas_sombra": "1",
         "achado_modelo": "Massa/Nódulo", "achado_sombra": "Microcalcificação", "achado_confirmado": "Microcalcificação",
         "exame_confirmacao": BIOPSIA},
        {"rodada": "r1", "mama": "M1", "imagem": "i1", "versao_sombra": "detector:20260101;achado:20260101",
         "origem_lesao": "detectada pelo modelo", "lesao_real": "não", "sombra_detectou": "não", "caixas_sombra": "1"},
    ])
    det = DL.comparar_sombra(b, "detector", "20260101").set_index("medida")
    assert det.at["lesões reais encontradas", "em uso"] == 1.0 and det.at["lesões reais encontradas", "candidato"] == 1.0
    assert det.at["caixas falsas por imagem", "em uso"] == 1.0 and det.at["caixas falsas por imagem", "candidato"] == 0.0
    ach = DL.comparar_sombra(b, "achado", "20260101").set_index("medida")
    assert ach.at["achado (acerto)", "em uso"] == 0.0 and ach.at["achado (acerto)", "candidato"] == 1.0
    assert DL.comparar_sombra(b, "densidade", "20260101").empty


def test_REQ_CAL_02_calibracao_local_por_partes():
    """A calibração local pode ter só a lesão, só a mama ou as duas; a escala do escore da mama não muda."""
    import maria
    cal = json.loads((RAIZ / "modelo/calibracao.json").read_text(encoding="utf-8"))
    so_mama = maria.aplicar_calibracao_local(cal, {"hibrido_gmic": {"alta": 0.5, "media": 0.2, "n": 60}})
    assert so_mama["hibrido_gmic"]["alta"] == 0.5 and so_mama["hibrido_gmic"]["a"] == cal["hibrido_gmic"]["a"]
    assert so_mama["patologia"] == cal["patologia"] and cal["hibrido_gmic"]["alta"] != 0.5   # original intacto
    so_lesao = maria.aplicar_calibracao_local(cal, {"patologia": {"a": 2.0, "b": 0.1}, "prioridade": {"alta": 0.4, "media": 0.1}})
    assert so_lesao["patologia_contexto"]["a"] == 2.0 and so_lesao["prioridade_contexto"]["alta"] == 0.4
    assert so_lesao["hibrido_gmic"]["alta"] == cal["hibrido_gmic"]["alta"]


def test_REQ_RET_02_candidato_em_modo_sombra(tmp_path, monkeypatch):
    """O retreino aprovado vira candidato (não troca o modelo); promover troca e guarda o anterior; descartar apaga."""
    import torch
    import retreinar as R
    for rel in ("modelo_achado/resnet50_multitask_mama_do_zero.pth",):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"w": torch.zeros(1)}, tmp_path / rel)
    monkeypatch.setattr(R, "AQUI", tmp_path)
    monkeypatch.setattr(R, "CANDIDATO", tmp_path / "candidato")
    monkeypatch.setattr(R, "HISTORICO", tmp_path / "historico")
    monkeypatch.setattr(R, "auditar", lambda *a, **k: None)
    import types
    gerados = []                                     # não toca no MANIFESTO.json real
    monkeypatch.setitem(sys.modules, "manifesto", types.SimpleNamespace(gerar=gerados.append))
    rel = "modelo_achado/resnet50_multitask_mama_do_zero.pth"
    R.salvar_candidato("achado", {"w": torch.ones(1)}, rel, "20260101_000000", {}, {}, 12)
    assert torch.load(tmp_path / rel)["w"].item() == 0                    # o modelo em uso NÃO mudou
    assert "achado" in R.candidatos() and R.candidatos()["achado"]["casos_usuario"] == 12
    R.promover("achado")
    assert torch.load(tmp_path / rel)["w"].item() == 1                    # trocou só depois de aprovar
    assert any((p / rel).is_file() for p in (tmp_path / "historico").glob("*"))   # anterior guardado
    assert not R.candidatos() and len(gerados) == 1      # o manifesto é regerado ao trocar o modelo
    R.salvar_candidato("achado", {"w": torch.full((1,), 2.0)}, rel, "20260102_000000", {}, {}, 5)
    R.descartar("achado")
    assert not R.candidatos() and torch.load(tmp_path / rel)["w"].item() == 1
