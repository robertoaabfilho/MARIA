"""
Testes lentos (GPU recomendada, ~1-2 min cada). Rodar com:  MARIA_TESTES_LENTOS=1 python -m pytest testes -q
"""
import csv
import os
import subprocess
import sys
import time

import pytest

from conftest import RAIZ

lento = pytest.mark.skipif(os.environ.get("MARIA_TESTES_LENTOS") != "1", reason="defina MARIA_TESTES_LENTOS=1")


@lento
def test_REQ_REG_01_regressao_dos_exemplos(tmp_path):
    """As previsões nos exemplos são idênticas às de referência (detecta mudança não intencional no modelo)."""
    saida = tmp_path / "r.csv"
    r = subprocess.run([sys.executable, str(RAIZ / "modelo/maria.py"), "--pasta", str(RAIZ / "exemplos"),
                        "--planilha", str(RAIZ / "exemplos/planilha_medico.csv"), "--saida_csv", str(saida),
                        "--sem_mapas"], capture_output=True, text=True,
                       env=dict(os.environ, MARIA_DASHBOARD="1", MARIA_AUDITORIA=str(tmp_path / "a.jsonl")))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    novo = {l["amostra"]: l for l in csv.DictReader(open(saida, encoding="utf-8-sig"))}
    ref = {l["amostra"]: l for l in csv.DictReader(open(RAIZ / "testes/referencia_exemplos.csv", encoding="utf-8"))}
    assert set(novo) == set(ref)
    for a, l in ref.items():
        for c in ("prioridade", "patologia", "achado", "p_maligno", "p_maligno_calibrada", "qualidade"):
            assert novo[a][c] == l[c], f"{a}: {c} mudou ({l[c]} -> {novo[a][c]})"


@lento
def test_REQ_INT_01_02_no_dicom_com_pacs_simulado(tmp_path):
    """Recebe do 'PACS', recusa AE não autorizado, devolve Secondary Capture no mesmo estudo e grava a fila."""
    import json
    import pydicom
    from pynetdicom import AE, evt, AllStoragePresentationContexts, StoragePresentationContexts
    (tmp_path / "pacs").mkdir()
    cfg = {"ae_titulo": "MARIA", "porta": 11122, "ae_permitidos": ["TESTE"],
           "destino": {"ae": "PACSSIM", "host": "127.0.0.1", "porta": 11123}, "espera_segundos": 4,
           "sem_mapas": True, "apagar_recebidos": True, "pasta_entrada": str(tmp_path / "entrada"),
           "pasta_saida": str(tmp_path / "saida")}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    recebidos = []

    def guardar(e):
        recebidos.append(e.dataset)
        return 0
    pacs = AE(ae_title="PACSSIM"); pacs.supported_contexts = AllStoragePresentationContexts
    srv = pacs.start_server(("127.0.0.1", 11123), block=False, evt_handlers=[(evt.EVT_C_STORE, guardar)])
    no = subprocess.Popen([sys.executable, str(RAIZ / "integracao/no_dicom.py"), "--config", str(tmp_path / "cfg.json")],
                          env=dict(os.environ, MARIA_AUDITORIA=str(tmp_path / "aud.jsonl")),
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(4)

        def enviar(ae_titulo, arq):
            ae = AE(ae_title=ae_titulo); ae.requested_contexts = StoragePresentationContexts
            assoc = ae.associate("127.0.0.1", 11122, ae_title="MARIA")
            st = assoc.send_c_store(pydicom.dcmread(str(arq))).Status
            assoc.release()
            return st
        arq = RAIZ / "exemplos/1_dicom_filme_cbis/P_00066_LEFT_CC_completa.dcm"
        assert enviar("INTRUSO", arq) == 0x0124
        assert enviar("TESTE", arq) == 0x0000
        for _ in range(100):
            time.sleep(3)
            if len(recebidos) >= 2:
                break
    finally:
        no.terminate(); no.wait(); srv.shutdown()
    estudo = str(pydicom.dcmread(str(arq), stop_before_pixels=True).StudyInstanceUID)
    assert len(recebidos) >= 2
    assert all(str(d.StudyInstanceUID) == estudo for d in recebidos)
    assert all("MARIA" in str(d.SeriesDescription) for d in recebidos)
    assert (tmp_path / "saida/fila_priorizada.csv").is_file()
    assert not any((tmp_path / "entrada").glob("*/*.dcm"))
