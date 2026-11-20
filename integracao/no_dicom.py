"""
MARIA — nó DICOM para integração com o PACS (fluxo do radiologista).

Funcionamento:
  1. Recebe imagens por C-STORE (o PACS ou o mamógrafo "envia para MARIA"), uma pasta por estudo.
  2. Quando um estudo fica parado por `espera_segundos` (todas as imagens chegaram), roda o maria.py nele.
  3. Devolve ao PACS de destino, NO MESMO ESTUDO da paciente, uma nova série "MARIA - apoio à priorização":
       - uma imagem DICOM (Secondary Capture) por mamografia completa, com as caixas das lesões e a prioridade;
       - uma imagem-resumo com a tabela de todas as lesões (prioridade, P calibrada, achado, qualidade).
     Qualquer visualizador de PACS abre essas imagens.
  4. Atualiza integracao/saida/fila_priorizada.csv (um estudo por linha, com a maior prioridade), que o
     sistema de informação do serviço (RIS) pode ler para ordenar a lista de trabalho.
  5. Apaga as imagens recebidas depois de processar (minimização de dados — LGPD) e registra na auditoria.

Configuração: integracao/config_dicom.json (copie de config_dicom.exemplo.json). Rodar:
    python3 integracao/no_dicom.py                 (deixe rodando; Ctrl+C para parar)
    python3 integracao/no_dicom.py --teste-eco     (C-ECHO no PACS de destino, para testar a conexão)
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pynetdicom import AE, evt, AllStoragePresentationContexts
from pynetdicom.sop_class import SecondaryCaptureImageStorage, Verification

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "modelo"))
import auditoria  # noqa: E402

SERIE = "MARIA - apoio a priorizacao (nao e diagnostico)"
CORES = {"alta": (230, 40, 40), "média": (255, 170, 0), "baixa": (60, 180, 75)}


def carregar_config(caminho):
    cfg = json.loads(Path(caminho).read_text(encoding="utf-8"))
    for chave in ("pasta_entrada", "pasta_saida"):
        p = Path(cfg[chave])
        cfg[chave] = p if p.is_absolute() else (RAIZ / p)
        cfg[chave].mkdir(parents=True, exist_ok=True)
    return cfg


# ----------------------------------------------------------------------------- recepção
class Recepcao:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ultimo = {}                 # StudyInstanceUID -> horário da última imagem
        self.trava = threading.Lock()

    def ao_receber(self, evento):
        ae_chamador = evento.assoc.requestor.ae_title.strip()
        permitidos = self.cfg.get("ae_permitidos") or []
        if permitidos and ae_chamador not in permitidos:
            auditoria.registrar(f"nó DICOM ({ae_chamador})", "envio DICOM recusado (AE não autorizado)", {})
            return 0x0124               # "Refused: Not Authorized"
        ds = evento.dataset
        ds.file_meta = evento.file_meta
        estudo = str(getattr(ds, "StudyInstanceUID", "sem_estudo"))
        if str(getattr(ds, "SeriesDescription", "")) == SERIE:
            return 0x0000               # nossa própria saída voltando: ignora
        destino = self.cfg["pasta_entrada"] / estudo / f"{ds.SOPInstanceUID}.dcm"
        destino.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(destino, enforce_file_format=True)
        with self.trava:
            self.ultimo[estudo] = time.time()
        return 0x0000

    def estudos_prontos(self):
        agora = time.time()
        with self.trava:
            prontos = [e for e, t in self.ultimo.items() if agora - t >= self.cfg.get("espera_segundos", 30)]
            for e in prontos:
                self.ultimo.pop(e)
        return prontos


# ----------------------------------------------------------------------------- saída DICOM
def imagem_cinza(caminho):
    ds = pydicom.dcmread(str(caminho))
    a = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        a = a.max() - a
    lo, hi = np.percentile(a, [1, 99])
    return Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)), ds


def fonte(tamanho):
    for nome in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def secondary_capture(img_rgb, ref, descricao, numero, serie_uid):
    """Monta um DICOM Secondary Capture (RGB) no mesmo estudo/paciente de `ref`."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    for tag in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge", "StudyInstanceUID",
                "StudyDate", "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName", "StudyDescription"):
        if tag in ref:
            setattr(ds, tag, getattr(ref, tag))
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = serie_uid
    ds.Modality = "OT"
    ds.ConversionType = "WSD"
    ds.SeriesDescription = SERIE
    ds.SeriesNumber = 9901
    ds.InstanceNumber = numero
    ds.ImageComments = descricao[:10240]
    ds.Manufacturer = "MARIA (prototipo de pesquisa)"
    ds.ContentDate = time.strftime("%Y%m%d")
    ds.ContentTime = time.strftime("%H%M%S")
    a = np.asarray(img_rgb.convert("RGB"))
    ds.Rows, ds.Columns = a.shape[:2]
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = a.tobytes()
    return ds


def gerar_saidas(resultados, pasta_estudo):
    """resultados: linhas do results.csv do estudo -> lista de datasets Secondary Capture."""
    saidas, numero = [], 1
    arquivos_ref = sorted(pasta_estudo.glob("*.dcm"))
    if not arquivos_ref:
        return saidas
    ref = pydicom.dcmread(str(arquivos_ref[0]), stop_before_pixels=True)
    serie = generate_uid()
    por_imagem = {}
    for r in resultados:
        if r.get("x0") and r.get("imagem"):
            por_imagem.setdefault(r["imagem"], []).append(r)
    for caminho, linhas in por_imagem.items():
        try:
            img, _ = imagem_cinza(caminho)
        except Exception:
            continue
        fator = min(1.0, 2048 / max(img.size))
        img = img.resize((int(img.width * fator), int(img.height * fator))).convert("RGB")
        d = ImageDraw.Draw(img)
        f = fonte(max(14, img.width // 45))
        for r in linhas:
            x0, y0, x1, y1 = (float(r[c]) * fator for c in ("x0", "y0", "x1", "y1"))
            cor = CORES.get(r.get("prioridade", ""), (180, 180, 180))
            d.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=cor, width=max(3, img.width // 300))
            d.text((x0, max(0, y0 - f.size - 10)), f"{r.get('prioridade', '?')} P={r.get('p_maligno_calibrada', '?')}",
                   fill=cor, font=f)
        d.text((10, 10), "MARIA - apoio a priorizacao. NAO e diagnostico.", fill=(255, 255, 0), font=f)
        saidas.append(secondary_capture(img, ref, f"MARIA: {len(linhas)} lesao(oes) marcadas", numero, serie))
        numero += 1
    # imagem-resumo (tabela)
    linhas_txt = ["MARIA - resumo do estudo (apoio a priorizacao; o diagnostico e do radiologista)", ""]
    linhas_txt.append(f"{'lesao':<34}{'prioridade':<12}{'P calib.':<10}{'achado':<20}{'qualidade'}")
    for r in resultados:
        linhas_txt.append(f"{r.get('amostra', '')[:33]:<34}{r.get('prioridade', '')[:11]:<12}"
                          f"{r.get('p_maligno_calibrada', '')[:9]:<10}{r.get('achado', '')[:19]:<20}{r.get('qualidade', '')}")
    linhas_txt += ["", f"versao do modelo: {resultados[0].get('versao_modelo', '?') if resultados else '?'}",
                   "Prototipo de pesquisa. Nao e dispositivo medico registrado."]
    f = fonte(18)
    resumo = Image.new("RGB", (1400, 60 + 26 * len(linhas_txt)), (0, 0, 0))
    d = ImageDraw.Draw(resumo)
    for i, t in enumerate(linhas_txt):
        d.text((20, 20 + 26 * i), t, fill=(255, 255, 255), font=f)
    saidas.append(secondary_capture(resumo, ref, "MARIA: resumo do estudo", numero, serie))
    return saidas


def enviar(cfg, datasets):
    dest = cfg.get("destino")
    if not dest:
        return 0
    ae = AE(ae_title=cfg["ae_titulo"])
    ae.add_requested_context(SecondaryCaptureImageStorage, ExplicitVRLittleEndian)
    assoc = ae.associate(dest["host"], int(dest["porta"]), ae_title=dest["ae"])
    if not assoc.is_established:
        raise ConnectionError(f"não consegui conectar ao PACS {dest['ae']}@{dest['host']}:{dest['porta']}")
    enviados = 0
    try:
        for ds in datasets:
            st = assoc.send_c_store(ds)
            enviados += int(getattr(st, "Status", 0xFFFF) == 0x0000)
    finally:
        assoc.release()
    return enviados


# ----------------------------------------------------------------------------- processamento
def processar_estudo(cfg, estudo):
    pasta = cfg["pasta_entrada"] / estudo
    saida = cfg["pasta_saida"] / estudo
    saida.mkdir(parents=True, exist_ok=True)
    csv_saida = saida / "results.csv"
    cmd = [sys.executable, str(RAIZ / "modelo" / "maria.py"), "--pasta", str(pasta), "--saida_csv", str(csv_saida),
           "--limiar", str(cfg.get("limiar", 0.5))]
    if cfg.get("sem_mapas", False):
        cmd.append("--sem_mapas")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "MARIA_DASHBOARD": "1", "PYTHONIOENCODING": "utf-8"})
    (saida / "log.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not csv_saida.is_file():
        auditoria.registrar("nó DICOM", "estudo com erro no processamento", {"estudo": estudo[-12:]})
        print(f"❌ estudo …{estudo[-12:]}: erro (veja {saida / 'log.txt'})", flush=True)
        return
    with open(csv_saida, encoding="utf-8-sig") as f:
        resultados = list(csv.DictReader(f))
    datasets = gerar_saidas(resultados, pasta)
    for i, ds in enumerate(datasets, 1):
        ds.save_as(saida / f"maria_{i:02d}.dcm", enforce_file_format=True)
    try:
        enviados = enviar(cfg, datasets)
    except Exception as e:
        enviados = 0
        print(f"⚠️  estudo …{estudo[-12:]}: resultado gerado, mas não enviado ao PACS ({e})", flush=True)
    ordem = {"alta": 0, "média": 1, "baixa": 2}
    prios = sorted((r.get("prioridade", "") for r in resultados), key=lambda p: ordem.get(p, 9))
    fila = cfg["pasta_saida"] / "fila_priorizada.csv"
    novo = not fila.is_file()
    ref = pydicom.dcmread(str(next(pasta.glob("*.dcm"))), stop_before_pixels=True)
    with open(fila, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        if novo:
            w.writerow(["processado_em", "accession", "paciente_id", "estudo_uid", "maior_prioridade", "lesoes",
                        "qualidade_nao_usar", "versao_modelo"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), getattr(ref, "AccessionNumber", ""), getattr(ref, "PatientID", ""),
                    estudo, prios[0] if prios else "", len(resultados),
                    sum(r.get("qualidade") == "não usar" for r in resultados),
                    resultados[0].get("versao_modelo", "") if resultados else ""])
    auditoria.registrar("nó DICOM", "estudo processado", {"estudo": estudo[-12:], "lesoes": len(resultados),
                                                          "maior_prioridade": prios[0] if prios else "",
                                                          "enviados_ao_pacs": enviados})
    if cfg.get("apagar_recebidos", True):
        shutil.rmtree(pasta, ignore_errors=True)          # minimização de dados
    print(f"✅ estudo …{estudo[-12:]}: {len(resultados)} lesão(ões), maior prioridade "
          f"'{prios[0] if prios else '-'}', {enviados}/{len(datasets)} imagem(ns) enviada(s) ao PACS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(AQUI / "config_dicom.json"))
    ap.add_argument("--teste-eco", action="store_true")
    a = ap.parse_args()
    if not Path(a.config).is_file():
        raise SystemExit(f"❌ configuração não encontrada: {a.config}\n   Copie integracao/config_dicom.exemplo.json "
                         "para integracao/config_dicom.json e ajuste AE title, porta e PACS de destino.")
    cfg = carregar_config(a.config)
    if a.teste_eco:
        ae = AE(ae_title=cfg["ae_titulo"]); ae.add_requested_context(Verification)
        d = cfg["destino"]
        assoc = ae.associate(d["host"], int(d["porta"]), ae_title=d["ae"])
        ok = assoc.is_established and getattr(assoc.send_c_echo(), "Status", None) == 0x0000
        if assoc.is_established:
            assoc.release()
        raise SystemExit(0 if ok else f"❌ sem resposta do PACS {d['ae']}@{d['host']}:{d['porta']}")

    recepcao = Recepcao(cfg)
    ae = AE(ae_title=cfg["ae_titulo"])
    ae.supported_contexts = AllStoragePresentationContexts
    ae.add_supported_context(Verification)
    servidor = ae.start_server(("0.0.0.0", int(cfg["porta"])), block=False,
                               evt_handlers=[(evt.EVT_C_STORE, recepcao.ao_receber)])
    auditoria.registrar("nó DICOM", "nó DICOM iniciado", {"ae": cfg["ae_titulo"], "porta": cfg["porta"]})
    print(f"🩻 Nó DICOM MARIA ouvindo: AE '{cfg['ae_titulo']}', porta {cfg['porta']}. Ctrl+C para parar.", flush=True)
    try:
        while True:
            for estudo in recepcao.estudos_prontos():
                try:
                    processar_estudo(cfg, estudo)
                except Exception as e:
                    print(f"❌ estudo …{estudo[-12:]}: {e}", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        servidor.shutdown()
        auditoria.registrar("nó DICOM", "nó DICOM parado", {})


if __name__ == "__main__":
    for _fluxo in (sys.stdout, sys.stderr):          # Windows: saída redirecionada usa cp1252 e quebra com ✅/❌
        try:
            _fluxo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
